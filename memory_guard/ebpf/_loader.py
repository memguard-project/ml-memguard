"""BPF backend detector and kernel capability checker for memguard probes.

``BPFProbeLoader`` auto-detects the available BPF backend at runtime:

  1. **bcc** (primary) — runtime C compilation via libbcc.
  2. **libbpf-python** (fallback) — pre-compiled BPF skeleton loader.
  3. **unavailable** — graceful no-op; no ``ImportError`` raised.

Capability checks performed (Linux only):

  * Kernel ≥ 5.8 — required for ``cgroup:memory_high`` tracepoint and
    ``CAP_BPF`` / ``CAP_PERFMON`` split (older kernels need ``CAP_SYS_ADMIN``).
  * ``CAP_BPF`` + ``CAP_PERFMON`` **or** ``CAP_SYS_ADMIN`` in the effective
    capability set.
  * cgroup v2 mounted at ``/sys/fs/cgroup`` (sentinel:
    ``/sys/fs/cgroup/cgroup.controllers``).

All detection is lazy — computed on first access of :attr:`BPFProbeLoader.available`
and then cached.  The class never raises; all failures are logged as warnings.
"""

from __future__ import annotations

import logging
import os
import platform
import sys
from typing import Optional, Tuple

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Linux capability bit positions (defined in <uapi/linux/capability.h>)
# ---------------------------------------------------------------------------

_CAP_SYS_ADMIN: int = 21  #: Root-equivalent omnibus capability
_CAP_PERFMON:   int = 38  #: Access performance monitoring (Linux 5.8+)
_CAP_BPF:       int = 39  #: Load BPF programs (Linux 5.8+)


# ---------------------------------------------------------------------------
# Low-level helpers (module-level so tests can patch them individually)
# ---------------------------------------------------------------------------

def _kernel_version() -> Tuple[int, int]:
    """Return ``(major, minor)`` of the running Linux kernel.

    Returns ``(0, 0)`` on parse failure or non-Linux platforms.
    """
    try:
        release = platform.release()          # e.g. "5.15.0-76-generic"
        parts   = release.split(".")
        return int(parts[0]), int(parts[1])
    except Exception:
        return 0, 0


def _has_cap_bpf() -> bool:
    """Return ``True`` if the process has the required BPF capabilities.

    Accepts either ``CAP_BPF + CAP_PERFMON`` (Linux ≥ 5.8) or the legacy
    ``CAP_SYS_ADMIN`` grant.  Always returns ``True`` for root (uid 0).
    """
    try:
        if os.getuid() == 0:
            return True
    except AttributeError:
        pass  # Windows — getuid not available

    try:
        with open("/proc/self/status") as fh:
            for line in fh:
                if line.startswith("CapEff:"):
                    cap_eff       = int(line.split(":")[1].strip(), 16)
                    cap_sys_admin = (cap_eff >> _CAP_SYS_ADMIN) & 1
                    cap_bpf       = (cap_eff >> _CAP_BPF)       & 1
                    cap_perfmon   = (cap_eff >> _CAP_PERFMON)    & 1
                    return bool(cap_sys_admin or (cap_bpf and cap_perfmon))
    except Exception:
        pass
    return False


def _cgroupv2_mounted() -> bool:
    """Return ``True`` if cgroup v2 is mounted.

    The sentinel ``/sys/fs/cgroup/cgroup.controllers`` exists only on
    cgroup v2 (unified hierarchy) — absent on pure cgroup v1 systems.
    """
    return os.path.exists("/sys/fs/cgroup/cgroup.controllers")


def _detect_backend() -> Optional[str]:
    """Return the name of the first importable BPF library, or ``None``.

    Tries in preference order:
      1. ``bcc`` — BPF Compiler Collection (runtime C compilation)
      2. ``libbpf`` — libbpf-python bindings (pre-compiled skeleton)
    """
    try:
        import bcc  # type: ignore[import]  # noqa: F401
        return "bcc"
    except ImportError:
        pass

    try:
        import libbpf  # type: ignore[import]  # noqa: F401
        return "libbpf"
    except ImportError:
        pass

    return None


# ---------------------------------------------------------------------------
# BPFProbeLoader
# ---------------------------------------------------------------------------

class BPFProbeLoader:
    """Auto-detect BPF backend availability and verify kernel capabilities.

    All checks run once on first access of :attr:`available` and are
    cached for the lifetime of the instance.  The loader never raises —
    failures are surfaced through :attr:`available` (``False``) and
    :attr:`unavailable_reason`.

    Typical usage::

        loader = BPFProbeLoader()
        if not loader.available:
            logger.warning("eBPF disabled: %s", loader.unavailable_reason)
        else:
            logger.info("eBPF backend: %s", loader.backend)

    Attributes
    ----------
    available:
        ``True`` if eBPF probes can be loaded on this system.
    backend:
        ``"bcc"`` | ``"libbpf"`` | ``None`` — the detected BPF library.
    unavailable_reason:
        Human-readable explanation when :attr:`available` is ``False``.
    """

    def __init__(self) -> None:
        self._available: Optional[bool] = None
        self._backend:   Optional[str]  = None
        self._reason:    str            = ""

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def available(self) -> bool:
        """``True`` if BPF probes can be loaded on this system."""
        if self._available is None:
            self._available, self._reason = self._detect()
        return self._available

    @property
    def backend(self) -> Optional[str]:
        """The detected BPF backend: ``'bcc'``, ``'libbpf'``, or ``None``."""
        _ = self.available          # trigger detection if not yet run
        return self._backend

    @property
    def unavailable_reason(self) -> str:
        """Human-readable reason BPF is unavailable; empty string when available."""
        _ = self.available
        return self._reason

    # ------------------------------------------------------------------
    # Public methods
    # ------------------------------------------------------------------

    def check_capabilities(self) -> Tuple[bool, str]:
        """Return ``(ok, reason)`` — ``reason`` is an empty string when ``ok`` is ``True``."""
        ok = self.available
        return ok, self._reason

    def __repr__(self) -> str:
        if self._available is None:
            return "BPFProbeLoader(not-yet-probed)"
        if self._available:
            return f"BPFProbeLoader(available=True, backend={self._backend!r})"
        return f"BPFProbeLoader(available=False, reason={self._reason!r})"

    # ------------------------------------------------------------------
    # Detection logic
    # ------------------------------------------------------------------

    def _detect(self) -> Tuple[bool, str]:
        """Run all checks in order and return ``(available, reason)``."""

        # 1. Platform
        if sys.platform != "linux":
            return False, (
                f"eBPF requires Linux (current platform: {sys.platform!r})"
            )

        # 2. Kernel version
        major, minor = _kernel_version()
        if major < 5 or (major == 5 and minor < 8):
            return False, (
                f"eBPF memory.high tracepoint requires kernel ≥ 5.8 "
                f"(detected {major}.{minor})"
            )

        # 3. Capabilities
        if not _has_cap_bpf():
            return False, (
                "eBPF requires CAP_BPF + CAP_PERFMON or CAP_SYS_ADMIN "
                "(re-run as root or grant capabilities with setcap)"
            )

        # 4. cgroup v2
        if not _cgroupv2_mounted():
            return False, (
                "cgroup v2 not mounted at /sys/fs/cgroup "
                "(required for cgroup:memory_high tracepoint)"
            )

        # 5. BPF library
        backend = _detect_backend()
        if backend is None:
            return False, (
                "No BPF library found — install with: "
                "pip install 'ml-memguard[ebpf]'"
            )

        self._backend = backend
        return True, ""
