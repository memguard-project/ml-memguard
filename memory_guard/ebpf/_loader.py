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

Container / Kubernetes runtime detection (PR 57):

  ``BPFProbeLoader.__init__`` immediately detects the execution environment
  by inspecting ``KUBERNETES_SERVICE_HOST``, ``/.dockerenv``, and
  ``/proc/1/cgroup``.  This informs the BPF *attachment mode*:

  * **raw_tracepoint** — full ``CAP_BPF + CAP_PERFMON`` available; probes
    attach directly to kernel tracepoints for maximum observability.
  * **cgroup_skb** — running in a rootless container without ``CAP_BPF``;
    cgroup-scoped BPF programs can still observe memory events for the
    current cgroup subtree.
  * **none** — no BPF capability and no supported container runtime; probes
    are skipped and a clear log message explains how to enable them.

All detection is lazy for capabilities (computed on first access of
:attr:`BPFProbeLoader.available` and then cached), but the container runtime
check runs at ``__init__`` time because it is fast and never raises.
The class never raises; all failures are logged as warnings.
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


# ---------------------------------------------------------------------------
# Container / Kubernetes runtime detection (PR 57)
# ---------------------------------------------------------------------------

#: Sentinel file created by Docker inside every container.
_DOCKER_ENV_FILE: str = "/.dockerenv"
#: Linux file that lists the cgroup hierarchy for PID 1.
_PROC_1_CGROUP: str = "/proc/1/cgroup"


def _detect_container_runtime() -> str:
    """Return the container/orchestration runtime for this process.

    Checks in priority order:

    1. ``KUBERNETES_SERVICE_HOST`` env var — present in every Kubernetes pod.
    2. ``/.dockerenv`` file — created by the Docker daemon in every container.
    3. ``/proc/1/cgroup`` content — contains ``"docker"``, ``"containerd"``,
       ``"kubepods"``, or ``"lxc"`` when running inside a container.
    4. Falls back to ``"host"`` when none of the above match.

    Returns one of: ``"kubernetes"``, ``"docker"``, ``"container"``, ``"host"``.
    """
    # 1. Kubernetes pod: KUBERNETES_SERVICE_HOST is always injected
    if os.environ.get("KUBERNETES_SERVICE_HOST"):
        return "kubernetes"

    # 2. Docker: daemon creates /.dockerenv in every container
    if os.path.exists(_DOCKER_ENV_FILE):
        return "docker"

    # 3. Generic container: inspect PID 1 cgroup membership
    try:
        with open(_PROC_1_CGROUP) as fh:
            content = fh.read()
        if any(kw in content for kw in ("docker", "containerd", "kubepods", "lxc")):
            return "container"
    except Exception:
        pass

    return "host"


def _bpf_attachment_mode(runtime: str, has_cap: bool) -> str:
    """Return the appropriate BPF attachment mode for *runtime* + *has_cap*.

    Decision table:

    =============================  =========  ====================
    Runtime                        has_cap    Attachment mode
    =============================  =========  ====================
    any                            True       ``raw_tracepoint``
    kubernetes / docker / container False      ``cgroup_skb``
    host                           False      ``none``
    =============================  =========  ====================

    Parameters
    ----------
    runtime:
        Value returned by :func:`_detect_container_runtime`.
    has_cap:
        ``True`` when the process has ``CAP_BPF + CAP_PERFMON`` or
        ``CAP_SYS_ADMIN``.

    Returns
    -------
    str
        ``"raw_tracepoint"`` | ``"cgroup_skb"`` | ``"none"``
    """
    if has_cap:
        return "raw_tracepoint"
    if runtime in ("kubernetes", "docker", "container"):
        return "cgroup_skb"
    return "none"


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
    runtime:
        Container/orchestration runtime detected at init time:
        ``"kubernetes"``, ``"docker"``, ``"container"``, or ``"host"``.
    attachment_mode:
        BPF program attachment mode selected by capability + runtime:
        ``"raw_tracepoint"``, ``"cgroup_skb"``, or ``"none"``.
    """

    def __init__(self) -> None:
        self._available: Optional[bool] = None
        self._backend:   Optional[str]  = None
        self._reason:    str            = ""
        # Container runtime detected at init (lightweight, no blocking I/O)
        self._runtime:         str = _detect_container_runtime()
        self._attachment_mode: str = "none"

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

    @property
    def runtime(self) -> str:
        """Container/orchestration runtime detected at init time.

        One of ``"kubernetes"``, ``"docker"``, ``"container"``, ``"host"``.
        Detected at :meth:`__init__` time — never cached lazily.
        """
        return self._runtime

    @property
    def attachment_mode(self) -> str:
        """BPF program attachment mode selected by runtime + capability detection.

        Computed lazily alongside :attr:`available`.

        Returns
        -------
        str
            * ``"raw_tracepoint"`` — full ``CAP_BPF + CAP_PERFMON`` available;
              probes attach directly to kernel tracepoints.
            * ``"cgroup_skb"`` — running in a container without ``CAP_BPF``;
              cgroup-scoped BPF programs are used instead.
            * ``"none"`` — no BPF capability and no container runtime; probes
              are disabled and a log message explains how to enable them.
        """
        _ = self.available   # trigger detection
        return self._attachment_mode

    # ------------------------------------------------------------------
    # Public methods
    # ------------------------------------------------------------------

    def check_capabilities(self) -> Tuple[bool, str]:
        """Return ``(ok, reason)`` — ``reason`` is an empty string when ``ok`` is ``True``."""
        ok = self.available
        return ok, self._reason

    def __repr__(self) -> str:
        if self._available is None:
            return (
                f"BPFProbeLoader(not-yet-probed, runtime={self._runtime!r})"
            )
        if self._available:
            return (
                f"BPFProbeLoader(available=True, backend={self._backend!r}, "
                f"runtime={self._runtime!r}, "
                f"attachment_mode={self._attachment_mode!r})"
            )
        return (
            f"BPFProbeLoader(available=False, runtime={self._runtime!r}, "
            f"attachment_mode={self._attachment_mode!r}, "
            f"reason={self._reason!r})"
        )

    # ------------------------------------------------------------------
    # Detection logic
    # ------------------------------------------------------------------

    def _detect(self) -> Tuple[bool, str]:
        """Run all checks in order and return ``(available, reason)``."""

        # 1. Platform
        if sys.platform != "linux":
            self._attachment_mode = "none"
            return False, (
                f"eBPF requires Linux (current platform: {sys.platform!r})"
            )

        # 2. Kernel version
        major, minor = _kernel_version()
        if major < 5 or (major == 5 and minor < 8):
            self._attachment_mode = "none"
            return False, (
                f"eBPF memory.high tracepoint requires kernel ≥ 5.8 "
                f"(detected {major}.{minor})"
            )

        # 3. Capabilities — determine attachment mode immediately so the
        #    runtime/capability combination is always reflected even when
        #    the overall detection fails at a later step.
        has_cap = _has_cap_bpf()
        self._attachment_mode = _bpf_attachment_mode(self._runtime, has_cap)

        if not has_cap:
            if self._attachment_mode == "cgroup_skb":
                logger.info(
                    "[BPFProbeLoader] Container runtime detected (%r) — no "
                    "CAP_BPF or CAP_SYS_ADMIN. Attachment mode: cgroup_skb. "
                    "Add 'CAP_BPF' and 'CAP_PERFMON' to the sidecar "
                    "securityContext (ebpf.enabled: true in Helm values) "
                    "to unlock raw_tracepoint mode and full observability.",
                    self._runtime,
                )
            else:
                logger.debug(
                    "[BPFProbeLoader] eBPF probes disabled — no CAP_BPF, "
                    "CAP_PERFMON, or CAP_SYS_ADMIN and not running inside a "
                    "recognised container runtime. To enable: run as root, "
                    "or set ebpf.enabled: true in Helm values / "
                    "spec.ebpf.enabled: true in MemGuardPolicy.",
                )
            return False, (
                "eBPF requires CAP_BPF + CAP_PERFMON or CAP_SYS_ADMIN "
                "(re-run as root or grant capabilities with setcap / "
                "Helm ebpf.enabled: true)"
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
