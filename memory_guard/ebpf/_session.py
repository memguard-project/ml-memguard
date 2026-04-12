"""MemguardBPFSession — graceful context manager wrapping EBPFProbeManager.

:class:`MemguardBPFSession` converts :class:`EBPFProbeManager`'s exception-
raising :meth:`~EBPFProbeManager.load` into a graceful no-op when BPF is
unavailable.  Callers never need to wrap the session in a ``try/except``:

::

    with MemguardBPFSession(on_high=my_callback) as session:
        if session.available:
            logger.info("eBPF probes are active (backend=%s)", session.manager)
        else:
            logger.info("Running without eBPF — poll-based fallback only")
        server.serve_forever()

When ``eBPF`` is unavailable (non-Linux, missing capabilities, no BPF
library, or kernel too old), the context manager logs a single
``logger.warning`` and returns a session where :attr:`available` is ``False``
and :attr:`manager` is ``None``.
"""

from __future__ import annotations

import logging
import threading
from typing import Callable, Optional

from . import EBPFProbeManager
from ._loader import BPFProbeLoader
from .cgroup_memory import MemPressureEvent
from .preemption import PreemptionEvent

logger = logging.getLogger(__name__)


class MemguardBPFSession:
    """Graceful context manager wrapping :class:`EBPFProbeManager`.

    Converts BPF load failures into logged warnings rather than raised
    exceptions.  The calling code continues whether or not eBPF is active.

    Parameters
    ----------
    on_high:
        Callback ``Callable[[MemPressureEvent], None]`` — fired when a
        cgroup crosses its ``memory.high`` soft limit.  Called from the
        background polling thread; must be thread-safe.
    on_oom:
        Callback ``Callable[[MemPressureEvent], None]`` — fired when the
        OOM killer selects a cgroup victim.
    on_preemption:
        Callback ``Callable[[PreemptionEvent], None]`` — fired when the
        watched worker process exits.  Only meaningful when ``worker_pid``
        is also set.
    worker_pid:
        PID of the vLLM / SGLang worker process to watch.  When ``None``
        (default) the preemption probe is not loaded.
    poll_timeout_ms:
        Milliseconds per ``perf_buffer_poll`` call (default 10).
    ebpf_wake:
        Optional :class:`threading.Event` that the polling thread sets on
        every ``LEVEL_HIGH`` event — lets :class:`KVCacheMonitor` wake
        immediately instead of waiting for its next poll tick.
    loader:
        Override the :class:`BPFProbeLoader` used for capability detection.
        Primarily for testing.

    Attributes
    ----------
    available:
        ``True`` only while BPF probes are actually running inside the
        context block.
    manager:
        The live :class:`EBPFProbeManager` instance, or ``None`` when
        unavailable.
    """

    def __init__(
        self,
        on_high:       Optional[Callable[[MemPressureEvent], None]] = None,
        on_oom:        Optional[Callable[[MemPressureEvent], None]] = None,
        on_preemption: Optional[Callable[[PreemptionEvent], None]] = None,
        worker_pid:    Optional[int] = None,
        poll_timeout_ms: int = 10,
        ebpf_wake:     Optional[threading.Event] = None,
        loader:        Optional[BPFProbeLoader] = None,
    ) -> None:
        self._on_high         = on_high
        self._on_oom          = on_oom
        self._on_preemption   = on_preemption
        self._worker_pid      = worker_pid
        self._poll_timeout_ms = poll_timeout_ms
        self._ebpf_wake       = ebpf_wake
        self._loader          = loader or BPFProbeLoader()

        self._manager: Optional[EBPFProbeManager] = None
        self._active:  bool                       = False

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    @property
    def available(self) -> bool:
        """``True`` if BPF probes are actually running inside the ``with`` block."""
        return self._active

    @property
    def manager(self) -> Optional[EBPFProbeManager]:
        """The live :class:`EBPFProbeManager`, or ``None`` when unavailable."""
        return self._manager

    # ------------------------------------------------------------------
    # Context manager protocol
    # ------------------------------------------------------------------

    def __enter__(self) -> "MemguardBPFSession":
        if not self._loader.available:
            logger.warning(
                "[MemguardBPFSession] eBPF unavailable — running without kernel "
                "probes. Reason: %s",
                self._loader.unavailable_reason,
            )
            return self

        mgr = EBPFProbeManager(
            on_high       = self._on_high,
            on_oom        = self._on_oom,
            on_preemption = self._on_preemption,
            worker_pid    = self._worker_pid,
            poll_timeout_ms = self._poll_timeout_ms,
            ebpf_wake     = self._ebpf_wake,
        )
        try:
            mgr.load()
            mgr.start()
            self._manager = mgr
            self._active  = True
            logger.debug(
                "[MemguardBPFSession] probes active (backend=%s)",
                self._loader.backend,
            )
        except Exception as exc:
            logger.warning(
                "[MemguardBPFSession] probe load failed — running without eBPF. "
                "Error: %s",
                exc,
            )
            try:
                mgr.stop()
            except Exception:
                pass

        return self

    def __exit__(self, *_: object) -> None:
        if self._manager is not None:
            try:
                self._manager.stop()
            except Exception as exc:
                logger.debug("[MemguardBPFSession] stop error: %s", exc)
            self._manager = None
        self._active = False

    def __repr__(self) -> str:
        state   = "active"  if self._active else "inactive"
        backend = self._loader.backend if self._active else "none"
        return f"MemguardBPFSession(state={state!r}, backend={backend!r})"
