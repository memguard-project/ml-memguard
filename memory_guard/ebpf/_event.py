"""Unified eBPF event dataclass shared across all memguard probes.

All probes (cgroup memory, preemption, page fault, mmap) write into a single
:class:`MemguardBPFEvent` wire format so downstream consumers — VLLMWatchdog,
KVCacheMonitor, the cloud telemetry uploader — never need to know which probe
generated a given signal.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Any


# ---------------------------------------------------------------------------
# Event types
# ---------------------------------------------------------------------------

#: ``cgroup memory.high`` soft-limit crossed — kernel throttling begins.
EVENT_MEMORY_HIGH    = "memory_high"
#: OOM killer invoked — process termination is imminent.
EVENT_OOM_KILL       = "oom_kill"
#: Monitored worker process exited (silent-kill detection).
EVENT_PREEMPTION     = "preemption"
#: Page-fault rate spike in the monitored process (swap pressure).
EVENT_PAGE_FAULT     = "page_fault"
#: Anonymous mmap / brk growth in the monitored process.
EVENT_MMAP_GROWTH    = "mmap_growth"


@dataclass
class MemguardBPFEvent:
    """Unified event emitted by all memguard eBPF probes.

    Attributes
    ----------
    ts_ns:
        Kernel monotonic timestamp in nanoseconds (``bpf_ktime_get_ns()``).
        Use for ordering and velocity computation; not wall-clock time.
    event_type:
        One of the ``EVENT_*`` constants defined in this module.
    pressure_bytes:
        Bytes over the relevant limit (``memory.high`` watermark,
        allocation size, or growth delta).  Zero when not applicable
        (e.g. ``EVENT_PREEMPTION``).
    pid:
        PID of the process that triggered the event.
    cgroup_id:
        Cgroup path (``/kubepods/pod-abc/…``) or empty string when the
        cgroup path is unavailable from the BPF context (e.g. kprobe events).
    extra:
        Probe-specific fields (exit_code for preemption, fault_address for
        page_fault, etc.).  Optional; defaults to empty dict.
    """

    ts_ns:          int
    event_type:     str
    pressure_bytes: int
    pid:            int
    cgroup_id:      str
    extra:          Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Return a JSON-serialisable dict of all fields."""
        return {
            "ts_ns":          self.ts_ns,
            "event_type":     self.event_type,
            "pressure_bytes": self.pressure_bytes,
            "pid":            self.pid,
            "cgroup_id":      self.cgroup_id,
            **self.extra,
        }
