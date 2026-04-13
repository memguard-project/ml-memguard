"""vllm/v1/metrics/kv_metrics.py

Rate-based KV cache telemetry signals for the OTel Gen AI semantic
conventions (gen_ai.server.kv_cache.* and gen_ai.server.scheduler.*).

This module is only imported when VLLM_OTEL_KV_METRICS_ENABLED=true.
It has no Prometheus dependency — PrometheusStatLogger owns the gauge
objects and calls update() to get fresh values on each scheduler tick.

Three signals are computed from raw scheduler counters:

  eviction_rate
    KV cache block evictions per second.
    Derived from the length of kv_cache_eviction_events emitted by the
    scheduler each step.  Each element is one evicted block.

  allocation_velocity_mbps
    KV cache growth in MB/s over the last scheduler interval.
    Positive during active prefill; zero or near-zero when sequences
    are completing faster than new ones arrive.
    Returned as MB/s when kv_block_size_mb > 0, otherwise as blocks/s
    (directionally correct for OOM prediction even without block size).

  preemption_rate
    Sequence preemptions per second.
    One preemption suspends an entire sequence and evicts all of its
    KV blocks (N block evictions per preemption, N = seq_len / block_size).
    Distinct from eviction_rate — one preemption causes N evictions.

OTel spec: opentelemetry-specification/semantic-conventions#NNNN
Patch: docs/upstream-prs/vllm/0001-emit-otel-gen-ai-kv-cache-metrics.patch
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import TYPE_CHECKING, List, Optional

if TYPE_CHECKING:
    from vllm.config import VllmConfig
    from vllm.v1.metrics.stats import KVCacheEvictionEvent


@dataclass(frozen=True)
class KVCacheRates:
    """Per-interval rate signals ready for Prometheus gauge.set().

    All values are non-negative.  A value of 0.0 means the signal was
    not measured this interval (first call) or was genuinely zero.
    """

    # Block evictions per second
    eviction_rate: float

    # KV cache growth rate in MB/s (positive = growing)
    allocation_velocity_mbps: float

    # Sequence preemptions per second
    preemption_rate: float


class KVCacheMetricsCollector:
    """Compute rate signals from cumulative vLLM scheduler counters.

    Called once per PrometheusStatLogger.record() invocation — every
    scheduler step (typically every 1–5 s depending on request rate).
    Maintains running state across calls to compute finite differences.

    Args:
        total_gpu_blocks:  Total GPU KV cache blocks at engine startup.
                           Used to convert Δ(usage_fraction) → Δ(blocks).
                           Set to 0 to disable allocation velocity (returns
                           blocks/s without the MB conversion).
        kv_block_size_mb:  Size of one KV cache block in MB.
                           When > 0, allocation_velocity is in MB/s.
                           When 0.0, velocity is in blocks/s.
    """

    def __init__(
        self,
        total_gpu_blocks: int = 0,
        kv_block_size_mb: float = 0.0,
    ) -> None:
        self._total_gpu_blocks  = total_gpu_blocks
        self._kv_block_size_mb  = kv_block_size_mb
        self._prev_usage_frac:  Optional[float] = None
        self._prev_ts:          float           = time.monotonic()

    def update(
        self,
        kv_cache_usage_frac: float,
        eviction_events: "List[KVCacheEvictionEvent]",
        num_preemptions_delta: int,
    ) -> KVCacheRates:
        """Compute rates for the current scheduler step.

        Args:
            kv_cache_usage_frac:    Fraction of GPU KV cache blocks in use
                                    [0.0, 1.0].  From SchedulerStats.kv_cache_usage.
            eviction_events:        KVCacheEvictionEvent list from the current
                                    scheduler step.  len() gives the raw eviction
                                    count for this interval.
            num_preemptions_delta:  Preemptions during this step (incremental,
                                    not cumulative).  From IterationStats.

        Returns:
            KVCacheRates.  On the first call all rates are 0.0 (no previous
            state to diff against).
        """
        now     = time.monotonic()
        elapsed = max(now - self._prev_ts, 1e-6)  # guard zero-division on first call

        # ---- eviction rate -----------------------------------------------
        # Each KVCacheEvictionEvent represents one evicted KV cache block.
        eviction_rate = len(eviction_events) / elapsed

        # ---- allocation velocity -----------------------------------------
        # Positive Δusage × total_blocks × block_size → MB grown this interval.
        # We clamp to 0 because a negative delta (blocks freed) is not growth.
        alloc_vel = 0.0
        if self._prev_usage_frac is not None and self._total_gpu_blocks > 0:
            delta_frac   = kv_cache_usage_frac - self._prev_usage_frac
            delta_blocks = delta_frac * self._total_gpu_blocks
            blocks_per_s = delta_blocks / elapsed
            if self._kv_block_size_mb > 0:
                alloc_vel = max(0.0, blocks_per_s * self._kv_block_size_mb)
            else:
                alloc_vel = max(0.0, blocks_per_s)

        # ---- preemption rate ---------------------------------------------
        preemption_rate = max(0, num_preemptions_delta) / elapsed

        # Advance state
        self._prev_usage_frac = kv_cache_usage_frac
        self._prev_ts         = now

        return KVCacheRates(
            eviction_rate=eviction_rate,
            allocation_velocity_mbps=alloc_vel,
            preemption_rate=preemption_rate,
        )

    @classmethod
    def from_engine_config(cls, vllm_config: "VllmConfig") -> "KVCacheMetricsCollector":
        """Construct from a VllmConfig — extracts block count and size.

        Falls back to (0, 0.0) if config fields are unavailable so the
        collector degrades gracefully rather than raising at startup.
        """
        try:
            cache_config     = vllm_config.cache_config
            total_gpu_blocks = getattr(cache_config, "num_gpu_blocks", 0) or 0
            # block_size_bytes_approx_mb is a convenience attribute added by
            # CacheConfig to expose the per-block MB footprint.  It is not
            # guaranteed to exist in all vLLM versions — fall back to 0.0
            # (blocks/s mode) when absent.
            kv_block_size_mb = float(
                getattr(cache_config, "block_size_bytes_approx_mb", 0.0) or 0.0
            )
        except Exception:
            total_gpu_blocks = 0
            kv_block_size_mb = 0.0

        return cls(
            total_gpu_blocks=total_gpu_blocks,
            kv_block_size_mb=kv_block_size_mb,
        )
