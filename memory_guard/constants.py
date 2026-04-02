"""Named constants with documented rationale.

Every tunable number in memory-guard is defined here with an
explanation of where it comes from. If a value is a best-guess
rather than a measured quantity, that's stated explicitly.
"""

# Safety Ratios

SAFETY_RATIO_DEFAULT = 0.80
"""Use 80% of available memory as the training budget.

Not derived from measurement. This is a conservative engineering
default chosen to leave headroom for OS overhead, background
processes, and memory fragmentation. Users can override via
safety_ratio parameter.
"""

# Estimation

OVERHEAD_RATIO_TRAINING = 0.25
"""25% proportional overhead added to training memory estimate.

Accounts for: memory allocator fragmentation, communication buffers,
intermediate tensors, and metadata. Raised from 20% to 25% after
third-party benchmarks showed 31-44% under-estimation with the
original value. Not derived from measurement — engineering default.
The auto-calibration system corrects this further.
"""

FIXED_OVERHEAD_MB = 400
"""400 MB fixed overhead for framework runtime, Metal shader compilation,
Python interpreter, and allocator bookkeeping.

This is independent of model size — even an empty MLX program uses
~200-400 MB. Without this, small-model estimates are off by 40%+
because the proportional overhead (25% of a 500MB model = 125MB)
doesn't cover the fixed runtime cost.

Estimated at 200-500 MB based on one M4 Max measurement (actual
overhead was ~400 MB). NOT validated on M1 or M2. Using 400 MB as
a conservative default. Calibration corrects this per-device.
"""

OVERHEAD_RATIO_INFERENCE = 0.15
"""15% overhead for inference (lower than training).

Inference has no optimizer states or gradient buffers, so there's
less fragmentation. Slightly lower than training overhead.
Best-guess, correctable via calibration.
"""

LAZY_EVAL_DISCOUNT = 0.80
"""20% activation memory reduction for lazy-evaluation frameworks (MLX).

MLX fuses operations and defers materialization, reducing peak
activation memory vs eager frameworks. The 20% figure is a
best-guess informed by MLX documentation and our single measured
data point (Qwen3.5-9B on M4 Max, 5.6% estimation error).
NOT measured across multiple models or devices. Calibration
corrects this.

Ref: https://developer.apple.com/videos/play/wwdc2025/315/
"""

SWAP_CREDIT_RATIO = 0.50
"""Count 50% of available swap toward the memory budget.

Swap (macOS compressor, Linux swap partition) provides additional
headroom but at significant performance cost. Using 100% would
avoid crashes but cause severe throughput degradation from
swap thrashing. Using 0% would be overly conservative on systems
with swap enabled. 50% is a compromise — not measured.
"""

# macOS

MACOS_RECLAIMABLE_DISCOUNT = 0.85
"""Use 85% of (total - active - wired) as available on macOS.

macOS can reclaim inactive and purgeable pages, but not all of
them are instantly available: dirty inactive pages require I/O.
The 15% discount accounts for this. Chosen to be conservative
after observing that our estimate was 5.6% under actual peak
on one M4 Max test. NOT calibrated across devices.

Ref: https://github.com/giampaolo/psutil/issues/1277
"""

# Linux Containers

CGROUP_HIGH_DISCOUNT = 0.90
"""Use 90% of memory.high as the effective cgroup limit.

The Linux kernel documentation states that memory.high "can be
exceeded" under concurrent allocation bursts, with observed
overshoots of ~10% in CockroachDB and DuckDB production
deployments. The 90% discount prevents treating memory.high
as a hard boundary when it's actually a soft throttle.

Refs:
  https://facebookmicrosites.github.io/cgroup2/docs/memory-controller.html
  https://github.com/cockroachdb/cockroach/issues/114774
  https://github.com/duckdb/duckdb/issues/15080
  https://www.kernel.org/doc/Documentation/cgroup-v2.txt
"""

PSI_CRITICAL_THRESHOLD = 25.0
"""PSI avg10 value (percentage) considered critical memory pressure.

Linux Pressure Stall Information reports the percentage of time
processes were stalled waiting for memory in the last 10 seconds.
At 25%, one quarter of time is stalled — training throughput is
significantly degraded. Facebook's cgroup2 documentation recommends
acting at 20-30% PSI levels. Previous value of 50% was too generous.

Ref: https://facebookmicrosites.github.io/cgroup2/docs/memory-controller.html
"""

# Runtime Monitor

PRESSURE_THRESHOLD_WARNING = 0.70
"""Log warning but don't downgrade. Engineering default, not derived from measurement."""

PRESSURE_THRESHOLD_CRITICAL = 0.85
"""Halve batch size. Engineering default, not derived from measurement."""

PRESSURE_THRESHOLD_EMERGENCY = 0.92
"""Halve batch size immediately. Engineering default, not derived from measurement.
Users can override all thresholds via RuntimeMonitor constructor."""

MONITOR_POLL_INTERVAL = 5.0
"""Seconds between memory pressure polls. Engineering default."""

MONITOR_MAX_DOWNGRADES = 3
"""Maximum automatic batch size reductions per training session.
Engineering default — prevents runaway downgrades to batch_size=1."""

MONITOR_COOLDOWN_SECONDS = 30.0
"""Minimum seconds between consecutive downgrades. Engineering default."""

MLX_LEAK_GROWTH_THRESHOLD_MB = 500
"""Trigger MLX memory leak alert after 500MB monotonic growth over 6 polls.

Based on the pattern described in
https://github.com/ml-explore/mlx-examples/issues/1262
where active memory rises continuously until crash. 500MB over
~30 seconds (6 x 5s polls) indicates a likely leak rather than
normal allocation variance.
"""

# Calibration

QLORA_DOUBLE_QUANT_OVERHEAD_BYTES = 0.0625
"""0.5 bits per parameter (0.0625 bytes) for NF4 double quantization.

QLoRA stores quantization scales and zero points for each group
of weights. Double quantization quantizes these scales too,
adding ~0.5 bits/param overhead. This is a mathematical constant
derived from the NF4 quantization scheme, not a tunable.

Ref: Dettmers et al., "QLoRA", NeurIPS 2023
https://arxiv.org/abs/2305.14314
"""

CALIBRATION_MIN_POINTS = 3
"""Minimum data points before applying calibration correction.

Engineering default. With fewer than 3 points, the median is
unreliable. The formula is used as-is until enough data accumulates.
"""

CALIBRATION_MAX_POINTS = 50
"""Maximum stored calibration points (FIFO, oldest dropped).
Engineering default."""

# Fallback

FALLBACK_MEMORY_MB = 8192
"""8GB fallback when memory detection fails entirely.

Conservative engineering default.
If this is wrong, the estimator will over-downgrade (safe) rather
than crash (dangerous).
"""

FALLBACK_AVAILABLE_RATIO = 0.60
"""Assume 60% of total RAM is available when detection fails.

Used when vm_stat, /proc/meminfo, and GlobalMemoryStatusEx all
fail. Conservative: assumes 40% is used by OS and other processes.
"""

FALLBACK_PRESSURE = 0.5
"""Return 0.5 (moderate) when memory pressure detection fails entirely.

Engineering default. Neither optimistic (0.0) nor pessimistic (1.0).
Won't trigger downgrades but will be visible in pressure_history.
"""
