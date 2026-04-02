# Changelog

## [0.1.0] - 2026-04-01

### Core
- Proactive memory estimation for training and inference
- Auto-downgrade in quality-preserving order (grad checkpoint, batch size, seq length, rank, layers)
- `MemoryGuard` unified API: `preflight()` + `monitor()` + `record_result()`
- `ModelSpec` presets for Llama, Mistral, Qwen, Phi, Mixtral, DeepSeek-MoE, LLaVA

### Estimation
- Per-projection activation buffers (Q/K/V/O) per HyC-LoRA and LoRA-FA research
- FlashAttention-aware: O(n) attention memory vs O(n^2) standard
- GQA-aware KV cache (uses `num_kv_heads`, not `num_heads`)
- MoE routing buffers and active expert FFN activations
- Multi-modal encoder memory (vision/audio)
- Full fine-tuning, LoRA, QLoRA (double quantization), DoRA support
- MLX lazy evaluation discount (20% activation reduction)
- Swap headroom credit (50% of available swap added to budget)

### Platform Support
- macOS: Mach `host_statistics` + `sysctlbyname` via ctypes (zero subprocess calls)
- macOS: ARM64 ABI-compliant `argtypes` on all ctypes calls (CPython #42880)
- macOS: `total - (active + wired) * 0.85` available memory formula
- macOS: MLX `mx.metal.get_active_memory()` ground-truth in runtime monitor
- Linux: PSI (`/proc/pressure/memory`), cgroups v1/v2
- Linux: `memory.high` preferred over `memory.max` (CockroachDB/DuckDB finding)
- Linux: Full cgroup hierarchy walk for nested containers (K8s, systemd slices)
- Linux: `memory.high` 90% discount for overshoot under concurrent allocations
- Linux: Docker, Podman, Kubernetes, systemd-nspawn, LXC detection
- Windows: `GlobalMemoryStatusEx` via ctypes
- CUDA: `torch.cuda.mem_get_info`, OOM catch-and-retry, binary search batch finder
- ROCm/HIP: Detection via `torch.version.hip`

### Runtime Monitoring
- Background thread with configurable thresholds (warning 70%, critical 85%, emergency 92%)
- MLX Metal memory leak detection (monotonic growth pattern from mlx-examples#1262)
- Cooldown between downgrades (30s default)
- Pressure history (last 60 readings)
- Thread-safe: `_mach_lock` around all ctypes Mach kernel calls

### Calibration
- Persistent calibration store (`~/.memory-guard/calibration.json`)
- Median correction factor (robust to outliers, requires 3+ data points)
- Auto-reads peak from `mx.metal.get_peak_memory()` or `torch.cuda.max_memory_allocated()`

### Quality
- `py.typed` marker (PEP 561)
- 100+ tests across estimation, platform, downgrade, calibration, thread safety
- Zero external dependencies (ctypes-only for macOS/Windows, /proc for Linux)
- Optional dependencies: `torch` (CUDA), `mlx` (Apple Silicon), `psutil` (fallback)
