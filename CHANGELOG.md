# Changelog

## [0.2.0] - 2026-04-10

### Added — Framework Adapters

- **`memory_guard.adapters` subpackage** (pure-Python, zero new hard dependencies)
  - `introspect_model(model)` — reads `hidden_size`, `num_attention_heads`,
    `num_hidden_layers`, `num_key_value_heads`, `model_bits`, and `num_parameters`
    from any HuggingFace-style model without importing torch or transformers
  - `optional_import(name, extra)` — lazy import helper that raises a clear
    `pip install ml-memguard[<extra>]` hint on missing dependencies

- **HuggingFace Transformers adapter** (`pip install ml-memguard[hf]`)
  - `guard_trainer(trainer, guard=None, **preflight_overrides)` — one-call setup:
    introspects `trainer.model`, runs `preflight()`, writes the safe config to
    `trainer.args`, and appends `MemoryGuardCallback`
  - `MemoryGuardCallback(TrainerCallback)` — memory monitoring for the HF
    training loop:
    - `on_train_begin` starts `guard.monitor(per_device_train_batch_size)`
    - `on_step_begin` records a pending batch-size downgrade when the monitor
      signals pressure; sets `control.should_log` to flush a warning entry
    - `on_epoch_begin` applies any pending downgrade atomically at the epoch
      boundary, scaling `gradient_accumulation_steps` by `old_bs // new_bs`
      so the effective batch size is preserved
    - `on_log` surfaces sustained pressure warnings through the Trainer logger
    - `on_train_end` stops the monitor and calls `guard.record_result()` for
      auto-calibration

- **Unsloth adapter** (`pip install ml-memguard[unsloth]`)
  - `guard_unsloth_model(model, guard=None, **preflight_overrides)` — run
    preflight *before* `FastLanguageModel.get_peft_model`; returns `SafeConfig`
    with `lora_rank`, `lora_layers`, `seq_length` ready to thread in
  - `guard_sft_trainer(trainer, guard=None, **preflight_overrides)` — identical
    to `guard_trainer` but named for TRL `SFTTrainer` workflows
  - BnB double-quantization detection (`bnb_4bit_use_double_quant=True`): applies
    a documented 5 % correction to `num_parameters`; `model_bits` stays 4;
    auto-calibration refines the estimate after 3+ runs

- **Lazy package exports** — all adapter symbols registered via `__getattr__` in
  `memory_guard/__init__.py` so `import memory_guard` is safe on a torch-free
  machine
- **New extras** in `pyproject.toml`:
  `hf = ["transformers>=4.30", "torch>=2.0", "accelerate>=1.1.0"]`,
  `unsloth = ["unsloth", "transformers>=4.30", "torch>=2.0", "accelerate>=1.1.0"]`

### Changed

- README `With HuggingFace Transformers` example replaced with the one-call
  `guard_trainer` version
- README new **Framework Adapters** section: introspection field table,
  `preflight_overrides` guide, QLoRA double-quant note
- README **API Reference** extended with all four adapter functions

### Tests

- 43 new unit tests across `test_adapters_base.py`, `test_adapters_huggingface.py`,
  `test_adapters_unsloth.py`, and `test_hf_mid_training_downgrade.py`
- `tests/test_adapters_smoke.py` — integration smoke test: `guard_trainer` +
  `MemoryGuardCallback` survive 2 real training steps on distilgpt2 (skipped
  unless transformers is installed; run with `pip install ml-memguard[hf]`)

---

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
