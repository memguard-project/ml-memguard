# Changelog

## [0.3.0] - 2026-04-10

### Added — Inference Memory Monitoring

- **`estimate_serving_memory()`** — KV cache ceiling estimator for inference
  serving workloads.  Computes the worst-case memory required when all
  `max_num_seqs` sequences are at full `max_seq_len` length:
  `2 × num_layers × num_kv_heads × head_dim × max_seq_len × max_num_seqs × dtype_bytes`.
  GQA-aware (`num_kv_heads` rather than `num_heads`).  Returns an
  `InferenceServingEstimate` dataclass with per-component breakdown and a
  `.fits_in(budget_mb)` helper.

- **`MemoryGuard.preflight_inference()`** — binary search over `max_num_seqs`
  to find the largest value that fits within the memory budget.  Returns an
  `InferenceSafeConfig` with `max_num_seqs`, `max_seq_len`, `gpu_memory_utilization`,
  and a `monitor` field (see below) ready to pass to vLLM / SGLang CLI flags.

- **`InferenceSafeConfig.monitor`** — `KVCacheMonitor` attached by the adapter
  functions below.  Unstarted on return; use `safe.monitor.start()` or
  `with safe.monitor.session(): ...` when ready to serve.

- **`KVCacheMonitor`** — background-thread KV cache utilization monitor for
  inference serving.  Polls a caller-supplied `poll_fn: () → (used, total)`,
  fires `on_warning(u)` at ≥ 80 % and `on_shed_load(u)` at ≥ 92 %
  utilization.  Shed-load takes priority over warning.  Both callbacks are
  subject to a per-level cooldown (default 30 s).  The monitor never reads or
  writes any attribute of the serving engine (ADR 003 — signals only, no
  engine mutation).  Use via `monitor.session()` context manager or explicit
  `start()` / `stop()`.

- **vLLM adapter** (`pip install ml-memguard[vllm]`)
  - `guard_vllm(llm, guard=None, **preflight_overrides) -> InferenceSafeConfig`
    accepts `vllm.LLM`, `vllm.AsyncLLMEngine`, or a bare `vllm.LLMEngine`.
    Reads `model_config.hf_config` for architecture params.
  - Back-calculates `max_num_seqs` from `cache_config.num_gpu_blocks`:
    `blocks_per_seq = ceil(max_seq_len / block_size)`;
    `actual_max_seqs = num_gpu_blocks // blocks_per_seq`.
    Keeps the preflight estimate and live utilization on the same scale.
  - Refines `gpu_memory_utilization` from the actual measured KV MB vs
    available memory.
  - Wires `KVCacheMonitor` poll_fn to
    `scheduler.block_manager.get_num_free_gpu_blocks()` and
    `.get_num_total_gpu_blocks()`.
  - Quantization detection: AWQ / GPTQ / AWQ-Marlin / GPTQ-Marlin / FP8 /
    SqueezeLLM → 4 bits; BitsAndBytes / SmoothQuant → 8 bits; else 16 bits.
  - Returns `InferenceSafeConfig` with `safe.monitor` set (unstarted).

- **SGLang adapter** (`pip install ml-memguard[sglang]`)
  - `guard_sglang(engine, guard=None, **preflight_overrides) -> InferenceSafeConfig`
    accepts `sglang.Runtime` (unwrapped via `.engine`) or any bare engine
    object with `server_args`.
  - Reads `server_args.{context_length, dtype, quantization}` and walks
    `tp_worker.model_runner.model.config` for HF architecture fields.
  - Back-calculates `max_num_seqs` from the actual token pool:
    `actual_max_seqs = total_token_slots // max_seq_len`.
  - Polls `engine.token_to_kv_pool.get_available_size()` (SGLang ≥ 0.3.0,
    preferred) or `engine.mem_pool.available` (older SGLang), with a
    `scheduler.get_stats()` fallback and a null fallback with a one-time
    warning.
  - **3-reading rolling-max smoothing**: SGLang's RadixAttention prefix-cache
    can evict large KV blocks suddenly, causing utilization to drop below the
    shed-load threshold.  The `poll_fn` tracks the last 3 raw utilization
    values and reports their maximum, suppressing transient drops without
    delaying recovery detection once pressure genuinely recedes.
  - Returns `InferenceSafeConfig` with `safe.monitor` set (unstarted).

- **ADR 003** (`docs/decisions/003-inference-signals-only.md`) — documents the
  signals-only design: why vLLM `max_num_seqs` cannot be hot-reconfigured,
  why mutation is invasive, and how the callback design composes with nginx,
  Envoy, PagerDuty, and Kubernetes autoscalers.

- **`docs/inference.md`** — new reference guide covering `estimate_serving_memory`,
  `preflight_inference`, `InferenceSafeConfig`, the `KVCacheMonitor` hook table
  (poll_fn contract, thresholds, callbacks, cooldown), and full vLLM / SGLang
  workflow examples.

- **New extras in `pyproject.toml`**:
  `vllm = ["vllm>=0.4"]`, `sglang = ["sglang>=0.3"]`.
  Both are included in `all`.

### Tests

- 122 new unit tests across `tests/test_inference_estimator.py`,
  `tests/test_kv_cache_monitor.py`, `tests/test_vllm_adapter.py`, and
  `tests/test_sglang_adapter.py`.  No vLLM or SGLang installation required —
  all framework objects are simulated with `MagicMock`.  Total suite: 325 tests.

---

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
