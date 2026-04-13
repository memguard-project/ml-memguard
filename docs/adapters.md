# Framework Adapters — Reference Guide

*Training adapters added in v0.2.0 (`pip install ml-memguard[hf]` / `pip install ml-memguard[unsloth]`)*  
*Inference serving adapters added in v0.3.0 (`pip install ml-memguard[vllm]` / `pip install ml-memguard[sglang]`)*

---

## Overview

The `memory_guard.adapters` subpackage provides one-call integration with
HuggingFace Transformers and Unsloth.  It reads the model's architecture
automatically so you never have to look up `hidden_size`, `num_heads`, or
`num_layers`.

All adapter symbols are available directly from the top-level package through
lazy imports — `import memory_guard` is safe on a machine without `torch` or
`transformers` installed.

```python
from memory_guard import guard_trainer          # HF Transformers
from memory_guard import guard_unsloth_model   # Unsloth
from memory_guard import guard_sft_trainer     # TRL SFTTrainer
from memory_guard import MemoryGuardCallback   # HF callback (advanced use)

from memory_guard import guard_vllm            # vLLM inference serving
from memory_guard import guard_sglang          # SGLang inference serving
```

---

## `introspect_model(model)`

**Module**: `memory_guard.adapters.base`

Reads the following fields from any HuggingFace-style model without importing
`torch` or `transformers` at call time:

| Field | Source | Fallback |
|---|---|---|
| `hidden_size` | `model.config.hidden_size` | — |
| `num_attention_heads` | `model.config.num_attention_heads` | — |
| `num_hidden_layers` | `model.config.num_hidden_layers` | — |
| `num_key_value_heads` | `model.config.num_key_value_heads` | `num_attention_heads` (MHA models) |
| `model_bits` | `quantization_config.load_in_4bit / load_in_8bit`, then `model.dtype` | 32 |
| `num_parameters` | `sum(p.numel() for p in model.parameters())` | — |

**Bits inference order**:

1. `quantization_config.load_in_4bit = True` → **4**
2. `quantization_config.quant_type` in `{"nf4", "fp4"}` → **4**
3. `quantization_config.load_in_8bit = True` → **8**
4. `model.dtype` contains `"float16"` or `"bfloat16"` → **16**
5. Default → **32**

---

## `optional_import(name, extra)`

**Module**: `memory_guard.adapters.base`

Lazy import helper that raises a clear install hint on missing dependencies:

```python
torch = optional_import("torch", "hf")
# → ImportError: 'torch' is required. Install: pip install ml-memguard[hf]
```

---

## HuggingFace Transformers Adapter

**Install**: `pip install ml-memguard[hf]`  
**Module**: `memory_guard.adapters.huggingface`

### `guard_trainer(trainer, guard=None, **preflight_overrides) -> SafeConfig`

One-call setup for HuggingFace `Trainer`.

1. Calls `introspect_model(trainer.model)` to read architecture and quantization
2. Runs `guard.preflight(...)` with the introspected values
3. Writes `safe.batch_size`, `safe.grad_accumulation`, `safe.grad_checkpoint`
   directly into `trainer.args`
4. Appends `MemoryGuardCallback(guard)` to `trainer.callback_handler.callbacks`
5. Returns the `SafeConfig`

```python
safe = guard_trainer(trainer)
# trainer.args is now patched — call trainer.train() directly
trainer.train()
```

**`preflight_overrides`**: any keyword accepted by `guard.preflight()` overrides
the introspected value.  Common overrides:

| Override | When to use |
|---|---|
| `batch_size=8` | Lock a specific batch size instead of the auto-selected safe value |
| `seq_length=4096` | Non-default sequence length |
| `lora_rank=32` | Fixed LoRA rank (introspection can't detect pre-LoRA intent) |
| `model_bits=16` | Model loaded at different precision than its config reports |

---

### `MemoryGuardCallback(guard)`

`TrainerCallback` subclass that manages runtime memory monitoring throughout
the HuggingFace training loop.

| Hook | What it does |
|---|---|
| `on_train_begin` | Starts `guard.monitor(per_device_train_batch_size)`; resets pending state |
| `on_step_begin` | If the monitor's `current_batch_size` has dropped below `args.per_device_train_batch_size`, records a *pending* downgrade (does **not** mutate `args` here) |
| `on_epoch_begin` | Applies any pending downgrade atomically: sets `args.per_device_train_batch_size = pending`, scales `args.gradient_accumulation_steps` by `old // new` to preserve effective batch size |
| `on_log` | Logs a warning if the latest pressure reading is above `THRESHOLD_WARNING` |
| `on_train_end` | Exits the monitor context; calls `guard.record_result()` for auto-calibration |

**Why epoch boundary?**  HuggingFace `Trainer` pre-builds its `DataLoader` once
per epoch via `get_train_dataloader()`.  Mutating `args.per_device_train_batch_size`
mid-epoch is silently ignored by the active DataLoader.  Deferring to
`on_epoch_begin` ensures the new value is picked up when the DataLoader is
rebuilt.  See [ADR 001](decisions/001-mid-training-downgrade-semantics.md) for
the full rationale.

**Single-epoch / `max_steps` training**: if training ends before an epoch
boundary fires, the pending downgrade is never applied.  The monitor still emits
warnings.  A `on_step_end` flush path may be added in v0.3 if needed.

---

## Unsloth Adapter

**Install**: `pip install ml-memguard[unsloth]`  
**Module**: `memory_guard.adapters.unsloth`

### `guard_unsloth_model(model, guard=None, **preflight_overrides) -> SafeConfig`

Run preflight **before** `FastLanguageModel.get_peft_model` is called.

```python
model, tokenizer = FastLanguageModel.from_pretrained(
    "unsloth/Meta-Llama-3.1-8B-bnb-4bit",
    max_seq_length=2048, load_in_4bit=True,
)

safe = guard_unsloth_model(model)   # ← before get_peft_model

model = FastLanguageModel.get_peft_model(
    model,
    r=safe.lora_rank,
    lora_alpha=safe.lora_rank * 2,
    max_seq_length=safe.seq_length,
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
)
```

Returns a `SafeConfig` with `lora_rank`, `lora_layers`, `seq_length` ready to
thread into `get_peft_model`.

**BnB double-quantization correction**: when `model.config.quantization_config.bnb_4bit_use_double_quant = True`
(Unsloth's default), `guard_unsloth_model` multiplies `num_parameters` by
`_DOUBLE_QUANT_CORRECTION = 0.95` before calling `preflight`.  `model_bits`
stays 4 — it is the actual bit-width.  The 0.95 factor is a conservative proxy
for the ~5 % weight-memory saving from quantizing the quantization constants.
Auto-calibration refines the estimate after 3+ training runs.

See [ADR 002](decisions/002-qlora-double-quant-bits.md) for the full
decision rationale, memory math, and override path.

---

### `guard_sft_trainer(trainer, guard=None, **preflight_overrides) -> SafeConfig`

Identical to `guard_trainer` but named for TRL `SFTTrainer` workflows.

```python
trainer = SFTTrainer(model=model, tokenizer=tokenizer, ...)
guard_sft_trainer(trainer)
trainer.train()
```

Internally delegates to `memory_guard.adapters.huggingface.guard_trainer`.

---

## Full Unsloth Workflow Example

```python
from memory_guard import guard_unsloth_model, guard_sft_trainer
from unsloth import FastLanguageModel
from trl import SFTTrainer, SFTConfig

# 1. Load
model, tokenizer = FastLanguageModel.from_pretrained(
    "unsloth/Meta-Llama-3.1-8B-bnb-4bit",
    max_seq_length=2048,
    load_in_4bit=True,
)

# 2. Preflight — introspects model, auto-downgrades, returns SafeConfig
safe = guard_unsloth_model(model)
# Override specific values if needed:
# safe = guard_unsloth_model(model, seq_length=4096, lora_rank=16)

# 3. Attach LoRA with safe values
model = FastLanguageModel.get_peft_model(
    model,
    r=safe.lora_rank,
    lora_alpha=safe.lora_rank * 2,
    max_seq_length=safe.seq_length,
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
    use_gradient_checkpointing=safe.grad_checkpoint,
)

# 4. Train with mid-training downgrade protection
trainer = SFTTrainer(
    model=model,
    tokenizer=tokenizer,
    train_dataset=dataset,
    args=SFTConfig(
        per_device_train_batch_size=safe.batch_size,
        gradient_accumulation_steps=safe.grad_accumulation,
        max_seq_length=safe.seq_length,
        output_dir="./output",
    ),
)
guard_sft_trainer(trainer)   # patches args + adds MemoryGuardCallback
trainer.train()
```

---

---

## Inference Serving Adapters (v0.3.0)

These adapters integrate with serving frameworks.  They never mutate a running
engine — all load-shedding is delegated to the caller via callbacks on the
`KVCacheMonitor` (ADR 003).

See **[`docs/inference.md`](inference.md)** for the full inference reference guide,
including the `KVCacheMonitor` hook table and end-to-end workflow examples.

### `guard_vllm(llm, guard=None, **preflight_overrides) -> InferenceSafeConfig`

**Install**: `pip install ml-memguard[vllm]`  
**Module**: `memory_guard.adapters.vllm`

```python
from vllm import LLM
from memory_guard import guard_vllm

llm = LLM(model="meta-llama/Meta-Llama-3-8B", ...)

safe = guard_vllm(llm)
print(safe.max_num_seqs)            # pass to vLLM --max-num-seqs
print(safe.gpu_memory_utilization)  # pass to vLLM --gpu-memory-utilization

# Wire load-shedding callbacks before starting
safe.monitor.on_shed_load = lambda u: load_balancer.reduce_weight("primary", 0)

with safe.monitor.session():
    server.serve_forever()
```

**Accepted types**: `vllm.LLM` (has `.llm_engine`), `vllm.AsyncLLMEngine`
(has `.engine`), bare `vllm.LLMEngine`.

**Architecture detection** — reads from `engine.model_config.hf_config`:

| Field | Source |
|---|---|
| `num_kv_heads` | `hf_config.num_key_value_heads` → falls back to `num_attention_heads` |
| `head_dim` | `hf_config.hidden_size // num_attention_heads` |
| `num_layers` | `hf_config.num_hidden_layers` |
| `max_seq_len` | `model_config.max_model_len` |
| `dtype_bytes` | `model_config.dtype` (fp16/bf16 → 2, fp32 → 4, int8 → 1) |
| `model_bits` | `model_config.quantization` (awq/gptq/fp8/awq_marlin/gptq_marlin → 4, bnb/smooth → 8, else 16) |
| `model_params` | `hf_config.num_parameters` → else 12 × H² × L estimate |

**KV cache back-calculation**: when `cache_config.num_gpu_blocks > 0`, the adapter
derives `max_num_seqs` from the actual vLLM pre-allocation:
`blocks_per_seq = ceil(max_seq_len / block_size)`;
`actual_max_seqs = num_gpu_blocks // blocks_per_seq`.
This keeps the preflight estimate and live utilization on the same scale.
`gpu_memory_utilization` is then refined from the measured KV MB vs `available_mb`.

**Poll path**: `engine.scheduler.block_manager.get_num_free_gpu_blocks()` and
`.get_num_total_gpu_blocks()`.  Falls back to null utilization (returns `(0, 1)`)
if the block manager is not accessible, with a one-time warning.

**`preflight_overrides`** — any keyword accepted by `preflight_inference()` overrides
the introspected value.  Common overrides:

| Override | When to use |
|---|---|
| `max_num_seqs=32` | Cap concurrency below the back-calculated value |
| `max_seq_len=4096` | Force a specific sequence length |
| `model_bits=16` | Model loaded at different precision than config reports |

---

### `guard_sglang(engine, guard=None, **preflight_overrides) -> InferenceSafeConfig`

**Install**: `pip install ml-memguard[sglang]`  
**Module**: `memory_guard.adapters.sglang`

```python
import sglang as sgl
from memory_guard import guard_sglang

runtime = sgl.Runtime(model_path="meta-llama/Meta-Llama-3-8B", ...)

safe = guard_sglang(runtime)
print(safe.max_num_seqs)            # pass to --max-running-requests
print(safe.gpu_memory_utilization)  # pass to --mem-fraction-static

safe.monitor.on_shed_load = lambda u: load_balancer.set_weight("primary", 0)

with safe.monitor.session():
    runtime.wait()
```

**Accepted types**: `sglang.Runtime` (unwrapped via `.engine`), bare
`TokenizerManager` or any engine object with `server_args`.

**Architecture detection** — reads from `engine.server_args`:

| Field | Source |
|---|---|
| `max_seq_len` | `server_args.context_length` → `server_args.max_total_tokens` → 8192 |
| `dtype_bytes` | `server_args.dtype` (fp16/bf16 → 2, float32/fp32 → 4, int8 → 1) |
| `model_bits` | `server_args.quantization` (awq/gptq/fp8/marlin → 4, bnb → 8, else 16) |

When `engine.tp_worker.model_runner.model.config` is accessible, `num_kv_heads`,
`num_layers`, and `hidden_size` are read from the HF config directly.  Otherwise
conservative defaults apply (32 layers, 32 kv_heads, 4096 hidden).

**Token pool back-calculation**: `actual_max_seqs = total_token_slots // max_seq_len`
from the actual pool size — same principle as the vLLM block back-calculation.

**Poll path** (tried in order):
1. `engine.token_to_kv_pool.get_available_size()` / `.size` — preferred (SGLang ≥ 0.3.0)
2. `engine.mem_pool.available` / `.size` — older SGLang
3. `engine.scheduler.get_stats().num_used_tokens` / `.num_total_tokens` — fallback
4. Null fallback `(0, 1)` with a one-time warning if nothing is accessible

**Rolling-max smoothing**: SGLang's RadixAttention prefix-cache can evict large KV
blocks at once, causing utilization to drop suddenly.  The adapter wraps the raw
poll with a 3-reading rolling maximum so that a single eviction event does not
prematurely clear a shed-load signal.  The signal clears naturally once all 3
consecutive readings fall below the threshold.

---

## Architectural Decision Records

The design decisions made during v0.2.0 and v0.3.0 are documented in:

- [`docs/decisions/001-mid-training-downgrade-semantics.md`](decisions/001-mid-training-downgrade-semantics.md) — why downgrade is deferred to epoch boundary rather than applied immediately
- [`docs/decisions/002-qlora-double-quant-bits.md`](decisions/002-qlora-double-quant-bits.md) — why `quantization_config` is trusted and a 5 % correction is applied rather than requiring explicit `model_bits`
- [`docs/decisions/003-inference-signals-only.md`](decisions/003-inference-signals-only.md) — why `KVCacheMonitor` fires callbacks only and never mutates the serving engine
