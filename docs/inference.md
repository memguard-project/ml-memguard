# Inference Memory Monitoring — Reference Guide

*Added in v0.3.0 (`pip install ml-memguard[vllm]` / `pip install ml-memguard[sglang]`)*

---

## Overview

The inference side of memory-guard provides three components:

1. **`estimate_serving_memory()`** — static KV cache ceiling estimator
2. **`MemoryGuard.preflight_inference()`** — preflight check with binary-search downgrade
3. **`KVCacheMonitor`** — background-thread runtime monitor with threshold callbacks

The adapter functions `guard_vllm` and `guard_sglang` wire all three together in a
single call.  See [`docs/adapters.md`](adapters.md) for adapter-specific details and
the signals-only design contract.

---

## `estimate_serving_memory()`

**Module**: `memory_guard.estimator`

Computes the **worst-case** KV cache memory when all `max_num_seqs` concurrent
sequences are at maximum length simultaneously:

```
KV cache = 2 × num_layers × num_kv_heads × head_dim
           × max_seq_len × max_num_seqs × dtype_bytes
```

The factor of 2 accounts for the separate key and value tensors per layer.
`num_kv_heads` (not `num_attention_heads`) is used — GQA models like Llama-3
have far fewer KV heads than attention heads, so using `num_attention_heads`
would overestimate by the GQA ratio.

```python
from memory_guard import estimate_serving_memory

est = estimate_serving_memory(
    max_num_seqs=64,
    max_seq_len=8192,
    num_kv_heads=8,
    head_dim=128,
    num_layers=32,
    dtype_bytes=2,        # fp16 / bf16
    model_params=8e9,
    model_bits=16,
    hidden_dim=4096,
)

print(est.total_mb)          # total: weights + KV cache
print(est.kv_cache_mb)       # KV cache component only
print(est.fits_in(40_000))   # True if total_mb ≤ 40 000 MB
```

### `InferenceServingEstimate` fields

| Field | Type | Description |
|---|---|---|
| `total_mb` | `float` | Weights + KV cache, in MB |
| `kv_cache_mb` | `float` | KV cache contribution only |
| `weights_mb` | `float` | Model weights at the given `model_bits` |
| `max_num_seqs` | `int` | The `max_num_seqs` value used |
| `max_seq_len` | `int` | The `max_seq_len` value used |
| `fits_in(budget_mb)` | `bool` | `total_mb <= budget_mb` |

---

## `MemoryGuard.preflight_inference()`

**Module**: `memory_guard.guard`

Runs a binary search over `max_num_seqs` to find the largest value whose
estimated `total_mb` fits within the memory budget.

```python
from memory_guard import MemoryGuard

guard = MemoryGuard.auto()

safe = guard.preflight_inference(
    model_params=8e9,
    model_bits=16,
    num_kv_heads=8,
    head_dim=128,
    num_layers=32,
    max_seq_len=8192,
    dtype_bytes=2,
    hidden_dim=4096,
    max_num_seqs=256,      # starting point; may be downgraded
)

print(safe.max_num_seqs)           # largest value that fits
print(safe.gpu_memory_utilization) # KV MB / available_mb, capped at 0.95
print(safe.fits)                   # True unless even max_num_seqs=1 doesn't fit
```

### `InferenceSafeConfig` fields

| Field | Type | Description |
|---|---|---|
| `max_num_seqs` | `int` | Safe maximum concurrent sequences |
| `max_seq_len` | `int` | Maximum sequence length used in the estimate |
| `gpu_memory_utilization` | `float` | KV MB / available_mb, capped at 0.95 |
| `estimate` | `InferenceServingEstimate` | Full breakdown at the safe `max_num_seqs` |
| `budget_mb` | `float` | Memory budget used for the search |
| `available_mb` | `float` | Total available GPU memory |
| `fits` | `bool` | False only when even `max_num_seqs=1` exceeds budget |
| `changes` | `list[str]` | Human-readable description of any downgrade applied |
| `monitor` | `KVCacheMonitor \| None` | Set by adapter functions; `None` when calling `preflight_inference` directly |

---

## `KVCacheMonitor`

**Module**: `memory_guard.inference_monitor`

A background daemon thread that polls a caller-supplied `poll_fn` and fires
threshold callbacks.  The monitor is **framework-agnostic** — it never imports
or references vLLM or SGLang.  All coupling to the serving engine is through
the `poll_fn` closure.

### Constructor

```python
from memory_guard import KVCacheMonitor

monitor = KVCacheMonitor(
    poll_fn=lambda: (used_blocks, total_blocks),
    poll_interval=5.0,          # seconds between polls
    on_warning=None,            # callable[[float], None] or None
    on_shed_load=None,          # callable[[float], None] or None
    on_log=None,                # callable[[str], None] — default: logger.warning
    cooldown_seconds=30.0,      # min gap between repeated callback firings
    history_size=60,            # utilization readings retained
)
```

### `poll_fn` contract

`poll_fn` must:

- Accept **no arguments**
- Return a `(used: int, total: int)` tuple of plain integers
- Be **thread-safe** — it is called from the background thread, not the main thread
- Return `(0, 1)` (not `(0, 0)`) as a safe null value when no data is available

The utilization passed to callbacks is `used / total` (0.0–1.0).

### Thresholds and callbacks

| Level | Threshold | Callback | Priority |
|---|---|---|---|
| Warning | ≥ 80 % | `on_warning(utilization)` | Lower — not fired when shed-load fires |
| Shed-load | ≥ 92 % | `on_shed_load(utilization)` | Higher — exclusive above 92 % |

Both callbacks receive the raw utilization value (float, 0.0–1.0) as their
only argument.  Each callback fires **at most once per `cooldown_seconds`**,
preventing log-spam and webhook floods during sustained pressure.

Callbacks are also assignable after construction:

```python
safe.monitor.on_warning = lambda u: logger.warning("KV cache %.0f%%", u * 100)
safe.monitor.on_shed_load = lambda u: load_balancer.reduce_concurrency()
```

### Lifecycle

```python
# Option A — context manager (preferred)
with safe.monitor.session():
    server.serve_forever()
# monitor is automatically stopped when the block exits

# Option B — explicit start / stop
safe.monitor.start()
try:
    server.serve_forever()
finally:
    safe.monitor.stop()
```

`session()` returns the monitor itself as the context value, so you can read
`current_utilization` and `utilization_history` from within the block:

```python
with safe.monitor.session() as mon:
    while True:
        handle_request()
        if mon.current_utilization > 0.7:
            throttle_new_connections()
```

### Properties

| Property | Type | Description |
|---|---|---|
| `is_running` | `bool` | True while the background thread is alive |
| `current_utilization` | `float` | Latest poll reading (0.0 if none yet) |
| `utilization_history` | `list[float]` | All retained readings, oldest first |

---

## Full vLLM Workflow Example

```python
from vllm import LLM
from vllm.entrypoints.api_server import app
from memory_guard import guard_vllm

# 1. Start the engine
llm = LLM(
    model="meta-llama/Meta-Llama-3-8B-Instruct",
    tensor_parallel_size=1,
)

# 2. Preflight — introspects engine, back-calculates from num_gpu_blocks,
#    returns InferenceSafeConfig with safe.monitor wired
safe = guard_vllm(llm)

print(safe)
# InferenceSafeConfig (FITS):
#   max_num_seqs:            128
#   max_seq_len:             8192
#   gpu_memory_utilization:  0.87
#   ...

# 3. Wire load-shedding callbacks
safe.monitor.on_warning = lambda u: print(f"[warn] KV cache {u:.1%}")
safe.monitor.on_shed_load = lambda u: (
    print(f"[alert] Shedding load — KV cache {u:.1%}")
    # or: post to PagerDuty, set Envoy cluster weight to 0, etc.
)

# 4. Start serving with the safe configuration
with safe.monitor.session():
    # The monitor polls vLLM's block manager every 5 s in the background.
    # It fires on_shed_load if KV cache ≥ 92 %, on_warning if ≥ 80 %.
    # It never touches the engine.
    app.run(
        host="0.0.0.0",
        port=8000,
        # vLLM flags sourced from safe config:
        # --max-num-seqs safe.max_num_seqs
        # --gpu-memory-utilization safe.gpu_memory_utilization
    )
```

**Override introspected values** via `**preflight_overrides`:

```python
# Cap concurrency lower than vLLM's allocated blocks suggest
safe = guard_vllm(llm, max_num_seqs=32)

# Force a different precision (e.g. model loaded in fp8 but config says fp16)
safe = guard_vllm(llm, model_bits=8)
```

---

## Full SGLang Workflow Example

```python
import sglang as sgl
from memory_guard import guard_sglang

# 1. Start the runtime
runtime = sgl.Runtime(
    model_path="meta-llama/Meta-Llama-3-8B-Instruct",
    tp_size=1,
)
sgl.set_default_backend(runtime)

# 2. Preflight — introspects engine.server_args and token_to_kv_pool,
#    returns InferenceSafeConfig with safe.monitor wired and smoothed
safe = guard_sglang(runtime)

print(safe.max_num_seqs)            # pass to --max-running-requests
print(safe.gpu_memory_utilization)  # pass to --mem-fraction-static

# 3. Wire load-shedding callbacks
safe.monitor.on_shed_load = lambda u: (
    print(f"[alert] KV cache pressure {u:.1%} — reducing load")
    # post webhook, update load balancer, etc.
)

# 4. Serve
with safe.monitor.session():
    # The monitor polls engine.token_to_kv_pool every 5 s.
    # A 3-reading rolling maximum suppresses transient drops from
    # RadixAttention prefix-cache evictions.
    runtime.wait()
```

**Rolling-max smoothing explained**: when SGLang's RadixAttention evicts a
cached prefix, utilization can drop from 95 % to 40 % in one poll.  Without
smoothing, this would clear the shed-load signal before a cooldown fires,
delaying the next alert.  The rolling max holds the high-water mark for 3
consecutive polls so alerts only clear once pressure genuinely recedes.

---

## Signals-Only Design (ADR 003)

> **`KVCacheMonitor` fires callbacks.  It never sets attributes or calls
> methods on the serving engine.**

The callbacks receive a single `float` (utilization 0.0–1.0).  What you do
with that signal is entirely up to your application:

- **Log** it to structured telemetry
- **Alert** via PagerDuty, Slack, or a metrics counter
- **Shed load** by removing the instance from a load-balancer pool
- **Scale out** by calling a Kubernetes autoscaler API

This design means memory-guard composes cleanly with any deployment topology.
It does not hard-code a load-shedding strategy or require knowledge of your
infrastructure.

See [`docs/decisions/003-inference-signals-only.md`](decisions/003-inference-signals-only.md)
for the full rationale, including why vLLM's `max_num_seqs` cannot be
hot-reconfigured and why engine mutation would be brittle.
