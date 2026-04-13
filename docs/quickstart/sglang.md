# SGLang + memguard in 3 Minutes

Stop KV cache exhaustion and RadixAttention dead-branch OOM before they take down your
serving traffic — on CUDA or Apple Silicon.

```bash
pip install ml-memguard[sglang]
```

---

## The Two-Line Setup

```python
from memory_guard import guard_sglang
import sglang as sgl

runtime = sgl.Runtime(
    model_path="meta-llama/Llama-3.1-8B-Instruct",
    mem_fraction_static=0.85,    # leave headroom for RadixAttention tree overhead
)

# Line 1: Calculate the largest safe max_num_seqs and start the KV monitor
safe = guard_sglang(runtime)

# Line 2: Wire the load-shed signal — fires when rolling-max KV cache hits 92%
safe.monitor.on_shed_load = lambda u: lb.reduce_weight(replica_id, 0)

with safe.monitor.session():
    runtime.wait()
```

`guard_sglang` reads `server_args.context_length`, `server_args.mem_fraction_static`,
and the running `token_to_kv_pool` size directly from the runtime object — no manual
architecture parameters required.

---

## Why SGLang Needs Different Settings Than vLLM

SGLang's RadixAttention prefix cache introduces two OOM failure modes that vLLM's paged
attention does not have:

**1. Dead branches from reasoning models.**
DeepSeek-R1, QwQ, and similar models insert `<think>` tokens into the radix tree at
completion time. Multi-turn conversations never match these branches again (the user
strips thinking tokens from the next prompt), so each turn leaves a permanently dead
branch that consumes 1.3–1.6 GB of KV cache per conversation. LRU eviction cannot reclaim
a branch until all downstream nodes are also pruned — with 50 concurrent sessions this
can accumulate 65–80 GB of wasted GPU memory.

**2. Eviction-driven utilization drops.**
When a cached prefix is freed, RadixAttention releases its KV token slots, causing
reported utilization to drop suddenly — e.g. from 88% to 62% in one poll cycle.
Without smoothing, this drop would reset any cooldown timer and delay the next
load-shed signal by several polls, allowing the server to crash before the signal fires.

**What memguard does differently for SGLang:**

- Applies **3-reading rolling-maximum** to the raw utilization: reports
  `max(last_3_readings)` instead of the current value, so transient eviction drops
  do not falsely signal recovery.
- Tracks `eviction_rate` — if eviction is high at moderate utilization, dead branches
  are the likely cause and the signal fires earlier.
- Recommends `mem_fraction_static=0.85` (not the SGLang default of 0.88) when
  RadixAttention tree node overhead is significant for your workload.

---

## What the Pre-flight Prints

```
[memguard] InferenceSafeConfig
  model:              meta-llama/Llama-3.1-8B-Instruct
  gpu_total:          24,564 MB  (RTX 4090)
  weights_mb:         15,258 MB
  kv_budget:           5,586 MB  (85% utilization, RadixAttention headroom reserved)
  max_num_seqs:            12   ← binary-searched to fit
  max_seq_len:          4,096
  estimated_peak:      22,979 MB
  budget:              20,879 MB
  status:              FITS

SGLang CLI equivalent:
  python -m sglang.launch_server \
    --model-path meta-llama/Llama-3.1-8B-Instruct \
    --max-running-requests 12 \
    --mem-fraction-static 0.85
```

---

## All Callbacks

```python
safe = guard_sglang(runtime)

# 80% rolling-max — early warning (may lag real pressure by ~15 s due to smoothing)
safe.monitor.on_warning = lambda u: metrics.gauge("sglang.kvcache.util", u)

# 92% rolling-max — stop routing new requests here
safe.monitor.on_shed_load = lambda u: load_balancer.reduce_weight(replica_id, 0)

# Tighten for reasoning model workloads where dead branches accumulate fast
safe.monitor.warning_threshold   = 0.75   # earlier warning for <think>-heavy traffic
safe.monitor.shed_load_threshold = 0.85   # shed sooner before dead branches fill the tree
```

---

## Apple Silicon (Metal backend)

SGLang's Metal backend serves models via Apple's MLX framework on M-series chips.
The API is identical — memguard detects the Metal backend automatically and uses
`mlx.core.get_active_memory()` instead of CUDA counters.

```bash
pip install ml-memguard[sglang,apple]

python -m sglang.launch_server \
    --model-path mlx-community/Meta-Llama-3-8B-Instruct-4bit \
    --device metal \
    --port 30000
```

```python
safe = guard_sglang(runtime)

# IMPORTANT on Apple Silicon: lower the threshold.
# macOS OOM is silent — no exception, no warning.
# When unified memory is exhausted the process freezes, then dies.
# Shed load at 80%, not 92%, to stay well clear of that cliff.
safe.monitor.shed_load_threshold = 0.80
safe.monitor.on_shed_load = lambda u: lb.reduce_weight(replica_id, 0)
```

memguard is the only KV cache monitoring tool that works on Apple Silicon SGLang
deployments. All other monitoring tools are CUDA-only.

---

## Kubernetes Sidecar (no code changes to SGLang)

The memguard sidecar works against SGLang's OpenAI-compatible `/metrics` endpoint
(port 30000 by default):

```yaml
containers:
  - name: memguard-sidecar
    image: python:3.11-slim
    command:
      - sh
      - -c
      - |
        pip install ml-memguard -q &&
        python -m memory_guard.sidecar \
          --vllm-url http://localhost:30000 \
          --port 8001 \
          --smoothing-window 3
    readinessProbe:
      httpGet:
        path: /readyz
        port: 8001
      periodSeconds: 10
      failureThreshold: 1
```

The `--smoothing-window 3` flag enables RadixAttention-aware rolling-max smoothing in
the sidecar path.

---

## Common Issues

**Pre-flight recommends `max_num_seqs=1` or `status: OOM`**

SGLang's `--mem-fraction-static` is a stricter budget than vLLM's
`--gpu-memory-utilization`. Try:
- Lower `--mem-fraction-static` to `0.80` to give the pre-flight more budget to work with
- Reduce `--max-seq-len` (e.g. 2048 instead of 4096)
- Use a quantized model (`mlx-community/` 4-bit variants on Apple Silicon)

**on_shed_load never fires even though SGLang crashes**

SGLang's rolling-max smoothing can delay the signal by up to `3 × poll_interval` seconds.
Reduce `smoothing_window` to `1` for faster reaction, at the cost of false-recovery
signals from prefix evictions:

```python
safe.monitor.smoothing_window = 1   # faster, noisier
```

**CUDA OOM on same hardware where vLLM works** (issue #12496)

SGLang reserves more memory upfront for the RadixAttention tree than vLLM's paged
attention. If vLLM runs fine at `--gpu-memory-utilization 0.90`, start SGLang at
`--mem-fraction-static 0.82` and let `guard_sglang()` binary-search from there.

---

## Next Steps

- [vLLM quick-start](../quickstart/vllm.md) — paged attention, no rolling-max needed
- [Fleet policy sync](../cloud.md) — learn optimal configs from your own run history
- [Full API reference](../inference.md)
- [Runnable example](../../examples/sglang_monitor.py)
