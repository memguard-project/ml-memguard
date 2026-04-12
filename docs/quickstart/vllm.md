# vLLM + memguard in 3 Minutes

Stop `No available memory for cache blocks` before it takes down your serving traffic.

```bash
pip install ml-memguard[vllm]
```

That is the only install command you need. No extra services, no config files.

---

## The Two-Line Setup

Add two lines to your existing vLLM launch script:

```python
from memory_guard import guard_vllm
from vllm import AsyncLLMEngine, AsyncEngineArgs

args   = AsyncEngineArgs(model="meta-llama/Llama-3.1-8B-Instruct", gpu_memory_utilization=0.90)
engine = AsyncLLMEngine.from_engine_args(args)

# Line 1: Calculate the largest safe concurrent batch and start the KV monitor
safe = guard_vllm(engine)

# Line 2: Wire the load-shed signal — reduce upstream weight when KV cache hits 90%
safe.monitor.on_shed_load = lambda u: print(f"[memguard] KV cache at {u:.0%} — shedding load")

# Start your server with the safe config
with safe.monitor.session():
    # safe.max_num_seqs → pass to your server's --max-num-seqs
    run_server(max_num_seqs=safe.max_num_seqs)
```

`guard_vllm` reads your model's architecture directly from `engine.engine.model_config.hf_config`
— no hidden_dim, no num_layers, no manual math required.

---

## What You Get

Running `guard_vllm` prints a pre-flight report before the server starts:

```
[memguard] InferenceSafeConfig
  model:            meta-llama/Llama-3.1-8B-Instruct
  gpu_total:        24,564 MB  (RTX 4090)
  weights_mb:       15,258 MB
  kv_budget:         7,051 MB
  max_num_seqs:          16   ← binary-searched to fit
  max_seq_len:        4,096
  estimated_peak:   23,309 MB
  budget:           22,107 MB
  status:           FITS
```

Then the background monitor runs silently. When the KV cache fills:

```
[memguard] WARNING  — KV cache at 80% (0.803)  [2026-04-12 14:22:07]
[memguard] SHED_LOAD — KV cache at 92% (0.924) [2026-04-12 14:22:19]
```

Without memguard, the next event would be:

```
RuntimeError: No available memory for the cache blocks.
  Try increasing gpu_memory_utilization or decreasing max_model_len.
INFO:     Application shutdown complete.
```

Your server is dead. With memguard, the `on_shed_load` callback fires and your load balancer
stops routing new requests to this replica — existing requests finish cleanly, and the server
keeps running.

---

## All Callbacks

```python
safe = guard_vllm(engine)

# Fires at 80% KV cache utilization — early warning
safe.monitor.on_warning = lambda util: metrics.gauge("kvcache.util", util)

# Fires at 92% KV cache utilization — stop routing new requests here
safe.monitor.on_shed_load = lambda util: load_balancer.reduce_weight(replica_id, weight=0)

# Optional: custom thresholds
safe.monitor.warning_threshold  = 0.80   # default
safe.monitor.shed_load_threshold = 0.92   # default
```

The monitor never touches the vLLM engine. It only emits signals — your code decides what to do.

---

## CLI: pre-flight only (no Python changes needed)

Check what `max_num_seqs` is safe for a model before writing a single line of code:

```bash
python -m memory_guard.preflight \
    --model meta-llama/Llama-3.1-8B-Instruct \
    --gpu-memory-utilization 0.90 \
    --max-seq-len 4096
```

Output:
```
model:         meta-llama/Llama-3.1-8B-Instruct
gpu_total:     24,564 MB
weights_mb:    15,258 MB
kv_budget:      7,051 MB
max_num_seqs:      16
status:        FITS
```

Pass `--max-seq-len 32768` for long-context workloads and see the number drop accordingly.

---

## Sidecar: Kubernetes-native load-shedding (no code changes to vLLM)

If you cannot modify your vLLM launch code, run the memguard sidecar alongside your pod.
The sidecar polls vLLM's Prometheus `/metrics` endpoint and exposes `/readyz`:

```bash
pip install 'ml-memguard[cloud]'
python -m memory_guard.sidecar \
    --vllm-url http://localhost:8000 \
    --port 8001
```

Wire Kubernetes readiness probe to `/readyz`:

```yaml
readinessProbe:
  httpGet:
    path: /readyz
    port: 8001
  periodSeconds: 10
  failureThreshold: 1
```

When KV cache exceeds the OOM threshold, `/readyz` returns `503`. Kubernetes removes the pod
from the Service endpoint set — no new traffic, zero code changes to vLLM.

See [`examples/vllm-quickstart/`](../../examples/vllm-quickstart/) for a complete
`docker compose` setup with both containers wired together.

---

## Common Issues

**`guard_vllm` returns `max_num_seqs=1`**

Your model is too large for the GPU budget at the requested sequence length. Try:
- Lower `--max-seq-len` (e.g. 4096 instead of 32768)
- Lower `--gpu-memory-utilization` to reflect your actual available VRAM
- Use a quantized model (AWQ / GPTQ / INT8)

**KV cache hits 100% immediately**

You have long-context requests arriving faster than they complete. The `max_num_seqs` was
safe at launch; the request mix changed. Lower `max_num_seqs` manually or set a tighter
`shed_load_threshold` (e.g. 0.80) to shed earlier.

**Monitor fires but server still crashes**

The `on_shed_load` callback fires but your load balancer isn't stopping traffic in time.
Ensure your load balancer acts on the callback synchronously, not asynchronously. For
immediate protection without a load balancer, pair with `VLLMWatchdog`:

```python
from memory_guard.inference_monitor import VLLMWatchdog

watchdog = VLLMWatchdog(vllm_url="http://localhost:8000", restart_cmd=["./start_vllm.sh"])
watchdog.start()
```

The watchdog detects when vLLM goes down and restarts it automatically.

---

## Next Steps

- [SGLang quick-start](../quickstart/sglang.md) — same two lines, works with RadixAttention
- [Kubernetes operator guide](../kubernetes.md) — CRD-based deployment with auto-injection
- [Fleet policy sync](../cloud.md) — learn optimal configs from your own run history
- [Full API reference](../inference.md)
