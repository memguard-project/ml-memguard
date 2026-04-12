# Draft Replies for SGLang GitHub Issues

Post these after the SGLang PR is open. Replace `{PR_URL}` with the actual PR link.

---

## Issue #22373 — Reasoning model thinking tokens pollute radix cache with unreachable entries

> **URL:** https://github.com/sgl-project/sglang/issues/22373

```markdown
This is exactly the dead-branch problem — each `<think>` token sequence creates a radix
tree node that multi-turn conversations never match again, and the LRU eviction policy
can't reclaim a non-leaf node until all downstream branches are pruned. With 50+
concurrent reasoning sessions you're looking at 65-80 GB of wasted KV cache that sits
until the tree prunes itself under eviction pressure.

There's no way to fix this without changes to SGLang's eviction policy, but you can
protect your serving availability in the meantime by monitoring `eviction_rate` as a
leading indicator — when eviction is high at moderate utilization, dead branches are
the cause and you need to shed load before the tree exhausts the KV pool.

I built **[ml-memguard](https://github.com/vgpprasad91/ml-memguard)** to track exactly
this. It exposes `eviction_rate` alongside `kv_cache_usage_perc`, applies a rolling-max
smoother to suppress false-recovery signals from the eviction drops, and fires
`on_shed_load` before the KV pool is exhausted:

```python
from memory_guard import guard_sglang

safe = guard_sglang(runtime)

def on_shed_load(util):
    # Stop new reasoning requests routing to this replica
    lb.reduce_weight(replica_id, 0)

def on_warning(util):
    # 80% rolling-max — dead branches likely accumulating, signal early
    metrics.increment("sglang.kvcache.dead_branch_pressure")

# Tighten for reasoning model workloads
safe.monitor.warning_threshold   = 0.75
safe.monitor.shed_load_threshold = 0.85
safe.monitor.on_warning          = on_warning
safe.monitor.on_shed_load        = on_shed_load

with safe.monitor.session():
    runtime.wait()
```

3-minute setup: https://github.com/vgpprasad91/ml-memguard/blob/main/docs/quickstart/sglang.md

I also submitted a PR to add this to the SGLang integrations docs: {PR_URL}
```

---

## Issue #12496 — CUDA OOM: unable to run same model as vLLM on same hardware

> **URL:** https://github.com/sgl-project/sglang/issues/12496

```markdown
This is a well-known asymmetry between SGLang and vLLM: SGLang's RadixAttention tree
reserves more GPU memory upfront for its prefix cache node structure than vLLM's block
table. So `--mem-fraction-static 0.88` in SGLang is not equivalent to
`--gpu-memory-utilization 0.90` in vLLM — the effective KV budget after model weights
and tree overhead is smaller in SGLang, which is why you hit OOM where vLLM doesn't.

The fix is to binary-search for the right `--mem-fraction-static` value for your specific
model+GPU combination rather than porting vLLM's utilization fraction directly.

**[ml-memguard](https://github.com/vgpprasad91/ml-memguard)** does this automatically:

```python
from memory_guard import guard_sglang

safe = guard_sglang(runtime)

print(safe)
# InferenceSafeConfig
#   gpu_total:     24,564 MB
#   weights_mb:    15,258 MB
#   kv_budget:      5,586 MB  (RadixAttention overhead reserved)
#   max_num_seqs:      12    ← binary-searched to fit
#   status:        FITS
#
# SGLang CLI: python -m sglang.launch_server \
#     --max-running-requests 12 --mem-fraction-static 0.85
```

It reads your model's architecture from the running runtime, accounts for RadixAttention
tree overhead, and gives you the exact `--max-running-requests` and
`--mem-fraction-static` values to use. Start from those numbers and the OOM goes away.

3-minute setup: https://github.com/vgpprasad91/ml-memguard/blob/main/docs/quickstart/sglang.md

I also submitted a PR to add this to the SGLang integrations docs: {PR_URL}
```
