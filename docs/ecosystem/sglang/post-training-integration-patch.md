# SGLang Docs Patch

**Target file in the SGLang repo:** `docs/references/post_training_integration.md`

The file uses a flat bullet list with the format:
`- [**Name**](URL): single-sentence description`

## Where to insert

Add a new section heading **"Production Monitoring & OOM Prevention"** at the bottom of
`post_training_integration.md`, after all existing post-training framework entries.
Then add the memguard bullet under it.

## Exact content to append

```markdown

## Production Monitoring & OOM Prevention

- [**ml-memguard**](https://github.com/memguard-project/ml-memguard): Proactive KV cache
  monitor and OOM prevention for SGLang — tracks RadixAttention eviction pressure,
  fragmentation ratio, and allocation velocity, fires load-shed signals before KV cache
  exhaustion, and works on both CUDA and Apple Silicon Metal backends.
```

## Full paragraph description (for PR body)

ml-memguard addresses three SGLang-specific memory failure modes that
`kv_cache_usage_perc` alone does not expose: (1) dead RadixAttention branches from
reasoning model thinking tokens that resist LRU eviction and accumulate 1.3–1.6 GB per
conversation; (2) eviction-driven utilization drops that mask genuine memory pressure by
suddenly releasing KV slots, causing monitoring tools to signal false recovery;
(3) CUDA OOM on hardware where vLLM runs safely, caused by SGLang's higher upfront
RadixAttention tree reservation vs vLLM's paged attention. memguard addresses all three
with a 3-reading rolling-maximum smoother on the raw utilization, an eviction_rate
signal that exposes dead-branch pressure at moderate utilization, and a pre-flight
binary search that accounts for RadixAttention overhead in the KV budget. It also
supports Apple Silicon Metal backends — the only monitoring tool that does — where
the absence of a CUDA OOM exception makes proactive load-shedding critical.
