# PR Body: Add memguard to SGLang integrations docs

> **Target repo:** https://github.com/sgl-project/sglang
> **Target file:** `docs/references/post_training_integration.md` (append to existing file)
> **Target branch:** `main`
>
> Apply the patch in `post-training-integration-patch.md` to the SGLang repo,
> then open a PR with the title and body below.

---

## PR Title

```
docs: add memguard — proactive KV cache monitoring and OOM prevention for SGLang
```

---

## PR Body

### What this adds

Adds a new "Production Monitoring & OOM Prevention" section to
`docs/references/post_training_integration.md` with an entry for
[ml-memguard](https://github.com/memguard-project/ml-memguard), an open-source KV cache
monitor designed for SGLang production deployments.

### Why

SGLang's RadixAttention prefix cache introduces OOM failure modes that are not present
in vLLM's paged attention architecture and are not detectable via `kv_cache_usage_perc`
alone:

1. **Dead branches from reasoning models** — DeepSeek-R1, QwQ, and similar models
   insert `<think>` tokens into the radix tree that multi-turn conversations never
   reuse, accumulating 1.3–1.6 GB of wasted KV cache per session (discussed in
   issue #22373). LRU eviction cannot reclaim non-leaf nodes, so 50 concurrent
   reasoning sessions can exhaust 65–80 GB of KV cache.

2. **Eviction-driven utilization drops** — When a cached prefix is freed, its KV
   token slots are released, causing reported utilization to drop suddenly (e.g.
   88% → 62% in one poll). Monitoring tools that react to the raw value signal
   false recovery and delay load-shedding.

3. **Higher upfront memory reservation** — SGLang reserves more GPU memory for
   the radix tree structure than vLLM's block table, causing CUDA OOM on hardware
   where vLLM runs safely at equivalent settings (discussed in issue #12496).

memguard addresses all three with a 3-reading rolling-maximum smoother on the raw
utilization, an `eviction_rate` signal that exposes dead-branch accumulation at
moderate utilization, and a pre-flight binary search that accounts for RadixAttention
overhead. It also supports **Apple Silicon Metal backends** — the only monitoring
tool that does.

### What this does not change

No SGLang source files are modified. This is a documentation-only addition to the
existing community integrations list.

### Related issues

- #22373 — Reasoning model thinking tokens pollute radix cache with unreachable entries
- #12496 — CUDA OOM: unable to run same model as vLLM on same hardware

---

## Checklist before opening the PR

1. Fork https://github.com/sgl-project/sglang
2. Create branch: `git checkout -b docs/memguard-integration`
3. Open `docs/references/post_training_integration.md`
4. Append the content from `post-training-integration-patch.md` (new section at the bottom)
5. Open PR against `sgl-project/sglang:main` with the title and body above
6. After PR is open, post the replies in `issue-replies.md` to issues #22373 and #12496,
   linking to the PR
