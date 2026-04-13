# vLLM upstream PR — OTel Gen AI KV cache metrics

Staged draft. **Do not submit until the OTel semantic conventions PR merges.**

## Files

| File | Purpose |
|---|---|
| `0001-emit-otel-gen-ai-kv-cache-metrics.patch` | Unified diff to apply to vLLM main |
| `kv_metrics.py` | New module (`vllm/v1/metrics/kv_metrics.py`) — readable standalone copy |
| `PR_DESCRIPTION.md` | Full GitHub PR description with benchmark table placeholders |

## To submit

1. Confirm OTel spec PR has merged → note the PR number (replaces `#NNNN` everywhere)
2. Fork `vllm-project/vllm` and checkout a branch from current main:
   ```bash
   git checkout -b otel-kv-cache-metrics
   ```
3. Apply and verify the patch:
   ```bash
   git apply --check docs/upstream-prs/vllm/0001-emit-otel-gen-ai-kv-cache-metrics.patch
   git apply docs/upstream-prs/vllm/0001-emit-otel-gen-ai-kv-cache-metrics.patch
   ```
   If `--check` fails, the patch context lines have drifted from vLLM main.
   Run `git diff HEAD~1 vllm/v1/metrics/loggers.py` to find the current
   insertion points and re-apply the `loggers.py` hunk manually.
4. Run the benchmark to populate the latency table in `PR_DESCRIPTION.md`:
   ```bash
   VLLM_OTEL_KV_METRICS_ENABLED=true \
   python benchmarks/benchmark_serving.py \
     --model meta-llama/Llama-3-8B-Instruct \
     --dataset-name sharegpt \
     --num-prompts 1000 \
     --request-rate 8
   ```
5. Update `PR_DESCRIPTION.md`: replace `#NNNN` with real PR number, fill benchmark table
6. Push branch and open PR against `vllm-project/vllm:main`
7. Tag `@vllm-project/metrics` reviewers in the PR body

## Patch base

Written against vLLM commit `8d825b87d6590ca971823890f9705988b8709add`
(April 13 2026). The `kv_metrics.py` new-file hunk applies cleanly to any
vLLM version ≥ 0.4.0. The `loggers.py` hunk may need line-number adjustment
if `PrometheusStatLogger.__init__` has changed since the base commit.
