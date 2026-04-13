# Emit OTel Gen AI KV cache semantic conventions (opentelemetry-specification/semantic-conventions#NNNN)

<!-- BEFORE SUBMISSION: replace #NNNN with the merged OTel spec PR number -->
<!-- BEFORE SUBMISSION: replace benchmark table placeholder rows with real measurements -->
<!-- BEFORE SUBMISSION: run: git apply --check docs/upstream-prs/vllm/0001-emit-otel-gen-ai-kv-cache-metrics.patch -->

## Summary

Adds three Prometheus gauges that implement the [OTel Gen AI KV cache semantic conventions](https://github.com/open-telemetry/semantic-conventions/pull/NNNN) merged in opentelemetry-specification/semantic-conventions#NNNN.

The signals are the minimum set needed for GPU OOM prediction and cross-engine KV cache observability:

| Prometheus metric name | OTel attribute name | Unit | Source |
|---|---|---|---|
| `gen_ai_server_kv_cache_eviction_rate` | `gen_ai.server.kv_cache.eviction_rate` | evictions/s | `len(kv_cache_eviction_events)` / elapsed |
| `gen_ai_server_kv_cache_allocation_velocity` | `gen_ai.server.kv_cache.allocation_velocity_mbps` | MB/s | Δ(`kv_cache_usage`) × total_blocks × block_size / elapsed |
| `gen_ai_server_scheduler_preemption_rate` | `gen_ai.server.scheduler.preemption_rate` | preemptions/s | `iteration_stats.num_preemptions` / elapsed |

**Feature flag**: `VLLM_OTEL_KV_METRICS_ENABLED=true` (default: off). When unset, no gauges are registered and no code runs on the critical path — zero overhead.

## Motivation

vLLM already exposes `vllm:kv_cache_usage_perc` (utilization snapshot) and `vllm:num_preemptions` (cumulative counter). These are insufficient for OOM prediction because:

1. **Utilization snapshot without velocity** — a cache at 72% that is growing at 500 MB/s will OOM in seconds; a cache at 72% that is stable will not. The snapshot cannot distinguish these.
2. **Cumulative counter without rate** — `vllm:num_preemptions_total` tells you how many preemptions have occurred since engine start; it does not tell you if preemptions are accelerating right now.
3. **No standardised names** — `vllm:gpu_cache_usage_perc` and SGLang's equivalent counter are different metric names for the same concept. Cross-engine dashboards require per-engine configuration.

The three signals added here, under the OTel-standardised `gen_ai.server.*` namespace, fix all three gaps and make vLLM's KV cache telemetry compatible with any OTel Collector pipeline.

## Implementation

### New file: `vllm/v1/metrics/kv_metrics.py`

`KVCacheMetricsCollector` — stateful rate calculator, no Prometheus dependency:
- Tracks `kv_cache_usage_frac`, `eviction_events`, `num_preemptions_delta` across scheduler steps
- Computes finite differences over `time.monotonic()` elapsed intervals
- Returns `KVCacheRates(eviction_rate, allocation_velocity_mbps, preemption_rate)` dataclass
- Falls back to blocks/s if `kv_block_size_bytes_approx_mb` is unavailable

### Modified: `vllm/v1/metrics/loggers.py`

`PrometheusStatLogger`:
- Three optional `prometheus_client.Gauge` attributes (set to `None` when feature flag is off)
- `_kv_collector: Optional[KVCacheMetricsCollector]` — constructed via `from_engine_config()` at startup
- `record()` calls `_kv_collector.update()` and calls `.set()` on each gauge immediately after the existing `kv_cache_usage` gauge update

No changes to the scheduler, worker, or model executor — this is purely an observability layer.

## Benchmark: latency regression at p50 / p99 / p999

Tested on A10G (24 GB), `meta-llama/Llama-3-8B-Instruct`, sharegpt dataset,
`max_num_seqs=256`, `gpu_memory_utilization=0.90`.

<!-- BEFORE SUBMISSION: replace placeholder rows with actual benchmark output -->
<!-- Run: python benchmarks/benchmark_serving.py --model meta-llama/Llama-3-8B-Instruct \
         --dataset-name sharegpt --num-prompts 1000 --request-rate 8 -->

| Metric | Baseline (main) | + this PR | Δ |
|---|---|---|---|
| Request throughput (req/s) | _TBD_ | _TBD_ | _TBD_ |
| TTFT p50 (ms) | _TBD_ | _TBD_ | _TBD_ |
| TTFT p99 (ms) | _TBD_ | _TBD_ | _TBD_ |
| ITL p50 (ms) | _TBD_ | _TBD_ | _TBD_ |
| ITL p99 (ms) | _TBD_ | _TBD_ | _TBD_ |
| ITL p999 (ms) | _TBD_ | _TBD_ | _TBD_ |

Expected regression: **< 0.1%** at p50/p99 (rate computation is one subtraction and one division per gauge per scheduler step; no lock contention, no network I/O, no allocation on the hot path).

## OTel Collector integration

Users can forward these metrics to any OTLP endpoint using the standard Prometheus receiver + OTLP exporter pipeline. Example config:

```yaml
# otelcol-contrib --config otelcol-mkcp.yaml
receivers:
  prometheus:
    config:
      scrape_configs:
        - job_name: vllm
          scrape_interval: 5s
          static_configs:
            - targets: ["localhost:8000"]

processors:
  filter/kv_signals:
    metrics:
      include:
        match_type: regexp
        metric_names:
          - "gen_ai_server_kv_cache_.*"
          - "gen_ai_server_scheduler_preemption_rate"

exporters:
  otlphttp:
    endpoint: https://your-otlp-backend.example.com
    headers:
      Authorization: "Bearer ${env:API_KEY}"

service:
  pipelines:
    metrics:
      receivers: [prometheus]
      processors: [filter/kv_signals]
      exporters: [otlphttp]
```

## Usage

```bash
# Enable the three OTel KV cache metrics
export VLLM_OTEL_KV_METRICS_ENABLED=true

# Start vLLM normally — metrics appear at /metrics
python -m vllm.entrypoints.openai.api_server \
  --model meta-llama/Llama-3-8B-Instruct \
  --gpu-memory-utilization 0.90

# Verify the three gauges are registered
curl -s http://localhost:8000/metrics | grep gen_ai_server
```

Expected output:
```
# HELP gen_ai_server_kv_cache_eviction_rate KV cache block eviction rate (evictions/s). OTel: gen_ai.server.kv_cache.eviction_rate
# TYPE gen_ai_server_kv_cache_eviction_rate gauge
gen_ai_server_kv_cache_eviction_rate{engine="0",model_name="meta-llama/Llama-3-8B-Instruct"} 0.0

# HELP gen_ai_server_kv_cache_allocation_velocity KV cache growth rate in MB/s (positive = growing). OTel: gen_ai.server.kv_cache.allocation_velocity_mbps
# TYPE gen_ai_server_kv_cache_allocation_velocity gauge
gen_ai_server_kv_cache_allocation_velocity{engine="0",model_name="meta-llama/Llama-3-8B-Instruct"} 0.0

# HELP gen_ai_server_scheduler_preemption_rate Sequence preemption rate (preemptions/s). OTel: gen_ai.server.scheduler.preemption_rate
# TYPE gen_ai_server_scheduler_preemption_rate gauge
gen_ai_server_scheduler_preemption_rate{engine="0",model_name="meta-llama/Llama-3-8B-Instruct"} 0.0
```

## Tests

```bash
# Unit tests for the rate calculator (no Prometheus, no vLLM engine)
pytest tests/v1/metrics/test_kv_metrics.py -v

# Integration: start a local vLLM, check the gauges appear
VLLM_OTEL_KV_METRICS_ENABLED=true pytest tests/metrics/test_otel_kv_gauges.py -v
```

## Checklist

- [ ] `VLLM_OTEL_KV_METRICS_ENABLED` env var documented in `docs/serving/env_vars.md`
- [ ] `test_kv_metrics.py` unit tests pass
- [ ] `git apply --check` passes against current vLLM main
- [ ] OTel spec PR number updated from `#NNNN` to actual merged PR
- [ ] Benchmark table populated with real numbers
- [ ] No regression in existing `tests/metrics/` test suite

## Related

- OTel Gen AI semantic conventions PR: opentelemetry-specification/semantic-conventions#NNNN
- OTel Collector config: [otelcol-mkcp.yaml](../../integrations/otelcol-mkcp.yaml)
- MKCP protocol spec: [mkcp-v1.md](../../protocol/mkcp-v1.md)
- Patch base commit: `8d825b87d6590ca971823890f9705988b8709add`
