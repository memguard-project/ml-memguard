# KV Cache Memory Pressure Semantic Conventions for LLM Inference Servers

**Status**: Experimental  
**Target namespace**: `gen_ai.server`  
**Related working group**: OpenTelemetry Gen AI Working Group

---

## Overview

This document proposes semantic conventions for KV cache memory pressure signals
emitted by large language model (LLM) inference servers.

Modern LLM inference engines (vLLM, SGLang, TensorRT-LLM, llama.cpp) manage a
**KV cache** — a GPU-resident block store of key-value attention tensors reused
across decoding steps. When the KV cache fills, the scheduler begins evicting
live sequences (preemption), increasing latency and risk of out-of-memory (OOM)
crashes. Operators currently have no standardised way to observe this across
engines or across cloud providers.

These conventions define the minimum set of gauge and counter signals needed to:

1. Alert on impending GPU OOM before it crashes the pod.
2. Enable horizontal scaling decisions based on real KV cache pressure.
3. Support cross-engine observability dashboards without per-engine configuration.
4. Feed predictive memory management models with a well-defined training signal.

---

## Attributes

<!-- semconv gen_ai.server.kv_cache -->

| Attribute | Type | Description | Examples | Requirement Level |
|---|---|---|---|---|
| `gen_ai.server.kv_cache.utilization` | double | Fraction of KV cache blocks currently allocated. Monotonically increases as requests are scheduled; drops as sequences complete or are evicted. | `0.0`, `0.72`, `1.0` | Recommended |
| `gen_ai.server.kv_cache.allocation_velocity_mbps` | double | Rate of KV cache growth in MB/s over the last polling interval. Non-zero during active prefill; zero or negative when cache is stable or shrinking. | `0.0`, `128.5`, `512.0` | Recommended |
| `gen_ai.server.kv_cache.eviction_rate` | double | Scheduler block evictions per second. Non-zero at moderate utilisation indicates the engine is under memory pressure and is preempting live sequences. | `0.0`, `4.2`, `100.0` | Recommended |
| `gen_ai.server.kv_cache.fragmentation_ratio` | double | Fraction of KV cache blocks that are allocated but not contiguous. Engines can OOM at less than 70% utilisation when fragmentation prevents a large contiguous allocation. | `0.0`, `0.35`, `1.0` | Opt-In |
| `gen_ai.server.memory.pressure_level` | double | OS-level cgroup memory pressure in megabytes above the configured high-watermark. Zero on systems without cgroup v2 memory.high support or without the required Linux capabilities. | `0.0`, `512.0`, `4096.0` | Opt-In |
| `gen_ai.server.memory.oom_risk_score` | double | Probability of an out-of-memory event in the next 30–120 seconds, as estimated by an external predictive model. Range [0.0, 1.0]. Zero when no predictive model is configured. | `0.0`, `0.45`, `0.92` | Opt-In |
| `gen_ai.server.memory.near_miss_count` | int | Cumulative count of GPU memory allocations that succeeded with less than the configured headroom remaining. Requires an instrumented allocator. Zero without custom allocator integration. | `0`, `14`, `3002` | Opt-In |
| `gen_ai.server.scheduler.preemption_rate` | double | Scheduler request preemptions per second. One preemption corresponds to evicting all KV cache blocks for a sequence. Distinct from `kv_cache.eviction_rate`: one preemption causes N block evictions. | `0.0`, `0.5`, `12.0` | Recommended |

<!-- endsemconv -->

### Attribute value constraints

| Attribute | Constraint |
|---|---|
| `gen_ai.server.kv_cache.utilization` | Range [0.0, 1.0] |
| `gen_ai.server.kv_cache.fragmentation_ratio` | Range [0.0, 1.0] |
| `gen_ai.server.memory.oom_risk_score` | Range [0.0, 1.0] |
| All `*_rate` and `*_velocity_mbps` attributes | Non-negative |
| `gen_ai.server.memory.near_miss_count` | Non-negative integer; cumulative (counter semantics) |

---

## Metrics

Each attribute above corresponds to a metric instrument:

| Metric | Instrument | Unit | Description |
|---|---|---|---|
| `gen_ai.server.kv_cache.utilization` | Gauge | `1` (fraction) | KV cache block utilization |
| `gen_ai.server.kv_cache.allocation_velocity` | Gauge | `By/s` | KV cache growth rate |
| `gen_ai.server.kv_cache.eviction_rate` | Gauge | `{eviction}/s` | Block eviction rate |
| `gen_ai.server.kv_cache.fragmentation_ratio` | Gauge | `1` (fraction) | Block fragmentation |
| `gen_ai.server.memory.pressure_level` | Gauge | `By` | Cgroup memory pressure above high-watermark |
| `gen_ai.server.memory.oom_risk_score` | Gauge | `1` (probability) | Predicted OOM probability |
| `gen_ai.server.memory.near_miss_count` | Counter | `{allocation}` | Near-miss GPU allocations |
| `gen_ai.server.scheduler.preemption_rate` | Gauge | `{preemption}/s` | Sequence preemption rate |

---

## Event: memory pressure state transition

In addition to the continuous gauges above, inference servers may emit a discrete
**memory pressure event** when a threshold is crossed. This follows the OTel
events specification (`Event` log record with `event.name` body key).

| Event name | Trigger condition | Traffic effect |
|---|---|---|
| `gen_ai.server.memory.pressure.warn` | `oom_risk_score` ≥ 0.50 or `kv_cache.utilization` ≥ 0.85 | None — alert only |
| `gen_ai.server.memory.pressure.shed_load` | `oom_risk_score` ≥ 0.70 (configurable) | Readiness probe returns 503; load balancer stops routing new requests |
| `gen_ai.server.memory.pressure.critical_restart` | Watchdog initiates graceful restart sequence | Pod will become unavailable within ~60 seconds |

Event log record body fields:

```json
{
  "event.name": "gen_ai.server.memory.pressure.warn",
  "triggered_by": "gen_ai.server.memory.oom_risk_score",
  "value": 0.52,
  "threshold": 0.50
}
```

---

## Polling interval

These signals are sampled at the polling interval of the inference engine's
scheduler loop. Typical values:

| Engine | Default polling interval |
|---|---|
| vLLM | 5 seconds (`--scheduler-delay-factor`) |
| SGLang | 1 second |
| TensorRT-LLM | 5 seconds |
| llama.cpp (server) | 1 second |

Implementations SHOULD emit metrics at each poll cycle. Consumers SHOULD
compute rolling aggregates (mean, max) over 30 s, 60 s, and 120 s windows
rather than reacting to instantaneous samples.

---

## Relationship to existing Gen AI conventions

These conventions extend the existing `gen_ai.server.*` namespace defined in
[gen-ai-server.md](./gen-ai-server.md). They are additive — no existing
attribute is renamed or removed.

The existing `gen_ai.server.request.duration` histogram captures per-request
tail latency; these new conventions capture the underlying memory resource
pressure that causes latency degradation before requests complete.

---

## Implementation notes

### Minimum viable emission

An inference server achieves minimum conformance by emitting the three
**Recommended**-level metrics:

- `gen_ai.server.kv_cache.utilization`
- `gen_ai.server.kv_cache.eviction_rate`
- `gen_ai.server.scheduler.preemption_rate`

The four **Opt-In** metrics require additional instrumentation:

- `fragmentation_ratio`: internal block allocator accounting
- `memory.pressure_level`: Linux cgroup v2 + CAP_BPF or equivalent
- `memory.oom_risk_score`: external predictive model integration
- `memory.near_miss_count`: instrumented GPU memory allocator

### Prometheus exposition

When exported via Prometheus (`/metrics`), the attribute names map to metric
names using the standard OTel-to-Prometheus bridge convention
(`gen_ai_server_` prefix, `.` → `_`):

```
# HELP gen_ai_server_kv_cache_utilization KV cache block utilization
# TYPE gen_ai_server_kv_cache_utilization gauge
gen_ai_server_kv_cache_utilization{model="meta-llama/Llama-3-8B",backend="cuda"} 0.72

# HELP gen_ai_server_kv_cache_eviction_rate Block eviction rate (evictions/s)
# TYPE gen_ai_server_kv_cache_eviction_rate gauge
gen_ai_server_kv_cache_eviction_rate{model="meta-llama/Llama-3-8B",backend="cuda"} 4.2

# HELP gen_ai_server_scheduler_preemption_rate Sequence preemption rate (preemptions/s)
# TYPE gen_ai_server_scheduler_preemption_rate gauge
gen_ai_server_scheduler_preemption_rate{model="meta-llama/Llama-3-8B",backend="cuda"} 0.5
```

### Recommended resource attributes

Implementations SHOULD attach the following OTel resource attributes to
disambiguate multi-model or multi-GPU deployments:

- `gen_ai.system` — e.g., `"vllm"`, `"sglang"`, `"trt-llm"`
- `gen_ai.request.model` — HuggingFace model identifier
- `service.instance.id` — pod name or host identifier
- `k8s.pod.name` / `k8s.node.name` — when running in Kubernetes

---

## Prior art and related work

- [vLLM metrics](https://docs.vllm.ai/en/latest/serving/metrics.html): exposes
  `vllm:gpu_cache_usage_perc` and `vllm:num_preemptions_total` but without
  standardised attribute names or units.
- [SGLang metrics](https://sglang.readthedocs.io/): similar Prometheus counters,
  incompatible naming.
- [OpenLLMetry](https://github.com/traceloop/openllmetry): instruments LLM SDK
  calls but does not cover server-side KV cache resource signals.
- [OTel Gen AI Working Group charter](https://github.com/open-telemetry/community/blob/main/projects/gen-ai-observability.md):
  the working group scope explicitly includes "LLM serving infrastructure signals."

---

## Open questions

1. **Instrument type for `eviction_rate` and `preemption_rate`**: Should these be
   `Gauge` (instantaneous rate computed by the engine) or `Counter` (cumulative
   eviction/preemption count, with rate computed by the consumer)? The current
   proposal uses `Gauge` to match vLLM's existing exposition, but Counter would
   be more idiomatic OTel.

2. **`kv_cache.allocation_velocity` units**: `By/s` (bytes per second) or a
   custom unit `{mb}/s`? Using `By/s` is idiomatic OTel but requires engines to
   emit raw bytes rather than MB.

3. **`memory.near_miss_count` reset semantics**: Should this reset on pod restart
   (process-lifetime counter) or on model reload? Current proposal is
   process-lifetime.

4. **Prefix cache hit rate**: SGLang's RadixAttention and vLLM's prefix caching
   expose a prefix cache hit rate signal. Should `gen_ai.server.kv_cache.prefix_cache_hit_ratio`
   be included here or deferred to a follow-up?

---

*Proposed by Guru Prasad Venkata Raghavan. Discussion welcome via the OTel Gen AI Working Group Slack channel (#otel-gen-ai) or the linked GitHub issue.*
