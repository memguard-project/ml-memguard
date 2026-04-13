# OTel Collector Integration — KV Cache Telemetry to memguard-cloud

Route vLLM (or any OTel-instrumented inference server) KV cache signals to
memguard-cloud through the OpenTelemetry Collector. The collector receives
OTLP metrics, filters to the three GBT-wired signals, and forwards them to
`POST /v1/ingest/otlp` — the same D1 table and GBT feature pipeline that
the native MKCP sidecar writes to.

**When to use this path vs the MKCP sidecar:**

| | MKCP sidecar | OTel Collector |
|---|---|---|
| Setup effort | One `pip install` | Collector + config |
| Works without vLLM changes | Yes | Requires `VLLM_OTEL_KV_METRICS_ENABLED=true` |
| Works with any OTel emitter | No | Yes |
| Prometheus scrape side-channel | Optional | Built-in |
| Recommended for | Single-node, dev/staging | Multi-node, existing OTel infra |

---

## Prerequisites

- **OpenTelemetry Collector Contrib ≥ 0.96.0**
  (`otelcol-contrib` — not the core distribution; the `filter` processor with
  `match_type: strict` is in contrib only)
- **vLLM ≥ 0.4.0** with OTel metrics emission enabled (step 1 below)
- A **memguard API key** — issue one via `POST /v1/keys` or the memguard dashboard

---

## Step 1 — Enable KV cache metrics in vLLM

Set the following environment variables before starting your vLLM server:

```bash
# Enable the gen_ai.server.* KV cache metric emission
export VLLM_OTEL_KV_METRICS_ENABLED=true

# Point vLLM's OTel exporter at your collector's gRPC port
export OTEL_EXPORTER_OTLP_ENDPOINT=http://localhost:4317
export OTEL_EXPORTER_OTLP_PROTOCOL=grpc

# Identify the service instance in the fleet
export OTEL_SERVICE_NAME=vllm-inference
export OTEL_RESOURCE_ATTRIBUTES="service.instance.id=pod/$(hostname),gen_ai.system=vllm"
```

Then start vLLM normally:

```bash
python -m vllm.entrypoints.openai.api_server \
  --model meta-llama/Llama-3-8B-Instruct \
  --gpu-memory-utilization 0.90
```

vLLM will emit the following metrics to the collector every 5 seconds:

| OTel metric name | MKCP name | Unit |
|---|---|---|
| `gen_ai.server.kv_cache.eviction_rate` | `kvcache.eviction_rate` | evictions/s |
| `gen_ai.server.kv_cache.allocation_velocity_mbps` | `kvcache.allocation_velocity_mbps` | MB/s |
| `gen_ai.server.scheduler.preemption_rate` | `scheduler.preemption_rate` | preemptions/s |

> **Note:** `VLLM_OTEL_KV_METRICS_ENABLED` is added by the memguard vLLM upstream
> contribution (PR 63). Until that PR merges, use the [MKCP sidecar](../protocol/mkcp-v1.md)
> to emit these signals directly.

---

## Step 2 — Configure the OTel Collector

Copy [`otelcol-mkcp.yaml`](otelcol-mkcp.yaml) to your collector config directory.
The key blocks are:

**Receiver** — listens for OTLP on the standard ports:
```yaml
receivers:
  otlp:
    protocols:
      grpc:
        endpoint: 0.0.0.0:4317
      http:
        endpoint: 0.0.0.0:4318
```

**Filter processor** — keeps only the three MKCP-wired signals, drops everything else:
```yaml
processors:
  filter/mkcp_signals:
    error_mode: ignore
    metrics:
      include:
        match_type: strict
        metric_names:
          - "gen_ai.server.kv_cache.eviction_rate"
          - "gen_ai.server.kv_cache.allocation_velocity_mbps"
          - "gen_ai.server.scheduler.preemption_rate"
```

**Exporter** — forwards to memguard-cloud:
```yaml
exporters:
  otlphttp/memguard:
    metrics_endpoint: https://api.memguard.io/v1/ingest/otlp
    headers:
      Authorization: "Bearer ${env:MEMGUARD_API_KEY}"
    compression: gzip
    retry_on_failure:
      enabled: true
      initial_interval: 5s
      max_interval: 60s
```

**Pipeline wiring:**
```yaml
service:
  pipelines:
    metrics/mkcp:
      receivers:  [otlp]
      processors: [resourcedetection/system, filter/mkcp_signals, batch/mkcp]
      exporters:  [otlphttp/memguard]
```

---

## Step 3 — Add your memguard API key

Set `MEMGUARD_API_KEY` in the collector's environment before starting it:

```bash
export MEMGUARD_API_KEY=mg_your_api_key_here
```

**In Kubernetes**, use a Secret:

```yaml
apiVersion: v1
kind: Secret
metadata:
  name: memguard-api-key
type: Opaque
stringData:
  key: mg_your_api_key_here
```

Reference it in the collector's Deployment:

```yaml
env:
  - name: MEMGUARD_API_KEY
    valueFrom:
      secretKeyRef:
        name: memguard-api-key
        key: key
```

**In Docker Compose:**

```yaml
services:
  otelcol:
    image: otel/opentelemetry-collector-contrib:0.96.0
    command: ["--config=/etc/otelcol/otelcol-mkcp.yaml"]
    volumes:
      - ./otelcol-mkcp.yaml:/etc/otelcol/otelcol-mkcp.yaml:ro
    environment:
      MEMGUARD_API_KEY: ${MEMGUARD_API_KEY}
    ports:
      - "4317:4317"   # gRPC
      - "4318:4318"   # HTTP
```

---

## Step 4 — Start the collector

```bash
# Direct binary
otelcol-contrib --config docs/integrations/otelcol-mkcp.yaml

# Docker
docker run -p 4317:4317 -p 4318:4318 \
  -e MEMGUARD_API_KEY=$MEMGUARD_API_KEY \
  -v $(pwd)/docs/integrations/otelcol-mkcp.yaml:/etc/otelcol/config.yaml \
  otel/opentelemetry-collector-contrib:0.96.0 \
  --config /etc/otelcol/config.yaml
```

The collector health check is at `http://localhost:13133`. A `200 OK` response
means it is running and the pipeline is wired.

---

## Step 5 — Verify data is flowing

Wait 30–60 seconds after starting both vLLM and the collector, then call
the memguard dataset stats endpoint:

```bash
curl -s https://api.memguard.io/v1/model/dataset-stats \
  -H "Authorization: Bearer $MEMGUARD_API_KEY" | jq .
```

A successful response looks like:

```json
{
  "total_rows": 147,
  "oom_rate": 0.02,
  "mkcp_rows": 144,
  "mkcp_ingest_path": {
    "mkcp": 0,
    "otlp": 144
  },
  "tier_distribution": { "cuda": 144 }
}
```

`mkcp_rows > 0` and `mkcp_ingest_path.otlp > 0` confirms the pipeline is working.

If `mkcp_rows` stays at 0 after 60 seconds, check:

1. **vLLM is emitting** — `curl http://localhost:9090/metrics | grep gen_ai_server`
   should return the three metric families.
2. **Collector is receiving** — `curl http://localhost:8888/metrics | grep otelcol_receiver_accepted`
   should show a non-zero `otelcol_receiver_accepted_metric_points_total`.
3. **Filter is passing data through** — `curl http://localhost:8888/metrics | grep otelcol_processor_filter`
   — if `otelcol_processor_filter_datapoints_filtered_total` is high and
   `otelcol_exporter_sent_metric_points_total` is 0, the filter config is too
   aggressive. Check metric names with `grep gen_ai` on the vLLM `/metrics` output.
4. **Auth is valid** — a 401 from the exporter appears as
   `otelcol_exporter_send_failed_metric_points_total` in the collector metrics.
   Verify `MEMGUARD_API_KEY` is set and not empty.

---

## Multi-node / Kubernetes fleet setup

For a fleet of inference pods, deploy the collector as a DaemonSet so each
node has a local collector. vLLM pods point their `OTEL_EXPORTER_OTLP_ENDPOINT`
at the node-local collector (`http://$(NODE_IP):4317`):

```yaml
# vLLM pod spec env
- name: OTEL_EXPORTER_OTLP_ENDPOINT
  value: "http://$(MY_NODE_IP):4317"
- name: MY_NODE_IP
  valueFrom:
    fieldRef:
      fieldPath: status.hostIP
- name: VLLM_OTEL_KV_METRICS_ENABLED
  value: "true"
- name: OTEL_RESOURCE_ATTRIBUTES
  value: "service.instance.id=$(POD_NAME),k8s.pod.name=$(POD_NAME),gen_ai.system=vllm"
```

The collector DaemonSet uses `otelcol-mkcp.yaml` unchanged — it receives from
all pods on the node and forwards to memguard-cloud with per-pod attribution
via the `service.instance.id` resource attribute.

---

## See also

- [`otelcol-mkcp.yaml`](otelcol-mkcp.yaml) — complete copy-paste-ready config
- [`docs/protocol/mkcp-v1.md`](../protocol/mkcp-v1.md) — MKCP wire format (native sidecar path)
- [`docs/protocol/otel-submission/`](../protocol/otel-submission/) — OTel semantic convention PR draft
- [memguard-cloud `/v1/ingest/otlp` endpoint](../../memguard-cloud/src/index.ts) — Worker implementation
