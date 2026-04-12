# memguard vLLM Quickstart

Stop vLLM OOM crashes in three commands.

## What this does

Runs two containers side-by-side:

| Container | Port | Role |
|-----------|------|------|
| `vllm` | 8000 | vLLM OpenAI-compatible inference server |
| `sidecar` | 8001 | memguard sidecar — `/healthz` + `/readyz` probes |

The sidecar polls vLLM's Prometheus `/metrics` endpoint every 5 seconds,
sends the KV cache signals to the memguard cloud prediction API, and reflects
the OOM risk score through the `/readyz` probe.

**When the predicted OOM probability exceeds 0.70**, `/readyz` returns `503`.
In Kubernetes, a `503` readiness response removes the pod from the Service
endpoint set — new requests stop routing to this replica with zero code changes
to vLLM or your application.

## Three commands

```bash
# 1. Clone (or cd into the repo if already cloned)
git clone https://github.com/memguard-project/ml-memguard
cd ml-memguard/examples/vllm-quickstart

# 2. Configure
cp .env.example .env
# Edit .env: set MEMGUARD_BACKEND_KEY, MODEL_NAME, HF_TOKEN

# 3. Start
docker compose up
```

## Test the endpoints

```bash
# vLLM liveness
curl http://localhost:8000/health

# Sidecar liveness (always 200)
curl http://localhost:8001/healthz

# Sidecar readiness (200 = safe, 503 = OOM risk > threshold)
curl -i http://localhost:8001/readyz
```

Example `/readyz` response when healthy:
```json
{"status": "ready", "oom_probability": 0.12, "threshold": 0.7}
```

Example `/readyz` response under OOM pressure:
```json
{"status": "not_ready", "oom_probability": 0.83, "threshold": 0.7}
```

## Kubernetes readiness probe

Add this to your vLLM `Deployment` spec to wire memguard into Kubernetes
load-shedding without any custom operator:

```yaml
containers:
  - name: vllm
    image: vllm/vllm-openai:latest
    ports:
      - containerPort: 8000

  - name: memguard-sidecar
    image: python:3.11-slim
    command:
      - sh
      - -c
      - |
        pip install 'ml-memguard[cloud]' -q &&
        python -m memory_guard.sidecar \
          --vllm-url http://localhost:8000 \
          --port 8001
    ports:
      - containerPort: 8001
    env:
      - name: MEMGUARD_BACKEND_KEY
        valueFrom:
          secretKeyRef:
            name: memguard-secret
            key: api-key
    readinessProbe:
      httpGet:
        path: /readyz
        port: 8001
      initialDelaySeconds: 30
      periodSeconds: 10
      failureThreshold: 1
    livenessProbe:
      httpGet:
        path: /healthz
        port: 8001
      initialDelaySeconds: 10
      periodSeconds: 30
```

## Without GPU (CPU-only testing)

Remove the `deploy.resources` section from `docker-compose.yml` and use a
smaller CPU-friendly model:

```bash
MODEL_NAME=facebook/opt-125m docker compose up
```

## Environment variables

| Variable | Required | Default | Description |
|----------|----------|---------|-------------|
| `MEMGUARD_BACKEND_KEY` | Yes | — | Your memguard cloud API key |
| `MODEL_NAME` | Yes | — | HuggingFace model ID |
| `HF_TOKEN` | For gated models | — | HuggingFace access token |
| `MEMGUARD_SHED_THRESHOLD` | No | `0.70` | OOM probability threshold for 503 |
| `MEMGUARD_POLL_INTERVAL` | No | `5.0` | Seconds between KV cache polls |
| `VLLM_PORT` | No | `8000` | Host port for vLLM |
| `SIDECAR_PORT` | No | `8001` | Host port for sidecar |

## Get an API key

Sign up at [memguard.dev](https://memguard.dev) to get a free API key.
Without a key the sidecar still works — it falls back to rule-based thresholds
(shed load at 92% KV cache utilization) and `/readyz` always returns 200 until
the reactive threshold is crossed.
