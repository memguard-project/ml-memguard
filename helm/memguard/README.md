# memguard Helm Chart

Stop vLLM / SGLang OOM crashes in Kubernetes — without changing a line of
inference server code.

## What it does

Deploys vLLM and the memguard sidecar as two containers in the same pod.
The sidecar polls vLLM's Prometheus `/metrics` endpoint every 5 seconds,
monitors KV cache utilization, and exposes `/healthz` + `/readyz`:

| Endpoint | Healthy | Under OOM pressure |
|----------|---------|-------------------|
| `/healthz` | `200 {"status":"ok"}` | `200 {"status":"ok"}` (always) |
| `/readyz` | `200 {"status":"ready",...}` | `503 {"status":"not_ready",...}` |

Kubernetes wires the `readinessProbe` to the sidecar's `/readyz`. When the
KV cache exceeds the OOM threshold, `/readyz` returns `503` and Kubernetes
removes the pod from the Service endpoint set — **no new traffic, zero
vLLM code changes, existing requests drain cleanly**.

## Prerequisites

- Kubernetes 1.26+
- Helm 3.10+
- NVIDIA GPU operator (or remove `nvidia.com/gpu` resource limit for CPU-only)

## Three commands

```bash
# 1. Add the chart repo (once)
helm repo add memguard https://memguard-project.github.io/ml-memguard
helm repo update

# 2. Install
helm install my-vllm memguard/memguard \
  --set vllm.modelName=meta-llama/Llama-3.1-8B-Instruct \
  --set vllm.hfToken=hf_YOUR_TOKEN \
  --namespace inference --create-namespace

# 3. Test
kubectl port-forward deployment/my-vllm 8000:8000 8001:8001 -n inference
curl -i http://localhost:8001/readyz
```

## Values reference

| Key | Default | Description |
|-----|---------|-------------|
| `vllm.modelName` | `""` | **Required.** HuggingFace model ID |
| `vllm.hfToken` | `""` | HuggingFace access token (gated models) |
| `vllm.image.tag` | `latest` | vLLM Docker image tag |
| `vllm.extraArgs` | see values.yaml | Extra flags passed to vLLM after `--model` |
| `vllm.resources` | 16Gi RAM, 1 GPU | vLLM container resource requests/limits |
| `vllm.modelCachePVC` | `""` | PVC name for HF model cache (emptyDir if blank) |
| `sidecar.shedThreshold` | `0.70` | OOM probability threshold for `/readyz` 503 |
| `sidecar.pollInterval` | `5.0` | Seconds between KV cache polls |
| `sidecar.smoothingWindow` | `1` | Rolling-max window (use `3` for SGLang) |
| `sidecar.modelName` | `""` | Attached to telemetry (defaults to `vllm.modelName`) |
| `sidecar.backend` | `""` | Backend string attached to telemetry |
| `sidecar.resources` | 256Mi RAM, 100m CPU | Sidecar resource requests/limits |
| `replicaCount` | `1` | Number of pod replicas |
| `nodeSelector` | `{}` | Pin pods to GPU nodes |
| `tolerations` | `[]` | Tolerations for GPU taints |
| `memguardCloud.apiKey` | `""` | API key for fleet telemetry (optional) |
| `memguardCloud.existingSecret` | `""` | Use an existing Secret for the API key |
| `service.type` | `ClusterIP` | Kubernetes Service type |

## SGLang

Set `--set sidecar.smoothingWindow=3` to enable RadixAttention-aware
rolling-max smoothing. The sidecar auto-detects SGLang's Prometheus format.

```bash
helm install my-sglang memguard/memguard \
  --set vllm.image.repository=lmsysorg/sglang \
  --set vllm.image.tag=latest \
  --set vllm.modelName=meta-llama/Llama-3.1-8B-Instruct \
  --set sidecar.smoothingWindow=3 \
  --set sidecar.shedThreshold=0.85
```

## Kubernetes readiness probe (what the chart wires automatically)

```yaml
readinessProbe:
  httpGet:
    path: /readyz
    port: 8001        # memguard sidecar, not vLLM
  initialDelaySeconds: 30
  periodSeconds: 10
  failureThreshold: 1  # one 503 is enough to pull the pod
```

`failureThreshold: 1` means a single 503 from `/readyz` removes the pod
from the endpoint set. This is intentional — by the time the KV cache
hits the threshold, there is no margin for another 503-causing request.

## memguard-cloud (optional fleet intelligence)

Set an API key to activate OOM prediction and fleet policy sync:

```bash
helm upgrade my-vllm memguard/memguard \
  --set memguardCloud.apiKey=mg_YOUR_KEY \
  --reuse-values
```

Or reference an existing Secret:

```bash
kubectl create secret generic memguard-cloud-key \
  --from-literal=MEMGUARD_API_KEY=mg_YOUR_KEY \
  -n inference

helm upgrade my-vllm memguard/memguard \
  --set memguardCloud.existingSecret=memguard-cloud-key \
  --reuse-values
```

## Without GPU (CPU-only testing)

Remove the GPU resource limit and use a small model:

```bash
helm install memguard-test memguard/memguard \
  --set vllm.modelName=facebook/opt-125m \
  --set vllm.resources.limits."nvidia\.com/gpu"=null \
  --set vllm.extraArgs="{--host,0.0.0.0,--port,8000,--max-model-len,512}"
```
