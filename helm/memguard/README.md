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

## How Kubernetes control-plane load-shedding works

This is the exact sequence of events when the KV cache exceeds the OOM threshold:

```
1. memguard sidecar polls vLLM /metrics every pollInterval seconds
2. KVCacheMonitor computes oom_probability from KV utilization, velocity,
   fragmentation ratio, and eviction rate
3. oom_probability > shedThreshold → /readyz returns HTTP 503
4. kubelet detects 503 on failureThreshold: 1 — pod marked NotReady immediately
5. kube-proxy updates iptables/ipvs rules on all nodes within ~1 second —
   pod IP removed from Service Endpoints slice
6. Ingress controller / Gateway API / cloud LB stops routing new connections
   to this pod on its next health-check cycle (typically < 5 s)
7. In-flight vLLM requests continue — pod is NOT killed, sequences drain cleanly
8. VLLMWatchdog schedules a graceful restart during the quiet drain window
9. After restart: /readyz returns 200 → pod re-enters the Endpoints slice →
   traffic resumes automatically — no manual intervention required
```

**Key property**: `failureThreshold: 1` means a single 503 pulls the pod.
This is intentional — by the time the KV cache hits the threshold there is
no margin for another request to land. The sidecar's threshold should be set
conservatively (default 0.70) to provide that margin.

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
| `podDisruptionBudget.enabled` | `true` | Create PodDisruptionBudget |
| `podDisruptionBudget.minAvailable` | `1` | Minimum pods available during disruptions |
| `autoscaling.keda.enabled` | `false` | Enable KEDA ScaledObject (requires KEDA CRD) |
| `autoscaling.keda.minReplicaCount` | `1` | Minimum replicas |
| `autoscaling.keda.maxReplicaCount` | `10` | Maximum replicas |
| `autoscaling.keda.scaleUpThreshold` | `"65"` | KV cache % to trigger scale-up |
| `autoscaling.keda.prometheusAddress` | see values.yaml | Prometheus server URL |
| `autoscaling.keda.prometheusQuery` | `avg(vllm_kv_cache_usage_perc)` | PromQL query |
| `patchExistingDeployment.enabled` | `false` | Patch a bring-your-own-vLLM Deployment |
| `patchExistingDeployment.targetDeployment` | `""` | Name of Deployment to patch |
| `patchExistingDeployment.targetNamespace` | `""` | Namespace (defaults to Release namespace) |
| `patchExistingDeployment.containerName` | `"vllm"` | Container name inside the Deployment |
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

## PodDisruptionBudget

Enabled by default (`podDisruptionBudget.enabled=true`, `minAvailable=1`).
Prevents node drains from evicting all replicas simultaneously. For a
single-replica deployment this blocks the drain until the replacement pod
passes `/readyz` — eliminating the 0-replica traffic window.

Disable only if you manage disruption budgets externally:

```bash
helm upgrade my-vllm memguard/memguard \
  --set podDisruptionBudget.enabled=false \
  --reuse-values
```

## KEDA autoscaling (optional)

Requires [KEDA](https://keda.sh) installed in the cluster:

```bash
# Check KEDA is present
kubectl get crd scaledobjects.keda.sh
```

Enable with a Prometheus address pointing to your monitoring stack:

```bash
helm upgrade my-vllm memguard/memguard \
  --set autoscaling.keda.enabled=true \
  --set autoscaling.keda.prometheusAddress=http://kube-prometheus-stack-prometheus.monitoring.svc:9090 \
  --set autoscaling.keda.scaleUpThreshold=65 \
  --set autoscaling.keda.maxReplicaCount=8 \
  --reuse-values
```

**Tuning rule**: set `scaleUpThreshold` below `sidecar.shedThreshold × 100`
so KEDA starts adding replicas *before* individual pods start shedding load:

```
sidecar.shedThreshold=0.70  →  autoscaling.keda.scaleUpThreshold="65"
sidecar.shedThreshold=0.85  →  autoscaling.keda.scaleUpThreshold="80"
```

## Bring your own vLLM Deployment

If vLLM is already running in your cluster (deployed via a different chart
or manifest), inject the memguard readiness gate without redeploying:

```bash
# 1. Deploy only the sidecar Deployment + Service
helm install memguard-probe memguard/memguard \
  --set vllm.modelName=meta-llama/Llama-3.1-8B-Instruct \
  --set patchExistingDeployment.enabled=true \
  --set patchExistingDeployment.targetDeployment=my-existing-vllm \
  --set patchExistingDeployment.targetNamespace=inference \
  --namespace inference
```

A Helm post-install Job runs `kubectl strategic-merge-patch` to inject:

```json
{
  "spec": {
    "template": {
      "spec": {
        "containers": [{
          "name": "vllm",
          "readinessProbe": {
            "httpGet": { "path": "/readyz", "port": 8001 },
            "failureThreshold": 1,
            "periodSeconds": 10
          }
        }]
      }
    }
  }
}
```

The patch is idempotent — safe to re-run on `helm upgrade`.

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
