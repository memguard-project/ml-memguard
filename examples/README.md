# memguard examples

Runnable code examples for common integration patterns.

## Inference serving (vLLM)
- [`vllm_watchdog.py`](vllm_watchdog.py) — auto-healing vLLM server with bandit policy
- [`vllm_allocator_integration.py`](vllm_allocator_integration.py) — memguard-allocator hook
- [`vllm-quickstart/`](vllm-quickstart/README.md) — Docker Compose stack: vLLM + memguard sidecar

## Inference serving (SGLang)
- [`sglang_monitor.py`](sglang_monitor.py) — KV-cache monitor with load-shedding callbacks

## Kubernetes
- [`k8s/memguardpolicy.yaml`](k8s/memguardpolicy.yaml) — MemGuardPolicy CRD example
