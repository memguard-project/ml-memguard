# memguard documentation

User-facing reference documentation for ml-memguard.

## Quickstart guides
- [vLLM quickstart](quickstart/vllm.md)
- [SGLang quickstart](quickstart/sglang.md)

## Reference
- [Adapters](adapters.md) — HuggingFace, Unsloth, vLLM, SGLang adapter API
- [Backends](backends.md) — backend detection and platform support
- [Inference serving](inference.md) — KV-cache monitoring and load-shedding
- [RL optimizer](rl_optimizer.md) — contextual bandit policy
- [Integrations](integrations/allocator.md) — memguard-allocator integration

## eBPF probes
- [Kubernetes eBPF setup](ebpf/ebpf-kubernetes.md)
- [cgroup_memory_high probe](ebpf/cgroup_memory_high_probe.md)

## Architecture decision records
Design decisions recorded at the time they were made:
- [001 — Mid-training downgrade semantics](decisions/001-mid-training-downgrade-semantics.md)
- [002 — QLoRA double-quant bits](decisions/002-qlora-double-quant-bits.md)
- [003 — Inference signals-only design](decisions/003-inference-signals-only.md)
- [004 — RL contextual bandit](decisions/004-rl-contextual-bandit.md)

## Protocol contributions
- [OpenTelemetry KV-cache semantic conventions](protocol/otel-submission/README.md)
