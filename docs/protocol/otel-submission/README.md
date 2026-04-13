# OTel Semantic Convention Submission — KV Cache Memory Pressure

This directory contains the materials for a submission to the
[OpenTelemetry semantic conventions](https://github.com/open-telemetry/semantic-conventions)
repository, specifically the Gen AI working group.

## What this proposes

8 new metric attributes under the `gen_ai.server.*` namespace covering KV
cache memory pressure signals emitted by LLM inference servers (vLLM, SGLang,
TensorRT-LLM, llama.cpp). These are the signals needed to:

- Alert on impending GPU OOM before pod crash
- Drive HPA / load-shedding decisions on KV cache pressure
- Enable cross-engine observability dashboards

## Submission target

**Repository**: `open-telemetry/semantic-conventions`  
**Working group**: Gen AI Observability (`#otel-gen-ai` on CNCF Slack)  
**PR template**: Enhancement with `area:gen-ai` label  

## Files

```
docs/attributes/gen-ai/kv-cache.md        ← Human-readable convention doc
model/registry/gen-ai-kv-cache.yaml       ← Machine-readable registry YAML (weaver/semconvgen format)
```

## Pre-submission checklist

- [ ] Fork `open-telemetry/semantic-conventions`
- [ ] Copy `docs/attributes/gen-ai/kv-cache.md` → `docs/attributes/gen-ai/kv-cache.md` in fork
- [ ] Copy `model/registry/gen-ai-kv-cache.yaml` → `model/registry/gen-ai-kv-cache.yaml` in fork
- [ ] Run `make table-generation` in the fork to validate YAML schema
- [ ] Open PR against `main` with title:
      `[gen-ai] Add KV cache memory pressure semantic conventions for LLM inference servers`
- [ ] Tag `@open-telemetry/specs-approvers` and `@open-telemetry/gen-ai-wg`
- [ ] Link to this PR from the Gen AI WG meeting agenda

## Status

Not submitted. All files are ready for submission pending OTel Gen AI WG review.
