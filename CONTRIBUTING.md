# Contributing to ml-memguard

## Report a hardware config

The most valuable contribution is a **hardware + framework verification report**.
Every report expands the supported hardware matrix in README.md and improves
OOM prediction accuracy for everyone on that hardware class.

### What to report

Open a [GitHub issue](https://github.com/memguard-project/ml-memguard/issues/new?template=hardware_report.yml)
using the **Hardware Config Report** template. It captures:

- GPU model and VRAM (e.g. RTX 4090 24GB, A100 80GB, M3 Max 36GB)
- Framework and version (vLLM 0.4.x, SGLang 0.3.x, Unsloth, HuggingFace Trainer, mlx_lm)
- Model name and quantization
- Whether memguard prevented an OOM (or detected one you didn't expect)
- The pre-flight output (`InferenceSafeConfig` or training estimate block)

### Why it matters

memguard's OOM prediction model trains on synthetic data generated from three
failure scenarios (`gradual_fill`, `burst_long_seqs`, `fragmentation_trap`).
Real telemetry from diverse hardware fills the gaps that synthetic data misses —
particularly for AMD ROCm, older CUDA GPUs, and small-VRAM Apple Silicon.

Each report also helps us populate the [Supported Hardware matrix](README.md#supported-hardware)
in the README, which is the first thing potential users look at before installing.

### Quick report (CLI output)

```bash
# For inference (vLLM)
python -c "
from memory_guard import guard_vllm
from vllm import LLM
llm = LLM(model='meta-llama/Llama-3.1-8B-Instruct', gpu_memory_utilization=0.9)
safe = guard_vllm(llm.llm_engine)
print(safe)
"

# For fine-tuning
python -c "
from memory_guard import MemoryGuard
mg = MemoryGuard(model_name='meta-llama/Llama-3.1-8B-Instruct', bits=4, lora_rank=16)
cfg = mg.preflight(batch_size=2, seq_length=2048)
print(cfg)
"
```

Paste the output block into your issue. That's the whole report.

---

## Hardware Config Report (issue template)

The template is at `.github/ISSUE_TEMPLATE/hardware_report.yml`. When you open
a new issue you will see it in the template picker.

Fields:

| Field | Example |
|-------|---------|
| GPU model | RTX 4090 |
| VRAM | 24 GB |
| Framework | vLLM 0.4.3 |
| Model | meta-llama/Llama-3.1-8B-Instruct |
| Quantization | FP16 |
| OOM prevented? | Yes — memguard shed load before vLLM crashed |
| Pre-flight output | (paste block) |
| OS | Ubuntu 22.04 / CUDA 12.1 |

---

## Hardware configs we especially need

These are the gaps in the current Supported Hardware matrix:

| Hardware | Gap |
|----------|-----|
| RTX 3060 / 3080 / 3090 (12–24 GB) | Not yet verified — fragmentation behavior differs |
| RTX 4070 / 4080 (12–16 GB) | Not yet verified |
| A100 40GB | Not yet verified — large KV budget changes signal distribution |
| H100 80GB | Not yet verified |
| M1 / M2 MacBook Air (8 GB, 16 GB) | Not yet verified — OOM behavior at small memory critical |
| M3 / M4 MacBook Pro (18 GB, 36 GB) | M4 Max 36 GB tested; other configs needed |
| AMD ROCm (RX 7900, MI300X) | Not yet verified |
| CPU-only (inference) | Minimal KV cache; different OOM profile |

---

## Running the benchmark

To get a pre-formatted report with estimated vs actual memory:

```bash
pip install ml-memguard

# Inference benchmark
python bench/bench_accuracy.py --framework vllm

# Training benchmark (Apple Silicon)
pip install ml-memguard mlx-lm
python bench/bench_accuracy.py --framework mlx_lm

# With a specific model
python bench/bench_accuracy.py --model mlx-community/Qwen3.5-9B-MLX-4bit --submit
```

The `--submit` flag generates a pre-formatted GitHub issue body. Copy it, open
a new issue at https://github.com/memguard-project/ml-memguard/issues/new, and paste.

---

## Other contributions

### Bug reports

If the memory estimate was off by more than 30%, that's a bug. Open an issue with:
- The `preflight()` output (estimated MB)
- Actual peak memory (`nvidia-smi` / `mlx.core.get_active_memory()`)
- Model config (name, bits, batch size, sequence length, LoRA rank)

### Framework adapters

Missing adapter: PyTorch Lightning, Axolotl, LitGPT. See `memory_guard/adapters/`
for the HuggingFace and Unsloth adapter pattern.

### Code

1. Fork the repo and create a feature branch
2. `pip install -e ".[dev]"` for the dev dependencies
3. `pytest` — all 101 tests must pass
4. Open a PR; the CI runs `ruff` + `mypy` + `pytest` automatically
