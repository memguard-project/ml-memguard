#!/usr/bin/env python3
"""Benchmark memory-guard estimation accuracy against real training.

Loads a model, runs preflight estimation, trains for a few steps,
compares estimated vs actual peak memory, and reports the error.

Usage:
    python bench/bench_accuracy.py
    python bench/bench_accuracy.py --model mlx-community/Mistral-7B-Instruct-v0.3-4bit
    python bench/bench_accuracy.py --submit  # GitHub issue template
"""

import argparse
import json
import platform
import sys
import tempfile
import time
from pathlib import Path


def run_benchmark(model_id: str, iters: int = 10) -> dict:
    try:
        import mlx.core as mx
        from mlx_lm import load
        from mlx_lm.tuner.trainer import TrainingArgs, train
        from mlx_lm.tuner.utils import linear_to_lora_layers
        from mlx_lm.tuner.datasets import ChatDataset, CacheDataset
        from mlx.utils import tree_flatten
        import mlx.optimizers as optim
    except ImportError:
        print("Requires mlx-lm: pip install mlx-lm")
        sys.exit(1)

    from memory_guard import MemoryGuard, detect_platform
    from memory_guard.platforms import get_mlx_peak_memory_mb, reset_mlx_peak_memory, get_mlx_active_memory_mb

    info = detect_platform()
    guard = MemoryGuard.auto()

    print(f"Platform: {info.chip_name} ({info.total_memory_mb:.0f}MB)")
    print(f"Model:    {model_id}")

    configs = [
        {"batch_size": 1, "seq_length": 512, "lora_rank": 8, "lora_layers": 8},
        {"batch_size": 2, "seq_length": 1024, "lora_rank": 16, "lora_layers": 16},
    ]

    # Training data
    train_data = [
        {"messages": [
            {"role": "user", "content": f"Explain concept {i} in detail."},
            {"role": "assistant", "content": f"Concept {i} involves several key principles. " * 20},
        ]}
        for i in range(20)
    ]

    results = []
    for cfg in configs:
        print(f"\n--- batch={cfg['batch_size']}, seq={cfg['seq_length']}, "
              f"rank={cfg['lora_rank']}, layers={cfg['lora_layers']} ---")

        # Fresh model load for each config (avoids LoRA-on-LoRA)
        print(f"  Loading model...", flush=True)
        reset_mlx_peak_memory()
        model, tokenizer = load(model_id)
        mx.eval(model.parameters())  # Force materialization

        # Measure ACTUAL model memory from Metal allocator (ground truth)
        model_memory_mb = (get_mlx_peak_memory_mb() or 0)

        total_params = sum(p.size for _, p in tree_flatten(model.parameters()))
        model_config = getattr(model, 'config', None) or getattr(model, 'args', None)
        hidden = getattr(model_config, 'hidden_size', None) or getattr(model_config, 'model_dim', 4096)
        heads = getattr(model_config, 'num_attention_heads', None) or getattr(model_config, 'n_heads', 32)
        layers = getattr(model_config, 'num_hidden_layers', None) or getattr(model_config, 'n_layers', 32)

        # For quantized models, tree_flatten gives quantized element count,
        # not original param count. Derive original from actual memory.
        bits = 4
        if hasattr(model_config, 'quantization'):
            bits = getattr(model_config.quantization, 'bits', 4)
        # Back-calculate original params from measured model memory
        if model_memory_mb > 0 and bits < 16:
            original_params = int(model_memory_mb * 1024 * 1024 / (bits / 8))
        else:
            original_params = total_params

        # Estimate
        safe = guard.preflight(
            model_params=original_params, model_bits=bits,
            hidden_dim=hidden, num_heads=heads, num_layers=layers,
            **cfg, grad_checkpoint=True,
        )
        estimated_mb = safe.estimate.total_mb

        if not safe.fits:
            print(f"  Skipped: {estimated_mb:.0f}MB > budget {safe.budget_mb:.0f}MB")
            del model
            continue

        # Reset peak memory, apply LoRA, train
        reset_mlx_peak_memory()

        model.freeze()
        lora_config = {"rank": safe.lora_rank, "scale": 20.0, "dropout": 0.0}
        linear_to_lora_layers(model, safe.lora_layers, lora_config)

        hf_tok = tokenizer._tokenizer if hasattr(tokenizer, '_tokenizer') else tokenizer
        train_set = CacheDataset(ChatDataset(train_data, hf_tok, mask_prompt=True))
        optimizer = optim.Adam(learning_rate=1e-4)

        args = TrainingArgs(
            batch_size=safe.batch_size,
            iters=iters,
            val_batches=0,
            steps_per_report=iters,
            steps_per_eval=iters + 1,
            steps_per_save=iters + 1,
            max_seq_length=safe.seq_length,
            adapter_file=str(Path(tempfile.gettempdir()) / "memory_guard_bench.safetensors"),
            grad_checkpoint=safe.grad_checkpoint,
        )

        model.train()
        try:
            train(model=model, optimizer=optimizer, train_dataset=train_set, args=args)
        except Exception as e:
            print(f"  Training failed: {e}")
            del model
            continue

        actual_mb = get_mlx_peak_memory_mb()
        del model  # Free memory before next config

        if actual_mb is None:
            print("  Could not read peak memory")
            continue

        error_pct = ((estimated_mb - actual_mb) / actual_mb) * 100
        direction = "over" if error_pct > 0 else "under"

        result = {
            "model": model_id,
            "batch_size": safe.batch_size,
            "seq_length": safe.seq_length,
            "lora_rank": safe.lora_rank,
            "lora_layers": safe.lora_layers,
            "grad_checkpoint": safe.grad_checkpoint,
            "estimated_mb": round(estimated_mb),
            "actual_mb": round(actual_mb),
            "error_pct": round(abs(error_pct), 1),
            "direction": direction,
            "platform": info.chip_name,
            "memory_gb": round(info.total_memory_mb / 1024),
        }
        results.append(result)

        print(f"  Estimated: {estimated_mb:.0f}MB")
        print(f"  Actual:    {actual_mb:.0f}MB")
        print(f"  Error:     {abs(error_pct):.1f}% ({direction}-estimate)")

        guard.record_result(actual_peak_mb=actual_mb, model_name=model_id)

    return {"results": results, "platform": info.chip_name, "memory_gb": round(info.total_memory_mb / 1024)}


def print_table(data: dict):
    results = data["results"]
    if not results:
        print("\nNo successful benchmarks.")
        return

    print(f"\n## Benchmark Results")
    print(f"Platform: {data['platform']} ({data['memory_gb']}GB)\n")
    print("| Model | Batch | Seq | Rank | Estimated | Actual | Error |")
    print("|-------|------:|----:|-----:|----------:|-------:|------:|")
    for r in results:
        model_short = r["model"].split("/")[-1][:30]
        print(f"| {model_short} | {r['batch_size']} | {r['seq_length']} | "
              f"{r['lora_rank']} | {r['estimated_mb']}MB | {r['actual_mb']}MB | "
              f"{r['error_pct']}% {r['direction']} |")


def print_submit(data: dict):
    results = data["results"]
    print("\n--- Copy below into a GitHub issue ---\n")
    print(f"**Platform**: {data['platform']} ({data['memory_gb']}GB)")
    import memory_guard
    print(f"**memory-guard version**: {memory_guard.__version__}")
    print(f"**Python**: {platform.python_version()}")
    print(f"**OS**: {platform.system()} {platform.release()}\n")
    print_table(data)
    print(f"\n<details><summary>Raw JSON</summary>\n")
    print(f"```json\n{json.dumps(data, indent=2)}\n```\n</details>")


def main():
    parser = argparse.ArgumentParser(description="Benchmark memory-guard accuracy")
    parser.add_argument("--model", default="mlx-community/Qwen3-0.6B-4bit",
                        help="Model to benchmark")
    parser.add_argument("--iters", type=int, default=10, help="Training iterations")
    parser.add_argument("--submit", action="store_true", help="GitHub issue template")
    args = parser.parse_args()

    data = run_benchmark(args.model, args.iters)
    print_table(data)

    if args.submit:
        print_submit(data)


if __name__ == "__main__":
    main()
