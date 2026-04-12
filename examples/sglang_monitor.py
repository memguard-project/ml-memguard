#!/usr/bin/env python3
"""SGLang + memguard: KV cache monitoring with RadixAttention-aware smoothing.

Demonstrates:
  1. guard_sglang() — pre-flight safe config (max_num_seqs, mem-fraction-static)
  2. KVCacheMonitor callbacks wired to on_warning and on_shed_load
  3. Why RadixAttention needs rolling-max smoothing and how it works
  4. Apple Silicon / Metal path (same API, different backend)

Run against a live SGLang server:

    pip install ml-memguard[sglang]
    python examples/sglang_monitor.py

Or with Apple Silicon (MLX Metal backend):

    pip install ml-memguard[sglang,apple]
    python examples/sglang_monitor.py --metal
"""

from __future__ import annotations

import argparse
import logging
import time

# ---------------------------------------------------------------------------
# Logging: show memguard signals in terminal
# ---------------------------------------------------------------------------

logging.basicConfig(
    format="%(asctime)s  %(levelname)-8s  %(name)s  %(message)s",
    datefmt="%H:%M:%S",
    level=logging.INFO,
)
logger = logging.getLogger("example")


# ---------------------------------------------------------------------------
# Step 1: pre-flight — safe max_num_seqs and mem-fraction-static
# ---------------------------------------------------------------------------

def run_preflight_demo(use_metal: bool = False) -> object:
    """Show what guard_sglang returns before the server starts.

    In production you would pass your actual SGLang Runtime object:

        import sglang as sgl
        runtime = sgl.Runtime(model_path="meta-llama/Llama-3.1-8B-Instruct",
                              mem_fraction_static=0.88)
        safe = guard_sglang(runtime)

    For this standalone demo we call preflight_inference() directly to
    avoid importing SGLang.

    SGLang-specific config note
    ---------------------------
    SGLang uses --mem-fraction-static (default 0.88) instead of vLLM's
    --gpu-memory-utilization.  guard_sglang reads this from server_args and
    passes it to preflight_inference() automatically.  The pre-flight output
    shows both the recommended max_num_seqs AND the optimal mem_fraction_static
    value for your model+GPU combination.
    """
    from memory_guard import preflight_inference

    # Llama 3.1 8B parameters — same model, different config than vLLM example
    # because SGLang's RadixAttention needs different KV budget headroom
    safe = preflight_inference(
        model_params=8_030_000_000,    # 8B parameters
        model_bits=16,                  # BF16 weights
        hidden_dim=4096,
        num_heads=32,
        num_kv_heads=8,                 # GQA — critical for SGLang KV budget calc
        num_layers=32,
        max_num_seqs=64,                # starting search point
        max_seq_len=4096,
        # SGLang reserves more headroom for RadixAttention prefix tree overhead
        # Use 0.85 instead of vLLM's 0.90 — RadixAttention tree nodes consume
        # ~5-8% additional GPU memory that vLLM's paged attention does not.
        gpu_memory_utilization=0.85,
    )

    print(safe)
    logger.info(
        "Pre-flight: max_num_seqs=%d  peak=%.0f MB  budget=%.0f MB  fits=%s",
        safe.max_num_seqs, safe.estimated_peak_mb, safe.budget_mb, safe.fits,
    )

    # SGLang CLI equivalent — what to pass at server start
    logger.info(
        "SGLang CLI: python -m sglang.launch_server --model-path <model> "
        "--max-running-requests %d --mem-fraction-static 0.85",
        safe.max_num_seqs,
    )

    return safe


# ---------------------------------------------------------------------------
# Step 2: KVCacheMonitor — RadixAttention-aware rolling-max smoothing
# ---------------------------------------------------------------------------

def run_monitor_demo(max_num_seqs: int) -> None:
    """Start the KV cache monitor against a live SGLang process.

    Why SGLang needs rolling-max smoothing
    --------------------------------------
    SGLang's RadixAttention prefix cache frees KV token slots when a cached
    prefix is evicted.  Without smoothing, this causes sudden utilization
    *drops* — e.g. from 88% down to 62% in one poll — which would reset any
    cooldown timer and delay the next shed-load signal by several polls.

    memguard applies a 3-reading rolling maximum to the raw utilization value:
    the monitor reports max(last_3_readings) instead of the current reading.
    This has two effects:
      1. Transient drops (RadixAttention evictions) don't falsely signal recovery
      2. Genuine sustained drops (scheduler successfully reclaiming memory) still
         clear the signal after 3 * poll_interval seconds (~15 s by default)

    The same smoothing is NOT applied to the vLLM adapter because vLLM's paged
    attention uses fixed-size blocks that do not spontaneously drop utilization.

    Dead-branch risk (issue #22373)
    --------------------------------
    Reasoning models (DeepSeek-R1, QwQ) can leave permanently dead branches
    in the RadixAttention tree — <think> tokens that are never reused but
    resist LRU eviction until downstream branches are also pruned.  With 50
    concurrent sessions these branches can consume 65-80 GB of wasted KV cache.
    The eviction_rate signal in on_shed_load reflects this: if eviction_rate
    is high but utilization is only moderate, dead branches are the likely cause.
    """
    from memory_guard.inference_monitor import KVCacheMonitor

    monitor = KVCacheMonitor(
        # SGLang's OpenAI-compatible server — same URL structure as vLLM
        vllm_url="http://localhost:30000",
        max_num_seqs=max_num_seqs,
        poll_interval=5.0,
        # SGLang-specific: enable rolling-max smoothing for RadixAttention
        # This is set automatically by guard_sglang() — shown here for clarity
        smoothing_window=3,
    )

    # --- on_warning fires at 80% rolling-max utilization ---
    # With RadixAttention, this may lag real pressure by ~15 s (3 polls).
    # Set warning_threshold to 0.75 if you need earlier signal.
    def on_warning(util: float) -> None:
        logger.warning(
            "RadixAttention KV cache WARNING: %.1f%% (rolling max) — "
            "check for dead prefix branches if eviction_rate is high",
            util * 100,
        )
        # Emit to your metrics system for dashboard alerting:
        # metrics.gauge("sglang.kvcache.util_rolling_max", util)

    # --- on_shed_load fires at 92% rolling-max utilization ---
    # This is a harder signal: even accounting for smoothing lag, the server
    # is genuinely memory-constrained.  Shed load immediately.
    def on_shed_load(util: float) -> None:
        logger.error(
            "RadixAttention KV cache SHED_LOAD: %.1f%% — stopping new requests",
            util * 100,
        )
        # Wire to your load balancer:
        #   nginx_upstream.set_weight(replica_id, 0)
        #   k8s_readiness.set_not_ready()
        #   lb.mark_backend_down(backend_id)

    monitor.on_warning   = on_warning
    monitor.on_shed_load = on_shed_load

    logger.info(
        "Monitor started — polling http://localhost:30000/metrics every 5s "
        "(rolling-max window=3 for RadixAttention eviction smoothing)"
    )
    logger.info("Press Ctrl-C to stop")

    with monitor.session():
        try:
            # In production: runtime.wait() or server.serve_forever()
            time.sleep(30)
        except KeyboardInterrupt:
            logger.info("Stopped by user")

    logger.info("Monitor stopped cleanly")


# ---------------------------------------------------------------------------
# Step 3: Apple Silicon / Metal backend
# ---------------------------------------------------------------------------

def run_metal_demo() -> None:
    """Run the same monitoring on Apple Silicon with the Metal backend.

    SGLang has growing Apple Silicon / MLX Metal support.  memguard works
    identically — the same guard_sglang() call, the same KVCacheMonitor
    callbacks — but uses Apple's Metal allocator for GPU memory accounting
    instead of CUDA.

    Apple Silicon OOM is silent: there is no CUDA OutOfMemoryError.  When
    unified memory is exhausted, macOS silently swaps to disk, the process
    freezes, and eventually the OS kills it.  This makes the on_shed_load
    callback even more critical — it is your only signal before the freeze.

    To use the Metal backend, install the apple extra:

        pip install ml-memguard[sglang,apple]

    Then launch SGLang with the Metal backend:

        python -m sglang.launch_server \\
            --model-path mlx-community/Meta-Llama-3-8B-Instruct-4bit \\
            --device metal \\
            --port 30000
    """
    logger.info("Apple Silicon / Metal backend example")
    logger.info(
        "Same API as CUDA — guard_sglang() and KVCacheMonitor work identically. "
        "The Metal allocator (mlx.core.metal) replaces torch.cuda for memory accounting."
    )
    logger.info(
        "Critical difference: Apple Silicon OOM is silent. on_shed_load is your only "
        "early warning before the process freezes. Set shed_load_threshold=0.80 (not 0.92) "
        "on Apple Silicon to shed load earlier and avoid the unresponsive-Mac problem."
    )

    # In production on Apple Silicon:
    #
    # from memory_guard import guard_sglang
    # import sglang as sgl
    #
    # runtime = sgl.Runtime(
    #     model_path="mlx-community/Meta-Llama-3-8B-Instruct-4bit",
    #     device="metal",
    # )
    # safe = guard_sglang(runtime)
    #
    # # Tighter threshold on Apple Silicon — no OOM exception to catch
    # safe.monitor.shed_load_threshold = 0.80
    # safe.monitor.on_shed_load = lambda u: lb.reduce_weight(host, 0)
    #
    # with safe.monitor.session():
    #     runtime.wait()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def _parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--metal", action="store_true",
                   help="Show Apple Silicon / Metal notes instead of CUDA")
    return p


if __name__ == "__main__":
    args = _parser().parse_args()

    print("=" * 60)
    print("Step 1: Pre-flight — safe max_num_seqs for SGLang")
    print("=" * 60)
    safe = run_preflight_demo(use_metal=args.metal)

    if args.metal:
        print()
        print("=" * 60)
        print("Step 2: Apple Silicon / Metal backend notes")
        print("=" * 60)
        run_metal_demo()
    else:
        print()
        print("=" * 60)
        print("Step 2: KV cache monitor (RadixAttention-aware)")
        print("=" * 60)
        run_monitor_demo(max_num_seqs=safe.max_num_seqs)
