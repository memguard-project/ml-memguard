#!/usr/bin/env python3
"""vLLM + memguard: KV cache monitoring and auto-watchdog example.

Demonstrates:
  1. guard_vllm() — pre-flight safe config calculation
  2. KVCacheMonitor callbacks — on_warning and on_shed_load
  3. VLLMWatchdog — auto-restart when vLLM goes down

Run (requires a live vLLM server at localhost:8000):

    pip install ml-memguard[vllm]
    python examples/vllm_watchdog.py

To simulate against a local vLLM process:

    vllm serve meta-llama/Llama-3.1-8B-Instruct \
        --port 8000 \
        --max-num-seqs 16 \
        --gpu-memory-utilization 0.90 &

    python examples/vllm_watchdog.py
"""

from __future__ import annotations

import logging
import time

# ---------------------------------------------------------------------------
# Logging: make memguard output visible
# ---------------------------------------------------------------------------

logging.basicConfig(
    format="%(asctime)s  %(levelname)-8s  %(name)s  %(message)s",
    datefmt="%H:%M:%S",
    level=logging.INFO,
)
logger = logging.getLogger("example")


# ---------------------------------------------------------------------------
# Step 1: pre-flight — calculate the largest safe max_num_seqs
# ---------------------------------------------------------------------------

def run_preflight_demo() -> None:
    """Show what guard_vllm returns before the server starts.

    In production you would import from your actual vLLM engine:

        from vllm import AsyncLLMEngine, AsyncEngineArgs
        args   = AsyncEngineArgs(model="meta-llama/Llama-3.1-8B-Instruct")
        engine = AsyncLLMEngine.from_engine_args(args)
        safe   = guard_vllm(engine)

    For this standalone demo we call preflight_inference() directly with
    explicit architecture parameters to avoid importing vLLM.
    """
    from memory_guard import preflight_inference

    # Llama 3.1 8B architecture parameters
    safe = preflight_inference(
        model_params=8_030_000_000,    # 8B parameters
        model_bits=16,                  # BF16 serving weights
        hidden_dim=4096,                # Llama 3.1 8B hidden size
        num_heads=32,                   # attention heads
        num_kv_heads=8,                 # grouped-query attention
        num_layers=32,                  # transformer layers
        max_num_seqs=64,                # starting search point
        max_seq_len=4096,               # token context window
        gpu_memory_utilization=0.90,    # fraction of VRAM to use
    )

    # Print the full pre-flight report
    print(safe)

    logger.info(
        "Pre-flight complete: max_num_seqs=%d  estimated_peak=%.0f MB  budget=%.0f MB  status=%s",
        safe.max_num_seqs,
        safe.estimated_peak_mb,
        safe.budget_mb,
        "FITS" if safe.fits else "OOM",
    )

    return safe


# ---------------------------------------------------------------------------
# Step 2: KVCacheMonitor — wire load-shed callbacks
# ---------------------------------------------------------------------------

def run_monitor_demo(max_num_seqs: int) -> None:
    """Start the KV cache monitor against a live vLLM process.

    Callbacks fire when utilization crosses thresholds:
      on_warning   → 80%  — early heads-up, useful for dashboards
      on_shed_load → 92%  — stop routing new requests here

    Neither callback touches the vLLM engine — they only emit signals.
    Your load balancer or health endpoint decides what to do.
    """
    from memory_guard.inference_monitor import KVCacheMonitor

    # Create the monitor pointed at your vLLM metrics endpoint
    monitor = KVCacheMonitor(
        vllm_url="http://localhost:8000",   # vLLM's Prometheus metrics endpoint
        max_num_seqs=max_num_seqs,          # from the pre-flight result above
        poll_interval=5.0,                  # seconds between KV cache samples
    )

    # --- Callback: on_warning fires at 80% utilization ---
    # Wire this to your metrics system or a Slack webhook for early visibility.
    def on_warning(utilization: float) -> None:
        logger.warning(
            "KV cache WARNING: %.1f%% utilization — consider reducing upstream weight",
            utilization * 100,
        )
        # Example: emit a Prometheus gauge
        # metrics.gauge("vllm.kvcache.utilization", utilization)

    # --- Callback: on_shed_load fires at 92% utilization ---
    # Wire this to your load balancer, health endpoint, or Kubernetes probe.
    # The server keeps running — only new request routing should stop.
    def on_shed_load(utilization: float) -> None:
        logger.error(
            "KV cache SHED_LOAD: %.1f%% utilization — stopping new traffic to this replica",
            utilization * 100,
        )
        # Example load balancer integrations:
        #   load_balancer.reduce_weight(replica_id, weight=0)
        #   health_endpoint.set_not_ready()
        #   requests.post("http://nginx-admin/upstream/primary/weight/0")

    monitor.on_warning   = on_warning
    monitor.on_shed_load = on_shed_load

    # Run the monitor for 30 seconds (in production: wrap your serve_forever)
    logger.info("Monitor started — polling http://localhost:8000/metrics every 5s")
    logger.info("Press Ctrl-C to stop")

    with monitor.session():
        try:
            # In production this is your actual server loop:
            #   server.serve_forever()
            # Here we just sleep to let the monitor run.
            time.sleep(30)
        except KeyboardInterrupt:
            logger.info("Stopped by user")

    logger.info("Monitor stopped cleanly")


# ---------------------------------------------------------------------------
# Step 3: VLLMWatchdog — auto-restart when the process crashes
# ---------------------------------------------------------------------------

def run_watchdog_demo() -> None:
    """Start the auto-restart watchdog alongside your vLLM process.

    VLLMWatchdog polls the vLLM /health endpoint. When it detects the
    process is down (connection refused or non-200 response), it runs
    restart_cmd and waits for the server to come back online.

    In production, run this in a separate thread alongside your server:

        import threading
        watchdog_thread = threading.Thread(target=watchdog.start, daemon=True)
        watchdog_thread.start()
        server.serve_forever()
    """
    from memory_guard.inference_monitor import VLLMWatchdog

    watchdog = VLLMWatchdog(
        vllm_url="http://localhost:8000",
        # Shell command to restart vLLM — replace with your actual launch command
        restart_cmd=[
            "vllm", "serve", "meta-llama/Llama-3.1-8B-Instruct",
            "--port", "8000",
            "--max-num-seqs", "16",
            "--gpu-memory-utilization", "0.90",
        ],
        check_interval=10.0,    # seconds between /health polls
        restart_timeout=120.0,  # seconds to wait for vLLM to come back after restart
        max_restarts=5,         # stop watching after this many consecutive restarts
    )

    # Optional: get notified when a restart happens
    watchdog.alert_callback = lambda msg: logger.critical("WATCHDOG: %s", msg)

    logger.info(
        "Watchdog started — monitoring http://localhost:8000/health (check every 10s)"
    )

    # Run the watchdog for 60 seconds in this demo
    # In production: watchdog.start() blocks until max_restarts is reached or stopped
    import threading
    t = threading.Thread(target=watchdog.start, daemon=True)
    t.start()
    time.sleep(60)
    watchdog.stop()

    logger.info("Watchdog stopped")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    print("=" * 60)
    print("Step 1: Pre-flight — safe max_num_seqs calculation")
    print("=" * 60)
    safe = run_preflight_demo()

    print()
    print("=" * 60)
    print("Step 2: KV cache monitor with load-shed callbacks")
    print("=" * 60)
    run_monitor_demo(max_num_seqs=safe.max_num_seqs)

    print()
    print("=" * 60)
    print("Step 3: Auto-restart watchdog")
    print("=" * 60)
    run_watchdog_demo()
