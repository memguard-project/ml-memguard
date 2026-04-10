"""vLLM adapter for memory-guard.

Provides:
    guard_vllm — introspect a running vLLM engine, run preflight_inference()
                 to find a safe max_num_seqs, and wire a KVCacheMonitor to
                 the engine's block manager.

Design contract (ADR 003)
--------------------------
``guard_vllm`` constructs and returns a ``KVCacheMonitor``.  It never calls
into the engine except through the returned monitor's ``poll_fn``.  The
``poll_fn`` reads ``used_blocks`` and ``total_blocks`` from the block manager
but never writes to it.  All load-shedding decisions are delegated to the
caller via ``on_warning`` and ``on_shed_load`` callbacks.

Supported vLLM object types
----------------------------
- ``vllm.LLM``             — offline batch inference wrapper; exposes
                             the engine as ``llm.llm_engine``.
- ``vllm.AsyncLLMEngine``  — async serving engine; exposes it as
                             ``llm.engine``.
- ``vllm.LLMEngine``       — the raw engine (passed through directly).

Usage::

    from vllm import LLM
    from memory_guard.adapters.vllm import guard_vllm

    llm = LLM(model="meta-llama/Meta-Llama-3-8B", ...)

    safe, monitor = guard_vllm(
        llm,
        on_shed_load=lambda u: load_balancer.reduce_weight("primary", 0),
    )

    print(safe)                         # InferenceSafeConfig
    print(safe.max_num_seqs)            # Pass to vLLM --max-num-seqs
    print(safe.gpu_memory_utilization)  # Pass to vLLM --gpu-memory-utilization

    with monitor.session():
        server.serve_forever()
"""

from __future__ import annotations

import logging
from typing import Callable, Optional

from ..constants import MONITOR_POLL_INTERVAL, OVERHEAD_RATIO_INFERENCE, FIXED_OVERHEAD_MB, SAFETY_RATIO_DEFAULT
from ..estimator import InferenceServingEstimate, estimate_serving_memory
from ..guard import InferenceSafeConfig
from ..inference_monitor import KVCacheMonitor
from .base import optional_import

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def guard_vllm(
    llm: object,
    available_mb: Optional[float] = None,
    max_num_seqs: Optional[int] = None,
    max_seq_len: Optional[int] = None,
    on_warning: Optional[Callable[[float], None]] = None,
    on_shed_load: Optional[Callable[[float], None]] = None,
    poll_interval: float = MONITOR_POLL_INTERVAL,
    cooldown_seconds: float = 30.0,
    safety_ratio: float = SAFETY_RATIO_DEFAULT,
    min_num_seqs: int = 1,
) -> tuple[InferenceSafeConfig, KVCacheMonitor]:
    """Preflight check and KV cache monitor for a vLLM engine.

    Reads the model architecture from the engine's ``model_config``, runs a
    binary search to find the largest ``max_num_seqs`` that fits in the GPU
    memory budget, and returns a wired-but-unstarted ``KVCacheMonitor``.

    The monitor polls ``block_manager.get_num_free_gpu_blocks()`` from a
    daemon thread and fires ``on_warning`` / ``on_shed_load`` callbacks —
    it never mutates the engine (ADR 003).

    Args:
        llm:               A ``vllm.LLM``, ``vllm.AsyncLLMEngine``, or
                           ``vllm.LLMEngine`` instance.
        available_mb:      Available GPU memory in MB.  Auto-detected from
                           CUDA if not provided.
        max_num_seqs:      Max concurrent requests for the preflight check.
                           Defaults to ``engine.scheduler_config.max_num_seqs``.
        max_seq_len:       Max sequence length.  Defaults to
                           ``engine.model_config.max_model_len``.
        on_warning:        Callback fired at ≥ 80 % KV cache utilization.
                           Receives the utilization float (0.0–1.0).
        on_shed_load:      Callback fired at ≥ 92 % KV cache utilization.
                           Receives the utilization float (0.0–1.0).
        poll_interval:     Seconds between block-manager polls (default 5 s).
        cooldown_seconds:  Minimum seconds between repeated callback firings.
        safety_ratio:      Fraction of available memory used as the budget.
        min_num_seqs:      Binary-search floor — never reduce below this.

    Returns:
        ``(InferenceSafeConfig, KVCacheMonitor)`` — the preflight result and
        an **unstarted** monitor.  Start it with ``monitor.start()`` or use
        ``with monitor.session(): ...``.

    Raises:
        ImportError: if vLLM is not installed
            (``pip install ml-memguard[vllm]``).
    """
    optional_import("vllm", "vllm")

    engine = _get_llm_engine(llm)
    info = _extract_model_info(engine)

    if max_seq_len is not None:
        info["max_seq_len"] = max_seq_len

    if max_num_seqs is None:
        sc = getattr(engine, "scheduler_config", None)
        max_num_seqs = getattr(sc, "max_num_seqs", 256) if sc else 256

    if available_mb is None:
        from ..platforms import get_available_memory_mb
        available_mb = get_available_memory_mb()

    budget_mb = available_mb * safety_ratio

    safe_config = _run_preflight(
        info=info,
        max_num_seqs=max_num_seqs,
        min_num_seqs=min_num_seqs,
        available_mb=available_mb,
        budget_mb=budget_mb,
    )

    poll_fn = _make_poll_fn(engine)

    monitor = KVCacheMonitor(
        poll_fn=poll_fn,
        poll_interval=poll_interval,
        on_warning=on_warning,
        on_shed_load=on_shed_load,
        cooldown_seconds=cooldown_seconds,
    )

    return safe_config, monitor


# ---------------------------------------------------------------------------
# Internal helpers — engine introspection
# ---------------------------------------------------------------------------


def _get_llm_engine(llm: object) -> object:
    """Return the underlying ``LLMEngine`` from any vLLM wrapper type.

    - ``vllm.LLM``            exposes the engine as ``llm.llm_engine``
    - ``vllm.AsyncLLMEngine`` exposes it as ``llm.engine``
    - ``vllm.LLMEngine``      is returned directly
    """
    if hasattr(llm, "llm_engine"):
        return llm.llm_engine
    if hasattr(llm, "engine"):
        return llm.engine
    return llm


def _extract_model_info(engine: object) -> dict:
    """Read model architecture metadata from a vLLM LLMEngine.

    Returns a dict with keys matching ``estimate_serving_memory()`` kwargs:
    ``model_params``, ``model_bits``, ``num_kv_heads``, ``head_dim``,
    ``num_layers``, ``max_seq_len``, ``dtype_bytes``, ``hidden_dim``.
    """
    mc = getattr(engine, "model_config", None)
    hf = getattr(mc, "hf_config", None) if mc else None

    # --- attention geometry -------------------------------------------------
    num_heads: int = getattr(hf, "num_attention_heads", 32) if hf else 32
    num_kv_heads: int = getattr(hf, "num_key_value_heads", num_heads) if hf else num_heads
    num_layers: int = getattr(hf, "num_hidden_layers", 32) if hf else 32
    hidden_size: int = getattr(hf, "hidden_size", 4096) if hf else 4096
    head_dim: int = hidden_size // num_heads if num_heads > 0 else 128

    # --- sequence length ----------------------------------------------------
    max_seq_len: int = getattr(mc, "max_model_len", 8192) if mc else 8192

    # --- dtype → bytes ------------------------------------------------------
    dtype = getattr(mc, "dtype", None) if mc else None
    dtype_str = str(dtype) if dtype is not None else ""
    if "float32" in dtype_str:
        dtype_bytes = 4
    elif "int8" in dtype_str:
        dtype_bytes = 1
    else:
        dtype_bytes = 2  # fp16 / bf16

    # --- quantization → model_bits ------------------------------------------
    quantization: Optional[str] = getattr(mc, "quantization", None) if mc else None
    _4BIT = {"awq", "awq_marlin", "gptq", "gptq_marlin", "squeezellm", "fp8"}
    _8BIT = {"bitsandbytes", "smooth_quant"}
    if quantization in _4BIT:
        model_bits = 4
    elif quantization in _8BIT:
        model_bits = 8
    else:
        model_bits = 16  # unquantized fp16/bf16

    # --- parameter count ----------------------------------------------------
    # Prefer hf_config attribute; fall back to a standard transformer estimate.
    num_params: int = getattr(hf, "num_parameters", 0) if hf else 0
    if num_params == 0:
        # Standard transformer approximation: 12 × H² × L
        # (attention + FFN params per layer, reasonable for most LLMs)
        num_params = int(12 * (hidden_size ** 2) * num_layers)

    return {
        "model_params": num_params,
        "model_bits": model_bits,
        "num_kv_heads": num_kv_heads,
        "head_dim": head_dim,
        "num_layers": num_layers,
        "max_seq_len": max_seq_len,
        "dtype_bytes": dtype_bytes,
        "hidden_dim": hidden_size,
    }


# ---------------------------------------------------------------------------
# Internal helpers — poll_fn construction
# ---------------------------------------------------------------------------


def _make_poll_fn(engine: object) -> Callable[[], tuple[int, int]]:
    """Return a zero-argument callable that reads ``(used, total)`` GPU blocks.

    Accesses ``engine.scheduler.block_manager``.  In vLLM ≥ 0.4.0, the
    scheduler may be a list; the first element is used in that case.
    """
    scheduler = getattr(engine, "scheduler", None)
    if isinstance(scheduler, list):
        scheduler = scheduler[0] if scheduler else None

    block_manager = getattr(scheduler, "block_manager", None) if scheduler else None

    if block_manager is None:
        logger.warning(
            "[memory-guard] vLLM block manager not found on engine.scheduler; "
            "KVCacheMonitor will return 0 utilization.  "
            "Verify you are using a supported vLLM version."
        )

        def _null_poll() -> tuple[int, int]:
            return 0, 1  # utilization = 0 (harmless no-op)

        return _null_poll

    def _poll() -> tuple[int, int]:
        free: int = block_manager.get_num_free_gpu_blocks()
        total: int = block_manager.get_num_total_gpu_blocks()
        used: int = total - free
        return used, total

    return _poll


# ---------------------------------------------------------------------------
# Internal helpers — preflight binary search
# ---------------------------------------------------------------------------


def _run_preflight(
    info: dict,
    max_num_seqs: int,
    min_num_seqs: int,
    available_mb: float,
    budget_mb: float,
) -> InferenceSafeConfig:
    """Binary-search for the largest ``max_num_seqs`` within *budget_mb*.

    Mirrors ``MemoryGuard.preflight_inference()`` but accepts an explicit
    budget so the adapter can pass a caller-supplied or auto-detected value.
    """
    _kw = {k: v for k, v in info.items() if k != "max_seq_len"}
    _kw["max_seq_len"] = info["max_seq_len"]

    # Fast path — requested config fits
    est = estimate_serving_memory(max_num_seqs=max_num_seqs, **_kw)
    if est.fits_in(budget_mb):
        gpu_util = min(0.95, est.total_mb / available_mb) if available_mb > 0 else 0.90
        return InferenceSafeConfig(
            max_num_seqs=max_num_seqs,
            max_seq_len=info["max_seq_len"],
            gpu_memory_utilization=round(gpu_util, 4),
            estimate=est,
            budget_mb=budget_mb,
            available_mb=available_mb,
            fits=True,
            changes=[],
        )

    logger.warning(
        "[memory-guard] vLLM preflight: %d seqs × %d tokens = %.0f MB "
        "exceeds budget %.0f MB. Binary-searching for safe max_num_seqs...",
        max_num_seqs, info["max_seq_len"], est.total_mb, budget_mb,
    )

    lo, hi = min_num_seqs, max_num_seqs
    safe_seqs: Optional[int] = None
    safe_est: Optional[InferenceServingEstimate] = None

    while lo <= hi:
        mid = (lo + hi) // 2
        candidate = estimate_serving_memory(max_num_seqs=mid, **_kw)
        if candidate.fits_in(budget_mb):
            safe_seqs = mid
            safe_est = candidate
            lo = mid + 1
        else:
            hi = mid - 1

    fits = safe_seqs is not None
    if not fits:
        safe_seqs = min_num_seqs
        safe_est = estimate_serving_memory(max_num_seqs=min_num_seqs, **_kw)

    gpu_util = min(0.95, safe_est.total_mb / available_mb) if available_mb > 0 else 0.90
    changes = [f"max_num_seqs reduced {max_num_seqs} → {safe_seqs}"]

    logger.warning(
        "[memory-guard] vLLM safe max_num_seqs: %d (%.0f MB). "
        "Suggested: vllm --max-num-seqs=%d --gpu-memory-utilization=%.4f",
        safe_seqs, safe_est.total_mb, safe_seqs, gpu_util,
    )

    return InferenceSafeConfig(
        max_num_seqs=safe_seqs,
        max_seq_len=info["max_seq_len"],
        gpu_memory_utilization=round(gpu_util, 4),
        estimate=safe_est,
        budget_mb=budget_mb,
        available_mb=available_mb,
        fits=fits,
        changes=changes,
    )
