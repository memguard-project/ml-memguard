"""Unsloth adapter for memory-guard.

Provides:
    guard_unsloth_model   — run preflight() before LoRA is attached, returning a
                            SafeConfig the caller threads into
                            FastLanguageModel.get_peft_model()
    guard_sft_trainer     — identical to guard_trainer() from the HF adapter but
                            explicitly documented for TRL / Unsloth SFTTrainer

Unsloth delegates its training loop to HuggingFace Trainer / TRL SFTTrainer, so
all mid-training downgrade logic from PRs 2–3 applies without modification.

BnB double-quantization correction
------------------------------------
Unsloth loads models with ``load_in_4bit=True`` and
``bnb_4bit_use_double_quant=True`` by default.  Double-quantization quantizes the
quantization constants themselves (~32 KB per layer → ~2 KB), saving roughly 5 %
of weight memory versus standard NF4.

Since :func:`~memory_guard.guard.MemoryGuard.preflight` accepts integer bits only,
we proxy the savings by multiplying ``num_parameters`` by
:data:`_DOUBLE_QUANT_CORRECTION` (0.95) before passing it to ``preflight``.  The
model is still reported as ``model_bits=4``; only the effective parameter count
shrinks.  Auto-calibration will refine the estimate after the first training run.
"""

from __future__ import annotations

import logging
from typing import Any, Optional

from ..guard import MemoryGuard, SafeConfig
from .base import introspect_model

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

_DOUBLE_QUANT_CORRECTION: float = 0.95
"""Multiplier applied to ``num_parameters`` when BnB double-quantization is
detected (``bnb_4bit_use_double_quant=True``).

Rationale: double-quant saves ~5 % of weight memory vs plain NF4 by also
quantizing the per-block quantization constants.  The savings are model- and
layer-count-dependent; 0.95 is a conservative floor.  Auto-calibration (3+
training runs) corrects the residual error automatically.
"""


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _is_double_quant(model: Any) -> bool:
    """Return ``True`` when the model uses BnB double-quantization."""
    config = getattr(model, "config", None)
    if config is None:
        return False
    qc = getattr(config, "quantization_config", None)
    if qc is None:
        return False
    return bool(getattr(qc, "bnb_4bit_use_double_quant", False))


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def guard_unsloth_model(
    model: Any,
    guard: Optional[Any] = None,
    **preflight_overrides: Any,
) -> SafeConfig:
    """Run preflight memory check on an Unsloth model before LoRA is attached.

    Call this **after** ``FastLanguageModel.from_pretrained`` but **before**
    ``FastLanguageModel.get_peft_model``.  Thread the returned
    :class:`~memory_guard.guard.SafeConfig` directly into ``get_peft_model``::

        model, tokenizer = FastLanguageModel.from_pretrained(
            "unsloth/Meta-Llama-3.1-8B-bnb-4bit",
            max_seq_length=2048,
            load_in_4bit=True,
        )
        safe = guard_unsloth_model(model)  # ← preflight before LoRA
        model = FastLanguageModel.get_peft_model(
            model,
            r=safe.lora_rank,
            lora_alpha=safe.lora_rank * 2,
            max_seq_length=safe.seq_length,
            target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
        )

    Steps:

    1. Introspects ``model.config`` to read architecture metadata.
    2. Detects BnB double-quantization and applies a 5 % parameter-count
       correction (see module docstring).
    3. Runs ``guard.preflight()`` with the introspected values (auto-downgrades
       if the config doesn't fit memory).

    Args:
        model: A model returned by ``FastLanguageModel.from_pretrained`` (or any
            HuggingFace-style model with a ``.config`` and ``.parameters()``).
        guard: A :class:`~memory_guard.MemoryGuard` instance.  ``None`` triggers
            ``MemoryGuard.auto()``.
        **preflight_overrides: Forwarded verbatim to ``guard.preflight()``
            (e.g. ``batch_size=4``, ``seq_length=2048``, ``lora_rank=16``).

    Returns:
        :class:`~memory_guard.guard.SafeConfig` — thread ``safe.lora_rank``,
        ``safe.lora_layers``, ``safe.seq_length``, and ``safe.batch_size`` into
        ``FastLanguageModel.get_peft_model`` and your ``SFTTrainer`` config.
    """
    if guard is None:
        guard = MemoryGuard.auto()

    model_info = introspect_model(model)

    # Apply double-quant correction before preflight
    num_params: int = model_info["num_parameters"]
    if _is_double_quant(model):
        corrected = int(num_params * _DOUBLE_QUANT_CORRECTION)
        logger.debug(
            "[memory-guard] BnB double-quant detected: applying %.0f%% correction "
            "to num_parameters (%d → %d). Auto-calibration will refine this.",
            (1 - _DOUBLE_QUANT_CORRECTION) * 100,
            num_params,
            corrected,
        )
        num_params = corrected

    preflight_kwargs: dict = dict(
        model_params=num_params,
        model_bits=model_info["model_bits"],
        hidden_dim=model_info["hidden_size"],
        num_heads=model_info["num_attention_heads"],
        num_layers=model_info["num_hidden_layers"],
    )
    preflight_kwargs.update(preflight_overrides)

    safe = guard.preflight(**preflight_kwargs)

    logger.info(
        "[memory-guard] guard_unsloth_model: lora_rank=%d, lora_layers=%d, "
        "seq_length=%d, batch_size=%d, grad_accum=%d",
        safe.lora_rank,
        safe.lora_layers,
        safe.seq_length,
        safe.batch_size,
        safe.grad_accumulation,
    )

    return safe


def guard_sft_trainer(
    trainer: Any,
    guard: Optional[Any] = None,
    **preflight_overrides: Any,
) -> SafeConfig:
    """Attach memory-guard to a TRL ``SFTTrainer`` in one call.

    ``SFTTrainer`` is a thin wrapper around HuggingFace ``Trainer``, so this
    function is identical to :func:`~memory_guard.adapters.huggingface.guard_trainer`
    but surfaced here so Unsloth workflows have a single named entry-point.

    Steps:

    1. Introspects ``trainer.model`` for architecture metadata.
    2. Runs ``guard.preflight()`` (auto-downgrades if needed).
    3. Writes the safe config into ``trainer.args``.
    4. Appends :class:`~memory_guard.adapters.huggingface.MemoryGuardCallback`
       to ``trainer.callback_handler`` — enabling mid-training downgrade (PR 3).

    Args:
        trainer: A ``trl.SFTTrainer`` or ``transformers.Trainer`` instance.
        guard: A :class:`~memory_guard.MemoryGuard` instance.  ``None`` → auto.
        **preflight_overrides: Forwarded verbatim to ``guard.preflight()``.

    Returns:
        :class:`~memory_guard.guard.SafeConfig` from preflight.
    """
    from .huggingface import guard_trainer
    return guard_trainer(trainer, guard=guard, **preflight_overrides)
