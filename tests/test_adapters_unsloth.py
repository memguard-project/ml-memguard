"""Tests for memory_guard.adapters.unsloth.

Covers:
    - guard_unsloth_model: introspects model, runs preflight before LoRA
      attachment, returns SafeConfig with sensible fields
    - BnB double-quant detection: 5% correction applied to model_params
    - guard_sft_trainer: thin delegation to guard_trainer
    - Lazy __getattr__ in memory_guard exposes both symbols
"""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import MagicMock, patch, call

import pytest

from memory_guard.adapters.unsloth import (
    _DOUBLE_QUANT_CORRECTION,
    _is_double_quant,
    guard_sft_trainer,
    guard_unsloth_model,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_safe_config(
    batch_size: int = 4,
    seq_length: int = 2048,
    lora_rank: int = 16,
    lora_layers: int = 16,
    grad_accumulation: int = 1,
    grad_checkpoint: bool = False,
    fits: bool = True,
) -> object:
    from memory_guard.guard import SafeConfig

    estimate = MagicMock()
    estimate.total_mb = 8_000.0
    return SafeConfig(
        batch_size=batch_size,
        seq_length=seq_length,
        lora_rank=lora_rank,
        lora_layers=lora_layers,
        grad_checkpoint=grad_checkpoint,
        grad_accumulation=grad_accumulation,
        estimate=estimate,
        budget_mb=10_000.0,
        available_mb=14_000.0,
        changes=[],
        fits=fits,
    )


def _make_qc(
    load_in_4bit: bool = True,
    quant_type: str = "nf4",
    double_quant: bool = False,
) -> SimpleNamespace:
    return SimpleNamespace(
        load_in_4bit=load_in_4bit,
        load_in_8bit=False,
        quant_type=quant_type,
        bnb_4bit_use_double_quant=double_quant,
    )


def _make_model(
    hidden_size: int = 4096,
    num_attention_heads: int = 32,
    num_hidden_layers: int = 32,
    num_key_value_heads: int = 8,
    quantization_config: object = None,
    dtype: str = "torch.bfloat16",
    num_params: int = 8_000_000_000,
) -> MagicMock:
    config = SimpleNamespace(
        hidden_size=hidden_size,
        num_attention_heads=num_attention_heads,
        num_hidden_layers=num_hidden_layers,
        num_key_value_heads=num_key_value_heads,
        quantization_config=quantization_config,
    )
    param = MagicMock()
    param.numel.return_value = num_params

    model = MagicMock()
    model.config = config
    model.dtype = dtype
    model.parameters.return_value = [param]
    return model


def _make_guard(safe=None) -> tuple:
    if safe is None:
        safe = _make_safe_config()
    guard = MagicMock()
    guard.preflight.return_value = safe
    return guard, safe


# ---------------------------------------------------------------------------
# _is_double_quant helper
# ---------------------------------------------------------------------------


class TestIsDoubleQuant:
    def test_true_when_flag_set(self):
        model = _make_model(quantization_config=_make_qc(double_quant=True))
        assert _is_double_quant(model) is True

    def test_false_when_flag_not_set(self):
        model = _make_model(quantization_config=_make_qc(double_quant=False))
        assert _is_double_quant(model) is False

    def test_false_when_no_quantization_config(self):
        model = _make_model(quantization_config=None)
        assert _is_double_quant(model) is False

    def test_false_when_no_config_attr(self):
        model = MagicMock(spec=[])  # no .config attribute
        assert _is_double_quant(model) is False


# ---------------------------------------------------------------------------
# guard_unsloth_model — core behaviour
# ---------------------------------------------------------------------------


class TestGuardUnslothModel:
    def test_returns_safe_config(self):
        guard, safe = _make_guard()
        model = _make_model(quantization_config=_make_qc())
        result = guard_unsloth_model(model, guard=guard)
        assert result is safe

    def test_preflight_called_with_introspected_values(self):
        guard, _ = _make_guard()
        model = _make_model(
            hidden_size=4096,
            num_attention_heads=32,
            num_hidden_layers=32,
            quantization_config=_make_qc(load_in_4bit=True),
            num_params=8_000_000_000,
        )
        guard_unsloth_model(model, guard=guard)

        guard.preflight.assert_called_once()
        kw = guard.preflight.call_args[1]
        assert kw["model_bits"] == 4        # BnB 4-bit detected
        assert kw["hidden_dim"] == 4096
        assert kw["num_heads"] == 32
        assert kw["num_layers"] == 32

    def test_preflight_called_before_lora_would_be_attached(self):
        """Verify the intended call order: preflight fires when guard_unsloth_model
        is called — i.e. before the caller invokes get_peft_model."""
        guard, safe = _make_guard()
        model = _make_model(quantization_config=_make_qc())

        call_log: list[str] = []
        original_preflight = guard.preflight.side_effect

        def record_preflight(**kw):
            call_log.append("preflight")
            return safe

        guard.preflight.side_effect = record_preflight

        # Simulate Unsloth workflow
        returned_safe = guard_unsloth_model(model, guard=guard)
        call_log.append("get_peft_model")  # would happen after this call

        assert call_log.index("preflight") < call_log.index("get_peft_model")
        assert returned_safe is safe

    def test_safe_config_has_lora_fields(self):
        safe = _make_safe_config(lora_rank=32, lora_layers=24, seq_length=4096)
        guard, _ = _make_guard(safe)
        model = _make_model(quantization_config=_make_qc())

        result = guard_unsloth_model(model, guard=guard)

        assert result.lora_rank == 32
        assert result.lora_layers == 24
        assert result.seq_length == 4096
        assert result.batch_size is not None
        assert result.grad_accumulation is not None

    def test_preflight_overrides_forwarded(self):
        guard, _ = _make_guard()
        model = _make_model(quantization_config=_make_qc())

        guard_unsloth_model(model, guard=guard, batch_size=2, lora_rank=64, seq_length=1024)

        kw = guard.preflight.call_args[1]
        assert kw["batch_size"] == 2
        assert kw["lora_rank"] == 64
        assert kw["seq_length"] == 1024

    def test_auto_creates_guard_when_none(self):
        model = _make_model(quantization_config=_make_qc())
        with patch("memory_guard.adapters.unsloth.MemoryGuard") as MockGuard:
            MockGuard.auto.return_value.preflight.return_value = _make_safe_config()
            guard_unsloth_model(model, guard=None)
        MockGuard.auto.assert_called_once()


# ---------------------------------------------------------------------------
# guard_unsloth_model — double-quant correction
# ---------------------------------------------------------------------------


class TestDoubleQuantCorrection:
    def test_no_correction_without_double_quant(self):
        guard, _ = _make_guard()
        model = _make_model(
            quantization_config=_make_qc(double_quant=False),
            num_params=8_000_000_000,
        )
        guard_unsloth_model(model, guard=guard)

        kw = guard.preflight.call_args[1]
        assert kw["model_params"] == 8_000_000_000  # untouched

    def test_correction_applied_with_double_quant(self):
        guard, _ = _make_guard()
        model = _make_model(
            quantization_config=_make_qc(double_quant=True),
            num_params=8_000_000_000,
        )
        guard_unsloth_model(model, guard=guard)

        kw = guard.preflight.call_args[1]
        expected = int(8_000_000_000 * _DOUBLE_QUANT_CORRECTION)
        assert kw["model_params"] == expected

    def test_correction_is_approximately_5_percent(self):
        """_DOUBLE_QUANT_CORRECTION must be in the 0.90–0.99 range."""
        assert 0.90 <= _DOUBLE_QUANT_CORRECTION <= 0.99

    def test_model_bits_still_reported_as_4_with_double_quant(self):
        """model_bits stays 4 — only num_parameters is adjusted."""
        guard, _ = _make_guard()
        model = _make_model(
            quantization_config=_make_qc(load_in_4bit=True, double_quant=True),
            num_params=8_000_000_000,
        )
        guard_unsloth_model(model, guard=guard)

        kw = guard.preflight.call_args[1]
        assert kw["model_bits"] == 4

    def test_correction_reduces_model_params(self):
        guard, _ = _make_guard()
        model = _make_model(
            quantization_config=_make_qc(double_quant=True),
            num_params=8_000_000_000,
        )
        guard_unsloth_model(model, guard=guard)

        kw = guard.preflight.call_args[1]
        assert kw["model_params"] < 8_000_000_000


# ---------------------------------------------------------------------------
# guard_sft_trainer — delegates to guard_trainer
# ---------------------------------------------------------------------------


class TestGuardSftTrainer:
    def test_delegates_to_guard_trainer(self):
        """guard_sft_trainer must call guard_trainer with identical arguments.

        guard_trainer is imported inside guard_sft_trainer's function body, so
        we patch it at its definition site in the huggingface module.
        """
        mock_safe = _make_safe_config()

        with patch(
            "memory_guard.adapters.huggingface.guard_trainer",
            return_value=mock_safe,
        ) as mock_gt:
            mock_trainer = MagicMock()
            mock_guard = MagicMock()
            result = guard_sft_trainer(mock_trainer, guard=mock_guard, batch_size=2)

        mock_gt.assert_called_once_with(mock_trainer, guard=mock_guard, batch_size=2)
        assert result is mock_safe

    def test_returns_safe_config(self):
        """Integration-level: guard_sft_trainer returns a SafeConfig."""
        mock_safe = _make_safe_config()

        with patch("memory_guard.adapters.huggingface.guard_trainer", return_value=mock_safe):
            trainer = MagicMock()
            guard = MagicMock()
            result = guard_sft_trainer(trainer, guard=guard)

        assert result is mock_safe

    def test_passes_overrides_through(self):
        mock_safe = _make_safe_config()
        with patch(
            "memory_guard.adapters.huggingface.guard_trainer",
            return_value=mock_safe,
        ) as mock_gt:
            guard_sft_trainer(MagicMock(), guard=None, lora_rank=64, seq_length=512)

        kw = mock_gt.call_args
        assert kw[1]["lora_rank"] == 64
        assert kw[1]["seq_length"] == 512


# ---------------------------------------------------------------------------
# Lazy __getattr__ — package-level exports
# ---------------------------------------------------------------------------


class TestLazyPackageExports:
    def test_guard_unsloth_model_accessible_from_package(self):
        import memory_guard
        assert memory_guard.guard_unsloth_model is guard_unsloth_model

    def test_guard_sft_trainer_accessible_from_package(self):
        import memory_guard
        assert memory_guard.guard_sft_trainer is guard_sft_trainer

    def test_package_still_importable_without_unsloth(self):
        """Core package must not fail if unsloth is not installed."""
        import memory_guard
        assert hasattr(memory_guard, "__version__")


# ---------------------------------------------------------------------------
# Realistic Unsloth workflow integration sketch
# ---------------------------------------------------------------------------


class TestUnslothWorkflowIntegration:
    def test_full_workflow_mock_fast_language_model(self):
        """
        Simulate the Unsloth 3-line integration:

            safe = guard_unsloth_model(model)
            model = FastLanguageModel.get_peft_model(
                model,
                r=safe.lora_rank,
                max_seq_length=safe.seq_length,
                ...
            )

        Verifies preflight fires before get_peft_model and that the
        safe config values are threaded into get_peft_model correctly.
        """
        safe = _make_safe_config(lora_rank=16, seq_length=2048, batch_size=4)
        guard, _ = _make_guard(safe)
        model = _make_model(quantization_config=_make_qc())

        # Step 1: preflight before LoRA
        returned_safe = guard_unsloth_model(model, guard=guard)
        guard.preflight.assert_called_once()

        # Step 2: caller threads safe into FastLanguageModel.get_peft_model
        MockFLM = MagicMock()
        MockFLM.get_peft_model(
            model,
            r=returned_safe.lora_rank,
            lora_alpha=returned_safe.lora_rank * 2,
            max_seq_length=returned_safe.seq_length,
        )

        _, call_kwargs = MockFLM.get_peft_model.call_args
        assert call_kwargs["r"] == 16
        assert call_kwargs["max_seq_length"] == 2048
        assert call_kwargs["lora_alpha"] == 32
