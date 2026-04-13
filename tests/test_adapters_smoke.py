"""Integration smoke test: guard_trainer + MemoryGuardCallback on real HF objects.

Skipped unless both ``transformers`` and ``torch`` are installed.
Run this test with:  pip install ml-memguard[hf] && pytest tests/test_adapters_smoke.py

Downloads distilgpt2 (~82 MB) on first run; subsequent runs use the local cache
at ~/.cache/huggingface.  Requires network access or a pre-warmed HF cache.

The test does NOT exercise quantization or LoRA — it uses distilgpt2 in fp32 to
keep the setup dependency-free (no bitsandbytes, no peft).  The goal is to
verify that the entire callback chain (on_train_begin, on_step_begin,
on_epoch_begin, on_log, on_train_end) executes without raising on real HF objects.
"""

from __future__ import annotations

import pytest

# ------------------------------------------------------------------
# Availability gate — skip the entire module if deps are absent.
# pytest.importorskip at module level skips with a clean message.
# ------------------------------------------------------------------

pytest.importorskip(
    "transformers",
    reason="requires transformers (pip install ml-memguard[hf])",
)
pytest.importorskip(
    "torch",
    reason="requires torch (pip install ml-memguard[hf])",
)
pytest.importorskip(
    "accelerate",
    reason="requires accelerate>=1.1.0 (pip install ml-memguard[hf])",
)

from transformers import (  # noqa: E402 — after importorskip
    AutoModelForCausalLM,
    Trainer,
    TrainingArguments,
)


# ------------------------------------------------------------------
# Minimal in-memory dataset — avoids the `datasets` library
# ------------------------------------------------------------------


class _TinyDataset:
    """Eight 16-token sequences: enough for 2 steps at batch_size=4."""

    def __init__(self, size: int = 8, seq_len: int = 16) -> None:
        self._rows = [
            {
                "input_ids": list(range(seq_len)),
                "attention_mask": [1] * seq_len,
                "labels": list(range(seq_len)),
            }
            for _ in range(size)
        ]

    def __len__(self) -> int:
        return len(self._rows)

    def __getitem__(self, idx: int) -> dict:
        return self._rows[idx]


# ------------------------------------------------------------------
# Smoke test
# ------------------------------------------------------------------


@pytest.mark.filterwarnings("ignore::FutureWarning")   # no_cuda deprecation
@pytest.mark.filterwarnings("ignore::UserWarning")     # HF misc warnings
def test_guard_trainer_smoke_distilgpt2(tmp_path):
    """guard_trainer survives 2 real training steps on distilgpt2 (CPU, fp32).

    Assertions:
    - guard_trainer returns a SafeConfig without raising
    - trainer.args is patched with the safe batch_size
    - trainer.train() completes 2 steps (no exception from callback chain)
    - guard.record_result() is reached (on_train_end fires)
    """
    from unittest.mock import patch

    from memory_guard import guard_trainer

    # Load tiny model — fp32, no quantization, no LoRA
    model = AutoModelForCausalLM.from_pretrained("distilgpt2")

    # Force CPU — use_cpu (>=4.38) with fallback to no_cuda for older versions
    _cpu_kwargs: dict = {}
    import inspect
    _ta_params = inspect.signature(TrainingArguments.__init__).parameters
    if "use_cpu" in _ta_params:
        _cpu_kwargs["use_cpu"] = True
    elif "no_cuda" in _ta_params:
        _cpu_kwargs["no_cuda"] = True

    training_args = TrainingArguments(
        output_dir=str(tmp_path),
        max_steps=2,
        per_device_train_batch_size=4,
        report_to="none",           # no wandb / tensorboard
        logging_steps=1,
        save_steps=9_999,           # no checkpoint saves during the smoke run
        dataloader_num_workers=0,
        **_cpu_kwargs,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=_TinyDataset(),
    )

    # guard_trainer must not raise and must return a SafeConfig
    safe = guard_trainer(trainer)

    assert safe is not None, "guard_trainer returned None"
    assert safe.batch_size >= 1
    assert safe.grad_accumulation >= 1

    # trainer.args must be patched
    assert trainer.args.per_device_train_batch_size == safe.batch_size, (
        f"trainer.args not patched: got {trainer.args.per_device_train_batch_size}, "
        f"expected {safe.batch_size}"
    )

    # Verify MemoryGuardCallback is present in callback_handler
    from memory_guard.adapters.huggingface import MemoryGuardCallback
    callbacks = trainer.callback_handler.callbacks
    assert any(isinstance(cb, MemoryGuardCallback) for cb in callbacks), (
        "MemoryGuardCallback not found in trainer.callback_handler.callbacks"
    )

    # Patch record_result to spy on it without touching disk calibration store
    from memory_guard.guard import MemoryGuard
    original_record = MemoryGuard.record_result

    record_result_called = []

    def _spy_record_result(self, *args, **kwargs):
        record_result_called.append(True)
        return original_record(self, *args, **kwargs)

    with patch.object(MemoryGuard, "record_result", _spy_record_result):
        # Full 2-step training — callback chain must complete without exception
        trainer.train()

    # on_train_end must have called record_result
    assert record_result_called, "on_train_end did not call guard.record_result()"
