"""Tests for the vLLM adapter (memory_guard.adapters.vllm).

All tests use MagicMock to simulate vLLM objects — vLLM is NOT required.

Covers:
  - _get_llm_engine: LLM, AsyncLLMEngine, bare LLMEngine
  - _extract_model_info: defaults, GQA, quantization, dtype, num_params
  - _make_poll_fn: normal path, list scheduler, missing block manager
  - guard_vllm: fits fast path, binary-search downgrade, fits=False floor,
                monitor wiring, pool_fn calls block manager, lazy import check
"""

from __future__ import annotations

import math
from unittest.mock import MagicMock, patch, PropertyMock

import pytest

from memory_guard.adapters.vllm import (
    _extract_model_info,
    _get_llm_engine,
    _make_poll_fn,
    _run_preflight,
    guard_vllm,
)
from memory_guard.inference_monitor import KVCacheMonitor
from memory_guard.guard import InferenceSafeConfig


# ---------------------------------------------------------------------------
# Helpers / fixtures
# ---------------------------------------------------------------------------

def _make_engine(
    num_heads=8,
    num_kv_heads=2,
    num_layers=4,
    hidden_size=512,
    max_model_len=2048,
    dtype_str="torch.float16",
    quantization=None,
    num_parameters=0,
    max_num_seqs=64,
    free_blocks=80,
    total_blocks=100,
) -> MagicMock:
    """Build a MagicMock that looks like a vLLM LLMEngine.

    Uses a restricted spec so that ``hasattr(engine, 'llm_engine')`` and
    ``hasattr(engine, 'engine')`` return False — preventing _get_llm_engine
    from mistaking this for an LLM / AsyncLLMEngine wrapper.
    """
    hf_config = MagicMock()
    hf_config.num_attention_heads = num_heads
    hf_config.num_key_value_heads = num_kv_heads
    hf_config.num_hidden_layers = num_layers
    hf_config.hidden_size = hidden_size
    hf_config.num_parameters = num_parameters

    model_config = MagicMock()
    model_config.hf_config = hf_config
    model_config.max_model_len = max_model_len
    model_config.dtype = dtype_str
    model_config.quantization = quantization

    scheduler_config = MagicMock()
    scheduler_config.max_num_seqs = max_num_seqs

    block_manager = MagicMock()
    block_manager.get_num_free_gpu_blocks.return_value = free_blocks
    block_manager.get_num_total_gpu_blocks.return_value = total_blocks

    scheduler = MagicMock()
    scheduler.block_manager = block_manager

    # Restrict spec to LLMEngine-only attrs so MagicMock does not auto-create
    # llm_engine / engine attributes (which would confuse _get_llm_engine).
    engine = MagicMock(spec=["model_config", "scheduler_config", "scheduler"])
    engine.model_config = model_config
    engine.scheduler_config = scheduler_config
    engine.scheduler = scheduler

    return engine


# ---------------------------------------------------------------------------
# _get_llm_engine
# ---------------------------------------------------------------------------

class TestGetLLMEngine:
    def test_llm_returns_llm_engine_attr(self):
        engine = MagicMock()
        llm = MagicMock()
        llm.llm_engine = engine
        # llm_engine takes precedence over engine
        assert _get_llm_engine(llm) is engine

    def test_async_llm_engine_returns_engine_attr(self):
        inner = MagicMock()
        # No llm_engine attr — only engine
        async_llm = MagicMock(spec=[])  # spec=[] removes all auto-attrs
        async_llm.engine = inner
        assert _get_llm_engine(async_llm) is inner

    def test_bare_llm_engine_returned_directly(self):
        engine = MagicMock(spec=[])  # no llm_engine, no engine
        assert _get_llm_engine(engine) is engine


# ---------------------------------------------------------------------------
# _extract_model_info
# ---------------------------------------------------------------------------

class TestExtractModelInfo:
    def test_num_kv_heads_reads_gqa_field(self):
        engine = _make_engine(num_heads=8, num_kv_heads=2)
        info = _extract_model_info(engine)
        assert info["num_kv_heads"] == 2

    def test_head_dim_is_hidden_over_num_heads(self):
        engine = _make_engine(num_heads=8, hidden_size=512)
        info = _extract_model_info(engine)
        assert info["head_dim"] == 512 // 8

    def test_num_layers_from_hf_config(self):
        engine = _make_engine(num_layers=12)
        info = _extract_model_info(engine)
        assert info["num_layers"] == 12

    def test_max_seq_len_from_model_config(self):
        engine = _make_engine(max_model_len=4096)
        info = _extract_model_info(engine)
        assert info["max_seq_len"] == 4096

    def test_dtype_fp16_gives_2_bytes(self):
        engine = _make_engine(dtype_str="torch.float16")
        assert _extract_model_info(engine)["dtype_bytes"] == 2

    def test_dtype_bfloat16_gives_2_bytes(self):
        engine = _make_engine(dtype_str="torch.bfloat16")
        assert _extract_model_info(engine)["dtype_bytes"] == 2

    def test_dtype_float32_gives_4_bytes(self):
        engine = _make_engine(dtype_str="torch.float32")
        assert _extract_model_info(engine)["dtype_bytes"] == 4

    def test_dtype_int8_gives_1_byte(self):
        engine = _make_engine(dtype_str="torch.int8")
        assert _extract_model_info(engine)["dtype_bytes"] == 1

    def test_quantization_awq_gives_4_bits(self):
        engine = _make_engine(quantization="awq")
        assert _extract_model_info(engine)["model_bits"] == 4

    def test_quantization_gptq_gives_4_bits(self):
        engine = _make_engine(quantization="gptq")
        assert _extract_model_info(engine)["model_bits"] == 4

    def test_quantization_none_gives_16_bits(self):
        engine = _make_engine(quantization=None)
        assert _extract_model_info(engine)["model_bits"] == 16

    def test_num_parameters_from_hf_config_when_nonzero(self):
        engine = _make_engine(num_parameters=7_000_000_000)
        info = _extract_model_info(engine)
        assert info["model_params"] == 7_000_000_000

    def test_num_parameters_estimated_when_zero(self):
        engine = _make_engine(num_parameters=0, num_heads=8, num_layers=4, hidden_size=512)
        info = _extract_model_info(engine)
        # rough formula: 12 × H² × L
        expected = int(12 * (512 ** 2) * 4)
        assert info["model_params"] == expected

    def test_defaults_when_no_model_config(self):
        engine = MagicMock(spec=[])  # no model_config
        info = _extract_model_info(engine)
        assert info["num_kv_heads"] == 32   # fallback num_heads
        assert info["num_layers"] == 32
        assert info["max_seq_len"] == 8192

    def test_hidden_dim_in_info(self):
        engine = _make_engine(hidden_size=2048)
        info = _extract_model_info(engine)
        assert info["hidden_dim"] == 2048


# ---------------------------------------------------------------------------
# _make_poll_fn
# ---------------------------------------------------------------------------

class TestMakePollFn:
    def test_returns_used_total_tuple(self):
        engine = _make_engine(free_blocks=70, total_blocks=100)
        poll = _make_poll_fn(engine)
        used, total = poll()
        assert total == 100
        assert used == 30  # 100 - 70

    def test_used_equals_total_minus_free(self):
        engine = _make_engine(free_blocks=1, total_blocks=10)
        poll = _make_poll_fn(engine)
        used, total = poll()
        assert used == 9
        assert total == 10

    def test_list_scheduler_uses_first_element(self):
        block_manager = MagicMock()
        block_manager.get_num_free_gpu_blocks.return_value = 50
        block_manager.get_num_total_gpu_blocks.return_value = 100
        sched = MagicMock()
        sched.block_manager = block_manager

        engine = MagicMock()
        engine.scheduler = [sched, MagicMock()]  # list — first is used
        poll = _make_poll_fn(engine)
        used, total = poll()
        assert total == 100
        assert used == 50

    def test_missing_block_manager_returns_zero_utilization(self):
        engine = MagicMock()
        engine.scheduler = MagicMock(spec=[])  # no block_manager attr
        poll = _make_poll_fn(engine)
        used, total = poll()
        assert total >= 1
        assert used == 0

    def test_missing_scheduler_returns_zero_utilization(self):
        engine = MagicMock(spec=[])  # no scheduler attr
        poll = _make_poll_fn(engine)
        used, total = poll()
        assert total >= 1
        assert used == 0


# ---------------------------------------------------------------------------
# _run_preflight
# ---------------------------------------------------------------------------

class TestRunPreflight:
    """Tests for the binary-search helper (no vLLM import needed)."""

    _BASE = dict(
        model_params=0,
        model_bits=16,
        num_kv_heads=2,
        head_dim=64,
        num_layers=4,
        dtype_bytes=2,
        hidden_dim=512,
    )

    def test_fast_path_when_fits(self):
        info = {**self._BASE, "max_seq_len": 512}
        # Budget larger than any estimate at 8 seqs
        result = _run_preflight(info, max_num_seqs=8, min_num_seqs=1,
                                available_mb=100_000, budget_mb=100_000)
        assert result.fits is True
        assert result.max_num_seqs == 8
        assert result.changes == []

    def test_downgrade_finds_largest_fitting(self):
        info = {**self._BASE, "max_seq_len": 2048}
        # Each seq costs 2 × 4 layers × 2 kv_heads × 64 head_dim × 2048 × 2 bytes
        # = 2 × 4 × 2 × 64 × 2048 × 2 = 4,194,304 bytes ≈ 4 MB / seq
        from memory_guard.estimator import estimate_serving_memory
        est_50 = estimate_serving_memory(max_num_seqs=50, **{k: v for k, v in info.items() if k != "max_seq_len"}, max_seq_len=info["max_seq_len"])
        est_51 = estimate_serving_memory(max_num_seqs=51, **{k: v for k, v in info.items() if k != "max_seq_len"}, max_seq_len=info["max_seq_len"])
        budget = (est_50.total_mb + est_51.total_mb) / 2

        result = _run_preflight(info, max_num_seqs=100, min_num_seqs=1,
                                available_mb=budget * 2, budget_mb=budget)
        assert result.max_num_seqs == 50
        assert result.fits is True

    def test_fits_false_when_nothing_fits(self):
        info = {**self._BASE, "max_seq_len": 2048}
        result = _run_preflight(info, max_num_seqs=4, min_num_seqs=2,
                                available_mb=1, budget_mb=0.001)  # impossibly small
        assert result.fits is False
        assert result.max_num_seqs == 2  # min_num_seqs floor

    def test_changes_list_contains_reduction(self):
        info = {**self._BASE, "max_seq_len": 2048}
        result = _run_preflight(info, max_num_seqs=100, min_num_seqs=1,
                                available_mb=0.1, budget_mb=0.001)
        assert len(result.changes) == 1
        assert "max_num_seqs" in result.changes[0]

    def test_gpu_memory_utilization_in_range(self):
        info = {**self._BASE, "max_seq_len": 512}
        result = _run_preflight(info, max_num_seqs=4, min_num_seqs=1,
                                available_mb=10_000, budget_mb=10_000)
        assert 0.0 <= result.gpu_memory_utilization <= 0.95


# ---------------------------------------------------------------------------
# guard_vllm (integration)
# ---------------------------------------------------------------------------

class TestGuardVllm:
    """End-to-end tests for guard_vllm with mocked vLLM and platform."""

    def _call(self, engine, **kwargs):
        """Call guard_vllm with mocked vllm import.

        Always supplies available_mb=50_000 unless the caller overrides it,
        so the lazy ``get_available_memory_mb`` import is never triggered.
        """
        kwargs.setdefault("available_mb", 50_000)
        with patch("memory_guard.adapters.vllm.optional_import"):
            return guard_vllm(engine, **kwargs)

    def test_returns_tuple_of_safe_config_and_monitor(self):
        engine = _make_engine()
        result = self._call(engine)
        assert isinstance(result, tuple) and len(result) == 2
        safe, mon = result
        assert isinstance(safe, InferenceSafeConfig)
        assert isinstance(mon, KVCacheMonitor)

    def test_monitor_not_started_on_return(self):
        engine = _make_engine()
        _, mon = self._call(engine)
        assert mon.is_running is False

    def test_max_num_seqs_read_from_scheduler_config(self):
        engine = _make_engine(max_num_seqs=128, free_blocks=80, total_blocks=100)
        safe, _ = self._call(engine, available_mb=100_000)
        # With 100_000 MB budget, 128 seqs should fit
        assert safe.max_num_seqs == 128

    def test_max_num_seqs_overridable_by_caller(self):
        engine = _make_engine(max_num_seqs=512)
        safe, _ = self._call(engine, max_num_seqs=64, available_mb=100_000)
        assert safe.max_num_seqs == 64

    def test_max_seq_len_overridable_by_caller(self):
        engine = _make_engine(max_model_len=8192)
        safe, _ = self._call(engine, max_seq_len=1024, available_mb=100_000)
        assert safe.max_seq_len == 1024

    def test_poll_fn_reads_block_manager(self):
        engine = _make_engine(free_blocks=70, total_blocks=100)
        _, mon = self._call(engine, available_mb=100_000)
        used, total = mon.poll_fn()
        assert total == 100
        assert used == 30

    def test_on_warning_callback_wired(self):
        fired = []
        engine = _make_engine()
        _, mon = self._call(engine, on_warning=lambda u: fired.append(u))
        assert mon.on_warning is not None
        mon.on_warning(0.85)
        assert fired == [0.85]

    def test_on_shed_load_callback_wired(self):
        fired = []
        engine = _make_engine()
        _, mon = self._call(engine, on_shed_load=lambda u: fired.append(u))
        assert mon.on_shed_load is not None
        mon.on_shed_load(0.95)
        assert fired == [0.95]

    def test_guard_vllm_raises_on_missing_vllm(self):
        engine = _make_engine()
        with patch(
            "memory_guard.adapters.vllm.optional_import",
            side_effect=ImportError("vllm required"),
        ):
            with pytest.raises(ImportError, match="vllm"):
                guard_vllm(engine)

    def test_llm_wrapper_unwrapped_correctly(self):
        """guard_vllm accepts an LLM wrapper (has llm_engine)."""
        engine = _make_engine()
        llm = MagicMock()
        llm.llm_engine = engine
        safe, _ = self._call(llm, available_mb=100_000)
        assert isinstance(safe, InferenceSafeConfig)


# ---------------------------------------------------------------------------
# Module-level exports
# ---------------------------------------------------------------------------

class TestModuleExports:
    def test_guard_vllm_importable(self):
        from memory_guard.adapters.vllm import guard_vllm as gv
        assert callable(gv)
