"""Tests for estimate_serving_memory() and preflight_inference().

Covers:
  - KV cache formula correctness and scaling properties
  - Model weight calculation across quantization levels
  - GQA vs MHA head count handling
  - dtype_bytes (fp16 vs fp32 vs int8)
  - Activation buffers (hidden_dim present / absent)
  - fits_in() and __str__()
  - preflight_inference() binary search, downgrade, min_num_seqs floor
  - InferenceSafeConfig fields and __str__()
"""

from __future__ import annotations

from unittest.mock import patch

import pytest

from memory_guard import (
    InferenceSafeConfig,
    InferenceServingEstimate,
    MemoryGuard,
    estimate_serving_memory,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_MB = 1024 * 1024

def _guard_with_budget(budget_mb: float) -> MemoryGuard:
    """Return a MemoryGuard whose budget_mb and available_mb are fixed."""
    guard = MemoryGuard.auto()
    # available_mb drives budget_mb (budget ≈ available × safety_ratio + swap)
    # Patch get_available_memory_mb so available_mb ≈ budget_mb / safety_ratio.
    available = budget_mb / guard.safety_ratio
    guard._patched_available = available
    return guard


# ---------------------------------------------------------------------------
# KV cache formula
# ---------------------------------------------------------------------------

class TestKVCacheFormula:
    """2 × layers × kv_heads × head_dim × max_seq_len × max_num_seqs × dtype_bytes"""

    def test_exact_formula_values(self):
        # 2 layers, 2 kv_heads, 4 head_dim, 8 max_seq_len, 1 seq, 2 bytes
        # Expected bytes: 2(K+V) × 2(layers) × 2(kv_heads) × 4(head_dim)
        #                 × 8(seq_len) × 1(num_seqs) × 2(dtype) = 512
        est = estimate_serving_memory(
            model_params=0, model_bits=16,
            num_kv_heads=2, head_dim=4, num_layers=2,
            max_num_seqs=1, max_seq_len=8, dtype_bytes=2,
        )
        expected_kv_mb = 512 / _MB
        assert abs(est.kv_cache_mb - expected_kv_mb) < 1e-9

    def test_kv_cache_linear_with_num_seqs(self):
        base = estimate_serving_memory(
            model_params=0, model_bits=16, num_kv_heads=8, head_dim=128,
            num_layers=32, max_num_seqs=64, max_seq_len=512, dtype_bytes=2,
        )
        doubled = estimate_serving_memory(
            model_params=0, model_bits=16, num_kv_heads=8, head_dim=128,
            num_layers=32, max_num_seqs=128, max_seq_len=512, dtype_bytes=2,
        )
        assert abs(doubled.kv_cache_mb / base.kv_cache_mb - 2.0) < 1e-6

    def test_kv_cache_linear_with_seq_len(self):
        base = estimate_serving_memory(
            model_params=0, model_bits=16, num_kv_heads=8, head_dim=128,
            num_layers=32, max_num_seqs=64, max_seq_len=512, dtype_bytes=2,
        )
        doubled = estimate_serving_memory(
            model_params=0, model_bits=16, num_kv_heads=8, head_dim=128,
            num_layers=32, max_num_seqs=64, max_seq_len=1024, dtype_bytes=2,
        )
        assert abs(doubled.kv_cache_mb / base.kv_cache_mb - 2.0) < 1e-6

    def test_kv_cache_linear_with_num_layers(self):
        base = estimate_serving_memory(
            model_params=0, model_bits=16, num_kv_heads=8, head_dim=128,
            num_layers=16, max_num_seqs=64, max_seq_len=512, dtype_bytes=2,
        )
        doubled = estimate_serving_memory(
            model_params=0, model_bits=16, num_kv_heads=8, head_dim=128,
            num_layers=32, max_num_seqs=64, max_seq_len=512, dtype_bytes=2,
        )
        assert abs(doubled.kv_cache_mb / base.kv_cache_mb - 2.0) < 1e-6

    def test_fp32_doubles_kv_cache_vs_fp16(self):
        fp16 = estimate_serving_memory(
            model_params=0, model_bits=16, num_kv_heads=8, head_dim=128,
            num_layers=32, max_num_seqs=64, max_seq_len=512, dtype_bytes=2,
        )
        fp32 = estimate_serving_memory(
            model_params=0, model_bits=16, num_kv_heads=8, head_dim=128,
            num_layers=32, max_num_seqs=64, max_seq_len=512, dtype_bytes=4,
        )
        assert abs(fp32.kv_cache_mb / fp16.kv_cache_mb - 2.0) < 1e-6

    def test_int8_kv_cache_half_of_fp16(self):
        int8 = estimate_serving_memory(
            model_params=0, model_bits=16, num_kv_heads=8, head_dim=128,
            num_layers=32, max_num_seqs=64, max_seq_len=512, dtype_bytes=1,
        )
        fp16 = estimate_serving_memory(
            model_params=0, model_bits=16, num_kv_heads=8, head_dim=128,
            num_layers=32, max_num_seqs=64, max_seq_len=512, dtype_bytes=2,
        )
        assert abs(fp16.kv_cache_mb / int8.kv_cache_mb - 2.0) < 1e-6

    def test_gqa_uses_kv_heads_not_num_attention_heads(self):
        # GQA Llama-style: 8 kv_heads; MHA equivalent would be 32 kv_heads
        gqa = estimate_serving_memory(
            model_params=0, model_bits=16, num_kv_heads=8, head_dim=128,
            num_layers=32, max_num_seqs=64, max_seq_len=512, dtype_bytes=2,
        )
        mha = estimate_serving_memory(
            model_params=0, model_bits=16, num_kv_heads=32, head_dim=128,
            num_layers=32, max_num_seqs=64, max_seq_len=512, dtype_bytes=2,
        )
        assert abs(mha.kv_cache_mb / gqa.kv_cache_mb - 4.0) < 1e-6


# ---------------------------------------------------------------------------
# Model weights
# ---------------------------------------------------------------------------

class TestModelWeights:
    def test_4bit_7b_weight_formula(self):
        # 7B × 0.5 bytes = ~3,338 MB
        est = estimate_serving_memory(
            model_params=7_000_000_000, model_bits=4,
            num_kv_heads=8, head_dim=128, num_layers=32,
            max_num_seqs=1, max_seq_len=1, dtype_bytes=2,
        )
        expected_mb = (7e9 * 0.5) / _MB
        assert abs(est.model_weights_mb - expected_mb) < 1.0

    def test_16bit_weights_4x_larger_than_4bit(self):
        est4 = estimate_serving_memory(
            model_params=7_000_000_000, model_bits=4,
            num_kv_heads=8, head_dim=128, num_layers=32,
            max_num_seqs=1, max_seq_len=1, dtype_bytes=2,
        )
        est16 = estimate_serving_memory(
            model_params=7_000_000_000, model_bits=16,
            num_kv_heads=8, head_dim=128, num_layers=32,
            max_num_seqs=1, max_seq_len=1, dtype_bytes=2,
        )
        assert abs(est16.model_weights_mb / est4.model_weights_mb - 4.0) < 0.01

    def test_zero_params_gives_zero_weights(self):
        est = estimate_serving_memory(
            model_params=0, model_bits=16, num_kv_heads=8, head_dim=128,
            num_layers=4, max_num_seqs=1, max_seq_len=1, dtype_bytes=2,
        )
        assert est.model_weights_mb == 0.0


# ---------------------------------------------------------------------------
# Activations and total
# ---------------------------------------------------------------------------

class TestActivationsAndTotal:
    def test_activations_zero_when_hidden_dim_not_provided(self):
        est = estimate_serving_memory(
            model_params=0, model_bits=16, num_kv_heads=8, head_dim=128,
            num_layers=32, max_num_seqs=64, max_seq_len=512, dtype_bytes=2,
        )
        assert est.activations_mb == 0.0

    def test_activations_nonzero_when_hidden_dim_provided(self):
        est = estimate_serving_memory(
            model_params=0, model_bits=16, num_kv_heads=8, head_dim=128,
            num_layers=32, max_num_seqs=64, max_seq_len=512, dtype_bytes=2,
            hidden_dim=4096,
        )
        assert est.activations_mb > 0.0

    def test_activations_scale_linearly_with_num_seqs(self):
        est1 = estimate_serving_memory(
            model_params=0, model_bits=16, num_kv_heads=8, head_dim=128,
            num_layers=32, max_num_seqs=32, max_seq_len=512, dtype_bytes=2,
            hidden_dim=4096,
        )
        est2 = estimate_serving_memory(
            model_params=0, model_bits=16, num_kv_heads=8, head_dim=128,
            num_layers=32, max_num_seqs=64, max_seq_len=512, dtype_bytes=2,
            hidden_dim=4096,
        )
        assert abs(est2.activations_mb / est1.activations_mb - 2.0) < 1e-6

    def test_total_equals_sum_of_components(self):
        est = estimate_serving_memory(
            model_params=7_000_000_000, model_bits=4, num_kv_heads=8,
            head_dim=128, num_layers=32, max_num_seqs=64, max_seq_len=512,
            dtype_bytes=2, hidden_dim=4096,
        )
        computed_subtotal = est.model_weights_mb + est.kv_cache_mb + est.activations_mb
        expected_overhead = computed_subtotal * 0.15 + 400  # OVERHEAD_RATIO_INFERENCE + FIXED_OVERHEAD_MB
        expected_total = computed_subtotal + expected_overhead
        assert abs(est.total_mb - expected_total) < 0.01

    def test_overhead_always_positive(self):
        est = estimate_serving_memory(
            model_params=0, model_bits=16, num_kv_heads=1, head_dim=1,
            num_layers=1, max_num_seqs=1, max_seq_len=1, dtype_bytes=2,
        )
        assert est.overhead_mb > 0.0

    def test_fits_in_true_below_budget(self):
        est = estimate_serving_memory(
            model_params=0, model_bits=16, num_kv_heads=8, head_dim=128,
            num_layers=4, max_num_seqs=8, max_seq_len=64, dtype_bytes=2,
        )
        assert est.fits_in(est.total_mb + 1.0)

    def test_fits_in_false_above_budget(self):
        est = estimate_serving_memory(
            model_params=7_000_000_000, model_bits=16, num_kv_heads=8,
            head_dim=128, num_layers=32, max_num_seqs=256, max_seq_len=8192,
            dtype_bytes=2,
        )
        assert not est.fits_in(est.total_mb - 1.0)

    def test_str_contains_key_sections(self):
        est = estimate_serving_memory(
            model_params=7_000_000_000, model_bits=4, num_kv_heads=8,
            head_dim=128, num_layers=32, max_num_seqs=64, max_seq_len=512,
            dtype_bytes=2,
        )
        s = str(est)
        assert "KV cache" in s
        assert "Model weights" in s
        assert "TOTAL" in s
        assert "max_num_seqs" in s

    def test_str_includes_activations_when_hidden_dim_provided(self):
        est = estimate_serving_memory(
            model_params=0, model_bits=16, num_kv_heads=8, head_dim=128,
            num_layers=4, max_num_seqs=8, max_seq_len=64, dtype_bytes=2,
            hidden_dim=4096,
        )
        assert "Activations" in str(est)

    def test_str_omits_activations_when_hidden_dim_zero(self):
        est = estimate_serving_memory(
            model_params=0, model_bits=16, num_kv_heads=8, head_dim=128,
            num_layers=4, max_num_seqs=8, max_seq_len=64, dtype_bytes=2,
        )
        assert "Activations" not in str(est)


# ---------------------------------------------------------------------------
# Input validation
# ---------------------------------------------------------------------------

class TestInputValidation:
    def test_negative_model_params_raises(self):
        with pytest.raises(ValueError, match="model_params"):
            estimate_serving_memory(model_params=-1, model_bits=16,
                                    num_kv_heads=8, head_dim=128, num_layers=1,
                                    max_num_seqs=1, max_seq_len=1)

    def test_invalid_model_bits_raises(self):
        with pytest.raises(ValueError, match="model_bits"):
            estimate_serving_memory(model_params=0, model_bits=7,
                                    num_kv_heads=8, head_dim=128, num_layers=1,
                                    max_num_seqs=1, max_seq_len=1)

    def test_invalid_dtype_bytes_raises(self):
        with pytest.raises(ValueError, match="dtype_bytes"):
            estimate_serving_memory(model_params=0, model_bits=16,
                                    num_kv_heads=8, head_dim=128, num_layers=1,
                                    max_num_seqs=1, max_seq_len=1, dtype_bytes=3)

    def test_zero_max_num_seqs_raises(self):
        with pytest.raises(ValueError, match="max_num_seqs"):
            estimate_serving_memory(model_params=0, model_bits=16,
                                    num_kv_heads=8, head_dim=128, num_layers=1,
                                    max_num_seqs=0, max_seq_len=1)


# ---------------------------------------------------------------------------
# preflight_inference() — happy path
# ---------------------------------------------------------------------------

class TestPreflightInferenceFits:
    def test_returns_inference_safe_config_type(self):
        guard = MemoryGuard.auto()
        with patch("memory_guard.guard.get_available_memory_mb", return_value=100_000):
            safe = guard.preflight_inference(
                model_params=0, model_bits=16, num_kv_heads=8, head_dim=128,
                num_layers=4, max_num_seqs=4, max_seq_len=64, dtype_bytes=2,
            )
        assert isinstance(safe, InferenceSafeConfig)

    def test_original_num_seqs_returned_when_fits(self):
        guard = MemoryGuard.auto()
        with patch("memory_guard.guard.get_available_memory_mb", return_value=100_000):
            safe = guard.preflight_inference(
                model_params=0, model_bits=16, num_kv_heads=8, head_dim=128,
                num_layers=4, max_num_seqs=4, max_seq_len=64, dtype_bytes=2,
            )
        assert safe.max_num_seqs == 4
        assert safe.fits is True
        assert safe.changes == []

    def test_estimate_embedded_in_result(self):
        guard = MemoryGuard.auto()
        with patch("memory_guard.guard.get_available_memory_mb", return_value=100_000):
            safe = guard.preflight_inference(
                model_params=0, model_bits=16, num_kv_heads=8, head_dim=128,
                num_layers=4, max_num_seqs=4, max_seq_len=64, dtype_bytes=2,
            )
        assert isinstance(safe.estimate, InferenceServingEstimate)
        assert safe.estimate.total_mb > 0

    def test_gpu_memory_utilization_in_valid_range(self):
        guard = MemoryGuard.auto()
        # Use available_mb close to the estimate so the ratio is meaningful
        with patch("memory_guard.guard.get_available_memory_mb", return_value=800):
            safe = guard.preflight_inference(
                model_params=0, model_bits=16, num_kv_heads=8, head_dim=128,
                num_layers=4, max_num_seqs=4, max_seq_len=64, dtype_bytes=2,
            )
        assert 0.0 <= safe.gpu_memory_utilization <= 0.95


# ---------------------------------------------------------------------------
# preflight_inference() — downgrade path
# ---------------------------------------------------------------------------

class TestPreflightInferenceDowngrade:
    def _small_budget(self, num_seqs: int) -> float:
        """Budget that fits exactly num_seqs (KV-only, tiny model)."""
        est = estimate_serving_memory(
            model_params=0, model_bits=16, num_kv_heads=2, head_dim=8,
            num_layers=2, max_num_seqs=num_seqs, max_seq_len=16, dtype_bytes=2,
        )
        return est.total_mb + 0.01  # Just above the boundary

    def test_reduces_num_seqs_when_budget_exceeded(self):
        budget = self._small_budget(50)
        guard = MemoryGuard.auto()
        with patch("memory_guard.guard.get_available_memory_mb", return_value=budget / guard.safety_ratio):
            safe = guard.preflight_inference(
                model_params=0, model_bits=16, num_kv_heads=2, head_dim=8,
                num_layers=2, max_num_seqs=200, max_seq_len=16, dtype_bytes=2,
            )
        assert safe.max_num_seqs < 200
        assert safe.fits is True

    def test_binary_search_finds_largest_fitting_value(self):
        # Use large per-seq cost (~64 MB each) so 50 vs 51 differ by 64 MB —
        # well above any overhead granularity.
        _kw = dict(model_params=0, model_bits=16, num_kv_heads=8, head_dim=128,
                   num_layers=8, max_seq_len=2048, dtype_bytes=2)
        est_50 = estimate_serving_memory(max_num_seqs=50, **_kw)
        est_51 = estimate_serving_memory(max_num_seqs=51, **_kw)
        # Budget sits exactly between the two estimates
        budget = (est_50.total_mb + est_51.total_mb) / 2
        assert est_50.fits_in(budget), "Test setup: 50 seqs must fit"
        assert not est_51.fits_in(budget), "Test setup: 51 seqs must not fit"

        guard = MemoryGuard.auto()
        with patch("memory_guard.guard.get_available_memory_mb",
                   return_value=budget / guard.safety_ratio):
            safe = guard.preflight_inference(max_num_seqs=200, **_kw)
        assert safe.max_num_seqs == 50

    def test_changes_populated_when_downgraded(self):
        budget = self._small_budget(10)
        guard = MemoryGuard.auto()
        with patch("memory_guard.guard.get_available_memory_mb", return_value=budget / guard.safety_ratio):
            safe = guard.preflight_inference(
                model_params=0, model_bits=16, num_kv_heads=2, head_dim=8,
                num_layers=2, max_num_seqs=100, max_seq_len=16, dtype_bytes=2,
            )
        assert len(safe.changes) == 1
        assert "max_num_seqs" in safe.changes[0]

    def test_min_num_seqs_floor_respected(self):
        # Budget so tiny that nothing can fit — fits should be False
        guard = MemoryGuard.auto()
        with patch("memory_guard.guard.get_available_memory_mb", return_value=1.0):
            safe = guard.preflight_inference(
                model_params=7_000_000_000, model_bits=16, num_kv_heads=32,
                head_dim=128, num_layers=32, max_num_seqs=256, max_seq_len=8192,
                dtype_bytes=2, min_num_seqs=8,
            )
        assert safe.max_num_seqs == 8
        assert safe.fits is False

    def test_fits_false_when_nothing_fits(self):
        guard = MemoryGuard.auto()
        with patch("memory_guard.guard.get_available_memory_mb", return_value=1.0):
            safe = guard.preflight_inference(
                model_params=7_000_000_000, model_bits=16, num_kv_heads=32,
                head_dim=128, num_layers=32, max_num_seqs=256, max_seq_len=8192,
                dtype_bytes=2,
            )
        assert safe.fits is False


# ---------------------------------------------------------------------------
# InferenceSafeConfig __str__
# ---------------------------------------------------------------------------

class TestInferenceSafeConfigStr:
    def test_str_contains_status_and_key_fields(self):
        guard = MemoryGuard.auto()
        with patch("memory_guard.guard.get_available_memory_mb", return_value=100_000):
            safe = guard.preflight_inference(
                model_params=0, model_bits=16, num_kv_heads=8, head_dim=128,
                num_layers=4, max_num_seqs=4, max_seq_len=64, dtype_bytes=2,
            )
        s = str(safe)
        assert "FITS" in s
        assert "max_num_seqs" in s
        assert "gpu_memory_utilization" in s
        assert "estimated memory" in s

    def test_str_shows_does_not_fit_when_not(self):
        guard = MemoryGuard.auto()
        with patch("memory_guard.guard.get_available_memory_mb", return_value=1.0):
            safe = guard.preflight_inference(
                model_params=7_000_000_000, model_bits=16, num_kv_heads=32,
                head_dim=128, num_layers=32, max_num_seqs=256, max_seq_len=8192,
                dtype_bytes=2,
            )
        assert "DOES NOT FIT" in str(safe)
