"""MemoryGuard — the main entry point.

Combines proactive estimation, auto-downgrade, and runtime monitoring
into a single, easy-to-use API.

    guard = MemoryGuard.auto()
    safe = guard.preflight(model_params=9e9, ...)
    with guard.monitor(safe.batch_size) as mon:
        for step in range(1000):
            train_step(batch_size=mon.current_batch_size)
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Optional

from .downgrade import DowngradeResult, auto_downgrade
from .estimator import MemoryEstimate, estimate_training_memory
from .monitor import RuntimeMonitor
from .platforms import Backend, PlatformInfo, detect_platform, get_available_memory_mb

logger = logging.getLogger(__name__)


@dataclass
class SafeConfig:
    """Memory-safe training configuration returned by preflight()."""
    batch_size: int
    seq_length: int
    lora_rank: int
    lora_layers: int
    grad_checkpoint: bool
    grad_accumulation: int
    estimate: MemoryEstimate
    budget_mb: float
    available_mb: float
    changes: list[str]
    fits: bool

    def __str__(self):
        status = "FITS" if self.fits else "DOES NOT FIT"
        lines = [
            f"SafeConfig ({status}):",
            f"  batch_size:       {self.batch_size}",
            f"  seq_length:       {self.seq_length}",
            f"  lora_rank:        {self.lora_rank}",
            f"  lora_layers:      {self.lora_layers}",
            f"  grad_checkpoint:  {self.grad_checkpoint}",
            f"  grad_accumulation:{self.grad_accumulation}",
            f"  estimated memory: {self.estimate.total_mb:.0f} MB",
            f"  budget:           {self.budget_mb:.0f} MB",
            f"  available:        {self.available_mb:.0f} MB",
        ]
        if self.changes:
            lines.append(f"  changes applied:  {len(self.changes)}")
            for c in self.changes:
                lines.append(f"    - {c}")
        return "\n".join(lines)


class MemoryGuard:
    """Cross-platform memory guard for ML training.

    Combines three layers of protection:
    1. Proactive estimation (preflight) — before training starts
    2. Auto-downgrade — iteratively reduces config to fit budget
    3. Runtime monitoring — background thread polls memory pressure

    Usage:
        # Auto-detect platform
        guard = MemoryGuard.auto()

        # Or specify
        guard = MemoryGuard(safety_ratio=0.75)

        # Pre-flight check
        safe = guard.preflight(
            model_params=9_000_000_000, model_bits=4,
            hidden_dim=4096, num_heads=32, num_layers=32,
            batch_size=4, seq_length=2048,
            lora_rank=32, lora_layers=16,
        )
        print(safe)  # Shows adjusted config

        # Runtime monitoring
        with guard.monitor(safe.batch_size) as mon:
            for step in range(num_steps):
                actual_bs = mon.current_batch_size  # May decrease mid-training
                train_step(batch_size=actual_bs)
    """

    def __init__(
        self,
        platform_info: Optional[PlatformInfo] = None,
        safety_ratio: float = 0.80,  # See constants.SAFETY_RATIO_DEFAULT
        enable_calibration: bool = True,
    ):
        """
        Args:
            platform_info: Detected platform, or None for auto-detect.
            safety_ratio: Use at most this fraction of available memory.
                         0.80 = leave 20% headroom (recommended).
                         0.90 = aggressive, higher risk of pressure.
                         0.70 = conservative, for shared machines.
            enable_calibration: If True, apply learned correction factors
                              from past training runs to improve accuracy.
        """
        self.platform = platform_info or detect_platform()
        self.safety_ratio = safety_ratio
        self.enable_calibration = enable_calibration

        self._calibration_store = None
        if enable_calibration:
            from .calibration import CalibrationStore
            self._calibration_store = CalibrationStore()

        self._last_estimate_mb: Optional[float] = None  # For post-training recording

    @classmethod
    def auto(cls, safety_ratio: float = 0.80, enable_calibration: bool = True) -> "MemoryGuard":
        """Create a MemoryGuard with auto-detected platform."""
        return cls(safety_ratio=safety_ratio, enable_calibration=enable_calibration)

    @property
    def available_mb(self) -> float:
        """Currently available memory in MB."""
        return get_available_memory_mb(self.platform.backend)

    @property
    def budget_mb(self) -> float:
        """Memory budget (available × safety_ratio + partial swap credit).

        Swap/compressor headroom is included at 50% credit because swap
        is much slower than RAM and causes performance degradation.
        Using it avoids crashes but at a throughput cost.
        """
        ram_budget = self.available_mb * self.safety_ratio
        from .constants import SWAP_CREDIT_RATIO
        swap_credit = self.platform.swap_available_mb * SWAP_CREDIT_RATIO
        return ram_budget + swap_credit

    def estimate(self, **kwargs) -> MemoryEstimate:
        """Estimate training memory without auto-downgrade.

        Pass same kwargs as estimate_training_memory().
        """
        return estimate_training_memory(**kwargs)

    def preflight(
        self,
        model_params: int,
        model_bits: int = 4,
        hidden_dim: int = 4096,
        num_heads: int = 32,
        num_layers: int = 32,
        batch_size: int = 4,
        seq_length: int = 2048,
        lora_rank: int = 8,
        lora_layers: int = 16,
        optimizer: str = "adam",
        grad_checkpoint: bool = False,
        grad_accumulation: int = 1,
        flash_attention: Optional[bool] = None,
        lazy_evaluation: Optional[bool] = None,
    ) -> SafeConfig:
        """Run pre-flight memory check and auto-downgrade if needed.

        Returns a SafeConfig with memory-safe parameters.
        If the original config fits, it's returned unchanged.
        If not, parameters are iteratively reduced.

        flash_attention and lazy_evaluation are auto-detected from
        platform if not explicitly set.
        """
        # Auto-detect framework features from platform
        if flash_attention is None:
            flash_attention = True  # Default on for all modern frameworks
        if lazy_evaluation is None:
            # MLX uses lazy evaluation on Apple Silicon only.
            # Intel Macs don't support MLX training (no Metal ML compute).
            lazy_evaluation = self.platform.backend == Backend.APPLE_SILICON

        available = self.available_mb
        budget = self.budget_mb  # Includes swap credit

        from .estimator import ModelSpec, TrainSpec

        model_spec = ModelSpec(
            params=model_params, hidden_dim=hidden_dim,
            num_heads=num_heads, num_layers=num_layers, bits=model_bits,
        )
        train_spec = TrainSpec(
            batch_size=batch_size, seq_length=seq_length,
            lora_rank=lora_rank, lora_layers=lora_layers,
            optimizer=optimizer, grad_checkpoint=grad_checkpoint,
            grad_accumulation=grad_accumulation,
            flash_attention=flash_attention,
            lazy_evaluation=lazy_evaluation,
        )

        # Initial estimate (formula-based)
        est = estimate_training_memory(model=model_spec, train=train_spec)

        # Apply auto-calibration correction if available
        # Apply calibration as a separate comparison value — don't mutate
        # est.total_mb, or __str__() will show component sum != total.
        effective_mb = est.total_mb
        if self.enable_calibration and self._calibration_store:
            from .calibration import apply_calibration
            corrected_mb, factor = apply_calibration(
                est.total_mb, backend=self.platform.backend.value,
                store=self._calibration_store,
            )
            if factor != 1.0:
                logger.info(
                    f"Calibration correction: {est.total_mb:.0f}MB × "
                    f"{factor:.3f} = {corrected_mb:.0f}MB "
                    f"(based on {self._calibration_store.num_points} past runs)"
                )
                effective_mb = corrected_mb

        self._last_estimate_mb = effective_mb

        if effective_mb <= budget:
            # Fits! Return original config.
            return SafeConfig(
                batch_size=batch_size, seq_length=seq_length,
                lora_rank=lora_rank, lora_layers=lora_layers,
                grad_checkpoint=grad_checkpoint,
                grad_accumulation=grad_accumulation,
                estimate=est, budget_mb=budget, available_mb=available,
                changes=[], fits=True,
            )

        # Doesn't fit — auto-downgrade
        logger.warning(
            f"Estimated {effective_mb:.0f}MB exceeds budget {budget:.0f}MB "
            f"({available:.0f}MB × {self.safety_ratio:.0%}). Auto-downgrading..."
        )

        result = auto_downgrade(
            budget_mb=budget,
            model_params=model_params, model_bits=model_bits,
            hidden_dim=hidden_dim, num_heads=num_heads, num_layers=num_layers,
            batch_size=batch_size, seq_length=seq_length,
            lora_rank=lora_rank, lora_layers=lora_layers,
            grad_checkpoint=grad_checkpoint,
            grad_accumulation=grad_accumulation,
            optimizer=optimizer,
            flash_attention=flash_attention,
            lazy_evaluation=lazy_evaluation,
        )

        return SafeConfig(
            batch_size=result.batch_size, seq_length=result.seq_length,
            lora_rank=result.lora_rank, lora_layers=result.lora_layers,
            grad_checkpoint=result.grad_checkpoint,
            grad_accumulation=result.grad_accumulation,
            estimate=result.final_estimate, budget_mb=budget,
            available_mb=available, changes=result.changes, fits=result.fits,
        )

    def monitor(
        self,
        batch_size: int,
        poll_interval: float = 5.0,
        max_downgrades: int = 3,
        **kwargs,
    ) -> RuntimeMonitor:
        """Create a RuntimeMonitor (use as context manager).

        Usage:
            with guard.monitor(batch_size=4) as mon:
                for step in training_loop:
                    train_step(batch_size=mon.current_batch_size)
        """
        mon = RuntimeMonitor(
            poll_interval=poll_interval,
            backend=self.platform.backend,
            max_downgrades=max_downgrades,
            memory_limit_mb=self.budget_mb,
            **kwargs,
        )
        return mon.session(batch_size)

    def record_result(
        self,
        actual_peak_mb: Optional[float] = None,
        model_name: str = "",
        **kwargs,
    ):
        """Record actual peak memory after training for auto-calibration.

        Call this after training completes. If actual_peak_mb is None,
        attempts to read from mx.metal.get_peak_memory() or
        torch.cuda.max_memory_allocated().

        Over time, this builds a calibration dataset that corrects
        the formula's output to match real-world measurements.
        """
        if not self.enable_calibration or not self._calibration_store:
            return

        # Auto-detect actual peak if not provided
        if actual_peak_mb is None:
            from .platforms import get_mlx_peak_memory_mb
            actual_peak_mb = get_mlx_peak_memory_mb()

        if actual_peak_mb is None:
            try:
                import torch
                if torch.cuda.is_available():
                    actual_peak_mb = torch.cuda.max_memory_allocated() / (1024 * 1024)
            except ImportError:
                pass

        if actual_peak_mb is None or actual_peak_mb <= 0:
            logger.debug("Could not determine actual peak memory for calibration")
            return

        if self._last_estimate_mb is None or self._last_estimate_mb <= 0:
            logger.debug("No estimate available to calibrate against")
            return

        from .calibration import record_training_result
        record_training_result(
            estimated_mb=self._last_estimate_mb,
            actual_peak_mb=actual_peak_mb,
            model_name=model_name,
            backend=self.platform.backend.value,
            store=self._calibration_store,
            **kwargs,
        )
