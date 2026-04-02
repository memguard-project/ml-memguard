"""memory-guard — Cross-platform memory guard for ML training.

Prevents OOM crashes on Apple Silicon, CUDA GPUs, and CPU-only systems
by proactive memory estimation before training and runtime pressure
monitoring during training.

Works with any training framework: mlx_lm, Unsloth, HuggingFace, PyTorch.

Quick start:
    from memory_guard import MemoryGuard

    guard = MemoryGuard.auto()
    safe = guard.preflight(
        model_params=9e9, model_bits=4,
        hidden_dim=4096, num_heads=32, num_layers=32,
        batch_size=4, seq_length=2048, lora_rank=32, lora_layers=16,
    )
    print(safe)  # Auto-downgraded if needed

    with guard.monitor(batch_size=safe.batch_size) as mon:
        for step in range(1000):
            train_step(batch_size=mon.current_batch_size)
"""

__version__ = "0.1.0"

from .estimator import (
    MemoryEstimate,
    ModelSpec,
    TrainSpec,
    FinetuneMethod,
    ModelArch,
    estimate_training_memory,
    estimate_inference_memory,
)
from .platforms import (
    detect_platform,
    get_available_memory_mb,
    get_memory_pressure,
    get_mlx_active_memory_mb,
    get_mlx_peak_memory_mb,
    reset_mlx_peak_memory,
    PlatformInfo,
    Backend,
)
from .calibration import CalibrationStore, record_training_result
from .guard import MemoryGuard, SafeConfig
from .monitor import RuntimeMonitor
from .cuda_recovery import CUDAOOMRecovery
from .downgrade import auto_downgrade, DowngradeResult

__all__ = [
    "MemoryGuard",
    "SafeConfig",
    "MemoryEstimate",
    "ModelSpec",
    "TrainSpec",
    "FinetuneMethod",
    "ModelArch",
    "estimate_training_memory",
    "estimate_inference_memory",
    "detect_platform",
    "get_available_memory_mb",
    "get_memory_pressure",
    "get_mlx_active_memory_mb",
    "get_mlx_peak_memory_mb",
    "reset_mlx_peak_memory",
    "PlatformInfo",
    "Backend",
    "RuntimeMonitor",
    "CUDAOOMRecovery",
    "CalibrationStore",
    "record_training_result",
    "auto_downgrade",
    "DowngradeResult",
]
