"""Auto-calibration — learns correction factors from real training runs.

Formula-based estimators are fundamentally limited (arxiv 2602.17817):
they can't capture framework-specific behaviors like operator fusion,
temporary buffers, and memory fragmentation. Auto-calibration bridges
this gap by collecting real mx.metal.get_peak_memory() readings during
training and using them to correct future estimates.

The calibration model is a simple linear correction:
    corrected_estimate = formula_estimate × correction_factor

The correction_factor is learned from past (estimate, actual) pairs
stored in ~/.memory-guard/calibration.json.
"""

from __future__ import annotations

import json
import logging
import statistics
import threading
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)

DEFAULT_CALIBRATION_DIR = Path.home() / ".memory-guard"
DEFAULT_CALIBRATION_FILE = DEFAULT_CALIBRATION_DIR / "calibration.json"


@dataclass
class CalibrationPoint:
    """A single (estimated, actual) measurement pair."""
    estimated_mb: float
    actual_peak_mb: float
    model_name: str = ""
    backend: str = ""
    batch_size: int = 0
    seq_length: int = 0
    lora_rank: int = 0
    flash_attention: bool = True
    timestamp: float = 0.0

    @property
    def correction_factor(self) -> float:
        """actual / estimated — multiply future estimates by this."""
        if self.estimated_mb > 0:
            return self.actual_peak_mb / self.estimated_mb
        return 1.0


class CalibrationStore:
    """Persists calibration data across sessions.

    Stores up to max_points recent (estimate, actual) pairs per
    backend and uses their median correction factor to adjust
    future estimates.
    """

    def __init__(
        self,
        path: Path = DEFAULT_CALIBRATION_FILE,
        max_points: int = 50,  # See constants.CALIBRATION_MAX_POINTS
    ):
        self.path = path
        self.max_points = max_points
        self._points: list[dict] = []
        self._lock = threading.Lock()
        self._load()

    def _load(self):
        """Load and validate calibration data from disk."""
        try:
            if self.path.exists():
                with open(self.path) as f:
                    data = json.load(f)
                points = data.get("points", [])
                # Validate structure: must be list of dicts with required keys
                if not isinstance(points, list):
                    logger.warning("Calibration file corrupted: 'points' is not a list, resetting")
                    self._points = []
                    return
                validated = []
                for p in points:
                    if (isinstance(p, dict)
                            and "correction_factor" in p
                            and isinstance(p["correction_factor"], (int, float))
                            and 0.8 <= p["correction_factor"] <= 1.3):
                        validated.append(p)
                self._points = validated[-self.max_points:]
        except (json.JSONDecodeError, KeyError, TypeError) as e:
            logger.warning(f"Calibration file corrupted: {e}, resetting")
            self._points = []
        except Exception as e:
            logger.debug(f"Could not load calibration data: {e}")
            self._points = []

    def _save(self):
        """Persist calibration data atomically with restricted permissions."""
        try:
            import os
            import tempfile
            parent = self.path.parent
            parent.mkdir(parents=True, exist_ok=True, mode=0o700)

            # Guard against symlink attacks on shared machines.
            # Check both the directory AND the target file.
            if os.path.islink(str(parent)):
                logger.warning("Calibration dir is a symlink, refusing to write")
                return
            if hasattr(os, 'getuid'):
                parent_stat = os.lstat(str(parent))
                if parent_stat.st_uid != os.getuid():
                    logger.warning("Calibration dir not owned by current user, refusing to write")
                    return
            # Remove target if it's a symlink (prevents symlink-following
            # on os.replace). Check immediately before replace to minimize
            # TOCTOU window.
            target = str(self.path)

            # Write to temp file first
            fd, tmp_path = tempfile.mkstemp(
                dir=str(parent), suffix=".tmp", prefix=".cal_"
            )
            try:
                with os.fdopen(fd, "w") as f:
                    json.dump({"points": self._points}, f, indent=2)
                os.chmod(tmp_path, 0o600)

                # Final symlink check right before replace (minimal TOCTOU)
                if os.path.lexists(target) and os.path.islink(target):
                    logger.warning("Calibration file is a symlink, refusing to write")
                    os.unlink(tmp_path)
                    return

                os.replace(tmp_path, target)  # Atomic on POSIX

                # Post-write verification: if someone raced us and target
                # is now a symlink, remove it (defense in depth)
                if os.path.islink(target):
                    logger.warning("Calibration file became a symlink after write, removing")
                    os.unlink(target)
            except BaseException:
                if os.path.exists(tmp_path):
                    os.unlink(tmp_path)
                raise
        except Exception as e:
            logger.warning(f"Could not save calibration data: {e}")

    def add_point(self, point: CalibrationPoint):
        """Record a new calibration measurement. Thread-safe."""
        with self._lock:
            self._add_point_locked(point)

    def _add_point_locked(self, point: CalibrationPoint):
        self._points.append({
            "estimated_mb": point.estimated_mb,
            "actual_peak_mb": point.actual_peak_mb,
            "model_name": point.model_name,
            "backend": point.backend,
            "batch_size": point.batch_size,
            "seq_length": point.seq_length,
            "lora_rank": point.lora_rank,
            "flash_attention": point.flash_attention,
            "timestamp": point.timestamp,
            "correction_factor": point.correction_factor,
        })
        # Trim oldest
        if len(self._points) > self.max_points:
            self._points = self._points[-self.max_points:]
        self._save()
        logger.info(
            f"Calibration point: estimated={point.estimated_mb:.0f}MB, "
            f"actual={point.actual_peak_mb:.0f}MB, "
            f"factor={point.correction_factor:.3f}"
        )

    def get_correction_factor(self, backend: str = "") -> float:
        """Get median correction factor for a backend. Thread-safe."""
        with self._lock:
            return self._get_correction_factor_locked(backend)

    def _get_correction_factor_locked(self, backend: str = "") -> float:
        if not self._points:
            return 1.0

        relevant = self._points
        if backend:
            filtered = [p for p in self._points if p.get("backend") == backend]
            if len(filtered) >= 3:
                relevant = filtered

        # Bounds: 0.8-1.3. Limits adversarial impact to 20% under-estimation
        # (0.8x) or 30% over-estimation (1.3x). Previous 0.7x allowed 30%
        # under-estimation which on a 36GB machine = ~11GB phantom headroom.
        # Threat model: any process with write access to ~/.memory-guard/
        # can inject calibration data. Bounds + file permissions are the
        # defense. Bounds + file permissions (0o600) are the mitigation.
        factors = [p["correction_factor"] for p in relevant
                   if "correction_factor" in p and 0.8 <= p["correction_factor"] <= 1.3]

        if len(factors) < 3:
            return 1.0  # Not enough data

        return statistics.median(factors)

    @property
    def num_points(self) -> int:
        return len(self._points)

    def clear(self):
        """Clear all calibration data."""
        self._points = []
        self._save()


def apply_calibration(
    estimated_mb: float,
    backend: str = "",
    store: Optional[CalibrationStore] = None,
) -> tuple[float, float]:
    """Apply calibration correction to a formula-based estimate.

    Returns:
        (corrected_estimate_mb, correction_factor)

    If no calibration data is available, returns the original estimate
    with factor=1.0.
    """
    if store is None:
        store = CalibrationStore()

    factor = store.get_correction_factor(backend)
    corrected = estimated_mb * factor

    return corrected, factor


def record_training_result(
    estimated_mb: float,
    actual_peak_mb: float,
    model_name: str = "",
    backend: str = "",
    batch_size: int = 0,
    seq_length: int = 0,
    lora_rank: int = 0,
    flash_attention: bool = True,
    store: Optional[CalibrationStore] = None,
):
    """Record the result of a training run for future calibration.

    Call this after training completes with the formula estimate
    and the actual peak memory (from mx.metal.get_peak_memory()
    or torch.cuda.max_memory_allocated()).
    """
    import time

    if store is None:
        store = CalibrationStore()

    point = CalibrationPoint(
        estimated_mb=estimated_mb,
        actual_peak_mb=actual_peak_mb,
        model_name=model_name,
        backend=backend,
        batch_size=batch_size,
        seq_length=seq_length,
        lora_rank=lora_rank,
        flash_attention=flash_attention,
        timestamp=time.time(),
    )
    store.add_point(point)
