"""Optional integration registry.

ml-memguard is a pure open-source library with zero required external
services. Optional integrations can be installed separately to provide
additional capabilities such as policy exchange, predictive scoring, or
telemetry export.

Integration registration (in the integration package's ``pyproject.toml``)::

    [project.entry-points."memory_guard.integrations"]
    my_integration = "your_package:YourIntegration"

For backward compatibility, legacy packages that still register under
``memory_guard.backends`` are also discovered for now.

The integration class must satisfy the :class:`FleetIntegration` protocol. If
no integration is installed, all functions in this module silently return
``None`` / ``False`` and the library stays on its local-only path.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, Optional, Protocol, runtime_checkable

logger = logging.getLogger(__name__)

_ENTRY_POINT_GROUPS = (
    "memory_guard.integrations",
    "memory_guard.backends",  # legacy group kept for compatibility
)


@runtime_checkable
class FleetIntegration(Protocol):
    """Interface that optional integrations must implement."""

    def upload_policy(self, policy_data: Dict[str, Any]) -> bool:
        """Upload the local Q-table to the integration store. Returns True on success."""
        ...

    def download_policy(self) -> Optional[Dict[str, Any]]:
        """Download the aggregated Q-table. Returns None on miss or failure."""
        ...

    def record_training_result(self, run_data: Dict[str, Any]) -> bool:
        """Post a single training-run record. Returns True on success."""
        ...

    def upload_inference_signals(self, signals: Any) -> bool:
        """Post a single inference monitoring cycle. Returns True on success."""
        ...

    def upload_source_baseline(self, baseline: Dict[str, Any]) -> bool:
        """Upload the startup memory snapshot to the active integration."""
        ...

    def predict_oom(
        self,
        signals: Dict[str, Any],
        model_name: str = "",
        backend: str = "",
    ) -> Optional[Dict[str, Any]]:
        """Return an OOM prediction dict or None."""
        ...

    def get_fleet_summary(self) -> Optional[Dict[str, Any]]:
        """Return aggregated summary statistics or None."""
        ...


# Legacy alias for older imports and third-party packages.
FleetBackend = FleetIntegration


_integration: Optional[FleetIntegration] = None
_discovered: bool = False


def _discover() -> None:
    """Load the first registered optional integration (lazily on first use)."""
    global _integration, _discovered
    if _discovered:
        return
    _discovered = True
    try:
        from importlib.metadata import entry_points

        for group in _ENTRY_POINT_GROUPS:
            try:
                eps = entry_points(group=group)
            except TypeError:
                eps = entry_points().get(group, ())

            for ep in eps:
                try:
                    cls = ep.load()
                    instance = cls()
                    if isinstance(instance, FleetIntegration):
                        _integration = instance
                        logger.debug(
                            "[memory-guard] Optional integration loaded: %s", ep.name
                        )
                        return
                except Exception as exc:
                    logger.debug(
                        "[memory-guard] Failed to load integration %r: %s", ep.name, exc
                    )
    except Exception:
        pass


def get_integration() -> Optional[FleetIntegration]:
    """Return the active optional integration, or ``None`` if none is installed."""
    _discover()
    return _integration


def get_backend() -> Optional[FleetIntegration]:
    """Compatibility alias for older callers."""
    return get_integration()


def predict_oom(
    signals: Dict[str, Any],
    model_name: str = "",
    backend_str: str = "",
) -> Optional[Dict[str, Any]]:
    """Call the active OOM predictor. Returns None when no integration is installed."""
    integ = get_integration()
    if integ is None:
        return None
    try:
        return integ.predict_oom(signals, model_name=model_name, backend=backend_str)
    except Exception as exc:
        logger.debug("[memory-guard] predict_oom raised: %s", exc)
        return None


def upload_policy(policy_data: Dict[str, Any]) -> bool:
    """Upload a Q-table snapshot. Returns False when no integration is installed."""
    integ = get_integration()
    if integ is None:
        return False
    try:
        return integ.upload_policy(policy_data)
    except Exception:
        return False


def download_policy() -> Optional[Dict[str, Any]]:
    """Download an aggregated Q-table. Returns None when no integration is installed."""
    integ = get_integration()
    if integ is None:
        return None
    try:
        return integ.download_policy()
    except Exception:
        return None


def record_training_result(run_data: Dict[str, Any]) -> bool:
    """Post a training-run record. Returns False when no integration is installed."""
    integ = get_integration()
    if integ is None:
        return False
    try:
        return integ.record_training_result(run_data)
    except Exception:
        return False


def upload_inference_signals(signals: Any) -> bool:
    """Post inference signals. Returns False when no integration is installed."""
    integ = get_integration()
    if integ is None:
        return False
    try:
        return integ.upload_inference_signals(signals)
    except Exception:
        return False


def upload_source_baseline(baseline: Dict[str, Any]) -> bool:
    """Post the startup memory snapshot. Returns False when no integration is installed."""
    integ = get_integration()
    if integ is None:
        return False
    try:
        return integ.upload_source_baseline(baseline)
    except Exception:
        return False


def get_fleet_summary() -> Optional[Dict[str, Any]]:
    """Return aggregated summary stats. Returns None when no integration is installed."""
    integ = get_integration()
    if integ is None:
        return None
    try:
        return integ.get_fleet_summary()
    except Exception:
        return None


__all__ = [
    "FleetIntegration",
    "FleetBackend",
    "get_integration",
    "get_backend",
    "predict_oom",
    "upload_policy",
    "download_policy",
    "record_training_result",
    "upload_inference_signals",
    "upload_source_baseline",
    "get_fleet_summary",
]
