"""Backend plugin registry.

ml-memguard is a pure open-source library with zero external dependencies.
Optional backend plugins can be installed separately to provide additional
capabilities (policy sync, OOM prediction, telemetry).

Plugin registration (in the plugin package's pyproject.toml)::

    [project.entry-points."memory_guard.backends"]
    my_plugin = "your_package:YourBackend"

The backend class must satisfy the :class:`FleetBackend` protocol.  If no
backend is installed all functions in this module silently return ``None`` /
``False`` — the library degrades gracefully to local-only mode.
"""
from __future__ import annotations

import logging
from typing import Any, Dict, Optional, Protocol, runtime_checkable

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Backend protocol — implemented by optional fleet packages
# ---------------------------------------------------------------------------

@runtime_checkable
class FleetBackend(Protocol):
    """Interface that backend plugins must implement.

    Every method must be safe to call from a background thread, must never
    raise (swallow all exceptions internally), and must return ``None`` /
    ``False`` on any failure so callers can treat the backend as best-effort.
    """

    def upload_policy(self, policy_data: Dict[str, Any]) -> bool:
        """Upload the local Q-table to the backend store.  Returns True on success."""
        ...

    def download_policy(self) -> Optional[Dict[str, Any]]:
        """Download the aggregated fleet Q-table.  Returns None on miss or failure."""
        ...

    def record_training_result(self, run_data: Dict[str, Any]) -> bool:
        """Post a single training-run record.  Returns True on success."""
        ...

    def upload_inference_signals(self, signals: Any) -> bool:
        """Post a single inference monitoring cycle.  Returns True on success."""
        ...

    def predict_oom(
        self,
        signals: Dict[str, Any],
        model_name: str = "",
        backend: str = "",
    ) -> Optional[Dict[str, Any]]:
        """Return OOM prediction dict or None.

        The dict must contain at least:
          ``oom_probability`` (float 0–1), ``action`` (str),
          ``horizon_seconds`` (int), ``confidence`` (float).
        """
        ...

    def get_fleet_summary(self) -> Optional[Dict[str, Any]]:
        """Return aggregated fleet statistics or None."""
        ...


# ---------------------------------------------------------------------------
# Registry — auto-discovered once per process via entry_points
# ---------------------------------------------------------------------------

_backend: Optional[FleetBackend] = None
_discovered: bool = False


def _discover() -> None:
    """Load the first registered backend plugin (called lazily on first use)."""
    global _backend, _discovered
    if _discovered:
        return
    _discovered = True
    try:
        from importlib.metadata import entry_points
        eps = entry_points(group="memory_guard.backends")
        for ep in eps:
            try:
                cls = ep.load()
                instance = cls()
                if isinstance(instance, FleetBackend):
                    _backend = instance
                    logger.debug(
                        "[memory-guard] Backend plugin loaded: %s", ep.name
                    )
                    break
            except Exception as exc:
                logger.debug(
                    "[memory-guard] Failed to load backend %r: %s", ep.name, exc
                )
    except Exception:
        pass


def get_backend() -> Optional[FleetBackend]:
    """Return the active backend plugin, or ``None`` if none is installed."""
    _discover()
    return _backend


# ---------------------------------------------------------------------------
# Convenience wrappers (no-op when no backend installed)
# ---------------------------------------------------------------------------

def predict_oom(
    signals: Dict[str, Any],
    model_name: str = "",
    backend_str: str = "",
) -> Optional[Dict[str, Any]]:
    """Call the fleet OOM predictor.  Returns None when no backend is installed."""
    b = get_backend()
    if b is None:
        return None
    try:
        return b.predict_oom(signals, model_name=model_name, backend=backend_str)
    except Exception as exc:
        logger.debug("[memory-guard] predict_oom raised: %s", exc)
        return None


def upload_policy(policy_data: Dict[str, Any]) -> bool:
    """Upload Q-table to backend store.  Returns False when no backend is installed."""
    b = get_backend()
    if b is None:
        return False
    try:
        return b.upload_policy(policy_data)
    except Exception:
        return False


def download_policy() -> Optional[Dict[str, Any]]:
    """Download fleet Q-table.  Returns None when no backend is installed."""
    b = get_backend()
    if b is None:
        return None
    try:
        return b.download_policy()
    except Exception:
        return None


def record_training_result(run_data: Dict[str, Any]) -> bool:
    """Post training run record.  Returns False when no backend is installed."""
    b = get_backend()
    if b is None:
        return False
    try:
        return b.record_training_result(run_data)
    except Exception:
        return False


def upload_inference_signals(signals: Any) -> bool:
    """Post inference signals.  Returns False when no backend is installed."""
    b = get_backend()
    if b is None:
        return False
    try:
        return b.upload_inference_signals(signals)
    except Exception:
        return False


def get_fleet_summary() -> Optional[Dict[str, Any]]:
    """Return fleet summary dict.  Returns None when no backend is installed."""
    b = get_backend()
    if b is None:
        return None
    try:
        return b.get_fleet_summary()
    except Exception:
        return None
