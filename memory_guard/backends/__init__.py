"""Compatibility shim for the legacy ``memory_guard.backends`` import path.

Prefer ``memory_guard.integrations`` for all new code and plugin registration.
"""

from __future__ import annotations

import sys

from .. import integrations as _integrations

sys.modules[__name__] = _integrations
