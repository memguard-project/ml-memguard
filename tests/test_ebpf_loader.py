"""Tests for PR 53: BPFProbeLoader, MemguardBPFEvent, MemguardBPFSession.

Covers:
  BPFProbeLoader
  --------------
  1.  available is False on non-Linux (macOS / Windows)
  2.  available is False when kernel version < 5.8
  3.  available is False when CAP_BPF is missing (non-root, no caps)
  4.  available is False when cgroup v2 is not mounted
  5.  available is False when no BPF library is installed
  6.  backend is 'bcc' when bcc is importable and all checks pass
  7.  backend is 'libbpf' when only libbpf-python is importable
  8.  check_capabilities() returns a (bool, str) tuple
  9.  repr before first probe shows 'not-yet-probed'

  MemguardBPFEvent
  ----------------
  10. to_dict() includes all required fields
  11. to_dict() merges extra dict into output
  12. EVENT_* constants have expected string values

  MemguardBPFSession
  ------------------
  13. __enter__ is a no-op when loader.available is False
  14. available is False before __enter__ is called
  15. __exit__ is safe when the session was never started (manager is None)
"""

from __future__ import annotations

import sys
from typing import Optional
from unittest.mock import MagicMock, patch

import pytest

# ---------------------------------------------------------------------------
# Imports under test
# ---------------------------------------------------------------------------

from memory_guard.ebpf._loader import BPFProbeLoader, _detect_backend
from memory_guard.ebpf._event import (
    MemguardBPFEvent,
    EVENT_MEMORY_HIGH,
    EVENT_OOM_KILL,
    EVENT_PREEMPTION,
    EVENT_PAGE_FAULT,
    EVENT_MMAP_GROWTH,
)
from memory_guard.ebpf._session import MemguardBPFSession


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_loader_available(backend: str = "bcc") -> BPFProbeLoader:
    """Return a pre-detected loader that reports available=True."""
    loader = BPFProbeLoader()
    loader._available = True
    loader._backend   = backend
    loader._reason    = ""
    return loader


def _make_loader_unavailable(reason: str = "test-unavailable") -> BPFProbeLoader:
    """Return a pre-detected loader that reports available=False."""
    loader = BPFProbeLoader()
    loader._available = False
    loader._backend   = None
    loader._reason    = reason
    return loader


# ===========================================================================
# BPFProbeLoader
# ===========================================================================

class TestBPFProbeLoaderUnavailable:

    def test_unavailable_on_non_linux(self):
        """available is False when sys.platform is not 'linux'."""
        with patch("memory_guard.ebpf._loader.sys") as mock_sys:
            mock_sys.platform = "darwin"
            loader = BPFProbeLoader()
            assert loader.available is False
            assert "Linux" in loader.unavailable_reason or "linux" in loader.unavailable_reason.lower()

    def test_unavailable_kernel_too_old(self):
        """available is False when kernel version is < 5.8."""
        with patch("memory_guard.ebpf._loader.sys") as mock_sys, \
             patch("memory_guard.ebpf._loader._kernel_version", return_value=(4, 19)), \
             patch("memory_guard.ebpf._loader._has_cap_bpf", return_value=True), \
             patch("memory_guard.ebpf._loader._cgroupv2_mounted", return_value=True), \
             patch("memory_guard.ebpf._loader._detect_backend", return_value="bcc"):
            mock_sys.platform = "linux"
            loader = BPFProbeLoader()
            assert loader.available is False
            assert "5.8" in loader.unavailable_reason

    def test_unavailable_missing_capabilities(self):
        """available is False when the process lacks CAP_BPF."""
        with patch("memory_guard.ebpf._loader.sys") as mock_sys, \
             patch("memory_guard.ebpf._loader._kernel_version", return_value=(5, 15)), \
             patch("memory_guard.ebpf._loader._has_cap_bpf", return_value=False), \
             patch("memory_guard.ebpf._loader._cgroupv2_mounted", return_value=True), \
             patch("memory_guard.ebpf._loader._detect_backend", return_value="bcc"):
            mock_sys.platform = "linux"
            loader = BPFProbeLoader()
            assert loader.available is False
            assert "CAP_BPF" in loader.unavailable_reason

    def test_unavailable_no_cgroupv2(self):
        """available is False when cgroup v2 is not mounted."""
        with patch("memory_guard.ebpf._loader.sys") as mock_sys, \
             patch("memory_guard.ebpf._loader._kernel_version", return_value=(5, 15)), \
             patch("memory_guard.ebpf._loader._has_cap_bpf", return_value=True), \
             patch("memory_guard.ebpf._loader._cgroupv2_mounted", return_value=False), \
             patch("memory_guard.ebpf._loader._detect_backend", return_value="bcc"):
            mock_sys.platform = "linux"
            loader = BPFProbeLoader()
            assert loader.available is False
            assert "cgroup" in loader.unavailable_reason.lower()

    def test_unavailable_no_bpf_library(self):
        """available is False when neither bcc nor libbpf-python is installed."""
        with patch("memory_guard.ebpf._loader.sys") as mock_sys, \
             patch("memory_guard.ebpf._loader._kernel_version", return_value=(5, 15)), \
             patch("memory_guard.ebpf._loader._has_cap_bpf", return_value=True), \
             patch("memory_guard.ebpf._loader._cgroupv2_mounted", return_value=True), \
             patch("memory_guard.ebpf._loader._detect_backend", return_value=None):
            mock_sys.platform = "linux"
            loader = BPFProbeLoader()
            assert loader.available is False
            assert "ml-memguard[ebpf]" in loader.unavailable_reason


class TestBPFProbeLoaderAvailable:

    def _all_pass_patches(self):
        """Return a context manager that makes all capability checks pass."""
        return (
            patch("memory_guard.ebpf._loader.sys"),
            patch("memory_guard.ebpf._loader._kernel_version", return_value=(5, 15)),
            patch("memory_guard.ebpf._loader._has_cap_bpf", return_value=True),
            patch("memory_guard.ebpf._loader._cgroupv2_mounted", return_value=True),
        )

    def test_backend_bcc_when_bcc_importable(self):
        """backend is 'bcc' when bcc is importable and all checks pass."""
        with patch("memory_guard.ebpf._loader.sys") as mock_sys, \
             patch("memory_guard.ebpf._loader._kernel_version", return_value=(5, 15)), \
             patch("memory_guard.ebpf._loader._has_cap_bpf", return_value=True), \
             patch("memory_guard.ebpf._loader._cgroupv2_mounted", return_value=True), \
             patch("memory_guard.ebpf._loader._detect_backend", return_value="bcc"):
            mock_sys.platform = "linux"
            loader = BPFProbeLoader()
            assert loader.available is True
            assert loader.backend == "bcc"

    def test_backend_libbpf_when_only_libbpf_importable(self):
        """backend is 'libbpf' when bcc is absent but libbpf-python is present."""
        with patch("memory_guard.ebpf._loader.sys") as mock_sys, \
             patch("memory_guard.ebpf._loader._kernel_version", return_value=(6, 1)), \
             patch("memory_guard.ebpf._loader._has_cap_bpf", return_value=True), \
             patch("memory_guard.ebpf._loader._cgroupv2_mounted", return_value=True), \
             patch("memory_guard.ebpf._loader._detect_backend", return_value="libbpf"):
            mock_sys.platform = "linux"
            loader = BPFProbeLoader()
            assert loader.available is True
            assert loader.backend == "libbpf"


class TestBPFProbeLoaderAPI:

    def test_check_capabilities_returns_bool_str_tuple(self):
        """check_capabilities() always returns (bool, str)."""
        loader = _make_loader_unavailable("test reason")
        ok, reason = loader.check_capabilities()
        assert isinstance(ok, bool)
        assert isinstance(reason, str)
        assert ok is False
        assert reason == "test reason"

    def test_repr_before_first_probe(self):
        """repr contains 'not-yet-probed' before first access of available."""
        loader = BPFProbeLoader()
        assert "not-yet-probed" in repr(loader)


# ===========================================================================
# MemguardBPFEvent
# ===========================================================================

class TestMemguardBPFEvent:

    def test_to_dict_includes_all_required_fields(self):
        """to_dict() includes ts_ns, event_type, pressure_bytes, pid, cgroup_id."""
        event = MemguardBPFEvent(
            ts_ns=123456789,
            event_type=EVENT_MEMORY_HIGH,
            pressure_bytes=1024 * 1024,
            pid=42,
            cgroup_id="/kubepods/pod-abc",
        )
        d = event.to_dict()
        assert d["ts_ns"]          == 123456789
        assert d["event_type"]     == "memory_high"
        assert d["pressure_bytes"] == 1024 * 1024
        assert d["pid"]            == 42
        assert d["cgroup_id"]      == "/kubepods/pod-abc"

    def test_to_dict_merges_extra_fields(self):
        """Extra dict is merged (unpacked) into the to_dict() output."""
        event = MemguardBPFEvent(
            ts_ns=1,
            event_type=EVENT_PAGE_FAULT,
            pressure_bytes=0,
            pid=99,
            cgroup_id="",
            extra={"fault_address": 0xDEADBEEF, "fault_flags": 3},
        )
        d = event.to_dict()
        assert d["fault_address"] == 0xDEADBEEF
        assert d["fault_flags"]   == 3

    def test_event_type_constants_have_correct_values(self):
        """Each EVENT_* constant matches its expected string value."""
        assert EVENT_MEMORY_HIGH == "memory_high"
        assert EVENT_OOM_KILL    == "oom_kill"
        assert EVENT_PREEMPTION  == "preemption"
        assert EVENT_PAGE_FAULT  == "page_fault"
        assert EVENT_MMAP_GROWTH == "mmap_growth"


# ===========================================================================
# MemguardBPFSession
# ===========================================================================

class TestMemguardBPFSession:

    def test_enter_is_noop_when_loader_unavailable(self):
        """Session.__enter__ returns self without loading probes when unavailable."""
        loader = _make_loader_unavailable("test: no BPF on CI")
        session = MemguardBPFSession(loader=loader)
        with session as s:
            assert s is session
            assert s.available is False
            assert s.manager is None

    def test_available_false_before_enter(self):
        """available is False before the with-block is entered."""
        loader = _make_loader_available()
        session = MemguardBPFSession(loader=loader)
        assert session.available is False

    def test_exit_safe_when_manager_never_set(self):
        """__exit__ does not raise when manager is None (session never started)."""
        loader = _make_loader_unavailable()
        session = MemguardBPFSession(loader=loader)
        # Call __exit__ directly without __enter__
        session.__exit__(None, None, None)   # must not raise
        assert session.available is False
