"""Tests for qector_decoder_v3.gpu_backend — the CuPy GPU backend abstraction.

Covers BOTH paths:
  * the CPU-fallback path (runs everywhere, including CuPy-less machines), and
  * the real GPU path (guarded with ``skipif(not gpu_available())`` so it only
    runs where a usable CUDA device + CuPy are present — e.g. this machine's
    GTX 1660 Ti with cupy 14.x).
"""

from types import ModuleType

import numpy as np
import pytest
from qector_decoder_v3 import gpu_backend as gb

GPU = gb.gpu_available()
requires_gpu = pytest.mark.skipif(not GPU, reason="no usable CuPy/CUDA device")


@pytest.fixture(autouse=True)
def _clean_state():
    """Each test starts with prefer_gpu=True and zeroed telemetry."""
    gb.set_prefer_gpu(True)
    gb.reset_telemetry()
    yield
    gb.set_prefer_gpu(True)
    gb.reset_telemetry()


# ---------------------------------------------------------------------------
# Detection
# ---------------------------------------------------------------------------
def test_detection_functions_return_bool():
    assert isinstance(gb.has_cupy(), bool)
    assert isinstance(gb.has_cuda_rust(), bool)
    assert isinstance(gb.gpu_available(), bool)


def test_gpu_available_implies_cupy():
    # gpu_available() can only be true if cupy is importable.
    if gb.gpu_available():
        assert gb.has_cupy() is True


@requires_gpu
def test_detection_truthy_on_this_machine():
    # This machine has cupy + a usable CUDA device, so all of these hold.
    assert gb.has_cupy() is True
    assert gb.gpu_available() is True
    bk = gb.get_backend()
    assert isinstance(bk.device_name, str) and bk.device_name
    assert isinstance(bk.total_mem_bytes, int) and bk.total_mem_bytes > 0


# ---------------------------------------------------------------------------
# get_array_module / xp dispatch
# ---------------------------------------------------------------------------
def test_get_array_module_cpu_dispatch():
    # prefer_gpu=False always yields numpy regardless of hardware.
    assert gb.get_array_module(np.zeros(4, np.uint8), prefer_gpu=False) is np
    assert gb.xp(prefer_gpu=False) is np


def test_get_array_module_returns_a_module():
    assert isinstance(gb.get_array_module(prefer_gpu=True), ModuleType)
    assert isinstance(gb.xp(prefer_gpu=True), ModuleType)


def test_set_prefer_gpu_global_override_forces_numpy():
    gb.set_prefer_gpu(False)
    # Even with prefer_gpu=True per-call, the global override wins.
    assert gb.xp(prefer_gpu=True) is np
    assert gb.get_array_module(np.zeros(4, np.uint8), prefer_gpu=True) is np


@requires_gpu
def test_get_array_module_gpu_dispatch():
    import cupy as cp

    assert gb.xp(prefer_gpu=True) is cp
    assert gb.get_array_module(np.zeros(4, np.uint8), prefer_gpu=True) is cp

    dev = gb.to_device(np.zeros(4, np.uint8))
    # Data already on the GPU forces cupy even when prefer_gpu=False.
    assert gb.get_array_module(dev, prefer_gpu=False) is cp


# ---------------------------------------------------------------------------
# is_on_gpu
# ---------------------------------------------------------------------------
def test_is_on_gpu_false_for_numpy_and_list():
    assert gb.is_on_gpu(np.arange(8)) is False
    assert gb.is_on_gpu([1, 2, 3]) is False


@requires_gpu
def test_is_on_gpu_true_for_device_array():
    dev = gb.to_device(np.arange(8, dtype=np.uint8))
    assert gb.is_on_gpu(dev) is True


# ---------------------------------------------------------------------------
# Host/device round trip + telemetry
# ---------------------------------------------------------------------------
def test_roundtrip_identity_cpu_path():
    gb.set_prefer_gpu(False)  # force host path even if a GPU exists
    host_in = np.arange(256, dtype=np.uint8).reshape(16, 16)
    dev = gb.to_device(host_in)
    assert not gb.is_on_gpu(dev)  # stayed on host
    back = gb.to_host(dev)
    np.testing.assert_array_equal(back, host_in)
    # to_device asked for the GPU but policy sent it to host => one fallback.
    assert gb.TELEMETRY["fallbacks"] >= 1


def test_to_host_on_numpy_is_noop_and_uncounted():
    host = np.arange(10, dtype=np.uint8)
    out = gb.to_host(host)
    np.testing.assert_array_equal(out, host)
    assert gb.TELEMETRY["d2h"] == 0


@requires_gpu
def test_roundtrip_identity_gpu_path_and_telemetry():
    import cupy as cp

    host_in = np.arange(256, dtype=np.uint8).reshape(16, 16)
    dev = gb.to_device(host_in)
    assert gb.is_on_gpu(dev)
    assert isinstance(dev, cp.ndarray)
    assert gb.TELEMETRY["h2d"] == 1

    back = gb.to_host(dev)
    assert isinstance(back, np.ndarray)
    assert gb.TELEMETRY["d2h"] == 1
    np.testing.assert_array_equal(back, host_in)

    # asnumpy is an alias of to_host.
    again = gb.asnumpy(dev)
    np.testing.assert_array_equal(again, host_in)
    assert gb.TELEMETRY["d2h"] == 2


@requires_gpu
def test_to_device_already_on_gpu_is_noop():
    dev = gb.to_device(np.zeros(4, np.uint8))
    assert gb.TELEMETRY["h2d"] == 1
    dev2 = gb.to_device(dev)  # already on device -> no extra transfer
    assert dev2 is dev
    assert gb.TELEMETRY["h2d"] == 1


# ---------------------------------------------------------------------------
# Telemetry counters
# ---------------------------------------------------------------------------
def test_telemetry_keys_and_reset():
    assert set(gb.TELEMETRY) == {"h2d", "d2h", "gpu_calls", "fallbacks"}
    gb.note_gpu_call(3)
    gb.note_fallback(2)
    assert gb.TELEMETRY["gpu_calls"] == 3
    assert gb.TELEMETRY["fallbacks"] == 2
    gb.reset_telemetry()
    assert all(v == 0 for v in gb.TELEMETRY.values())


def test_reset_telemetry_mutates_in_place():
    ref = gb.TELEMETRY  # external modules may hold this reference
    gb.note_gpu_call()
    gb.reset_telemetry()
    assert ref is gb.TELEMETRY  # same object, not rebound
    assert ref["gpu_calls"] == 0


# ---------------------------------------------------------------------------
# GpuBackend dataclass / get_backend
# ---------------------------------------------------------------------------
def test_get_backend_is_cached():
    assert gb.get_backend() is gb.get_backend()


def test_backend_dataclass_fields_and_summary():
    bk = gb.get_backend()
    assert isinstance(bk, gb.GpuBackend)
    assert isinstance(bk.cupy_available, bool)
    assert isinstance(bk.cuda_rust_available, bool)
    assert bk.device_name is None or isinstance(bk.device_name, str)
    assert bk.total_mem_bytes is None or isinstance(bk.total_mem_bytes, int)
    assert isinstance(bk.prefer_gpu, bool)

    assert isinstance(bk.module(), ModuleType)

    s = bk.summary()
    expected = {
        "cupy_available",
        "cuda_rust_available",
        "gpu_available",
        "device_name",
        "total_mem_bytes",
        "prefer_gpu",
        "active_module",
    }
    assert expected.issubset(s)
    assert s["active_module"] in ("numpy", "cupy")


def test_backend_module_respects_prefer_flag():
    bk = gb.get_backend()
    gb.set_prefer_gpu(False)
    assert bk.module() is np  # cached snapshot tracks the global flag
    assert bk.prefer_gpu is False
