"""Tests for qector_decoder_v3.backend — automatic backend selection."""
import numpy as np
import pytest

import qector_decoder_v3 as qd
from qector_decoder_v3 import codes
from qector_decoder_v3.backend import AutoDecoder, Backend, BackendConfig


def _faithful(out, H, syns):
    return all(np.array_equal((H @ out[i]) & 1, syns[i]) for i in range(len(syns)))


def test_selection_thresholds_cpu_only():
    code = codes.rotated_surface_code(5)
    cfg = BackendConfig(rayon_threshold=8, gpu_threshold=1 << 60, allow_gpu=False)
    ad = AutoDecoder(code.check_to_qubits, code.n_qubits, cfg)
    assert ad.select(1) == Backend.CPU_SINGLE
    assert ad.select(7) == Backend.CPU_SINGLE
    assert ad.select(8) == Backend.CPU_RAYON
    assert ad.select(10_000) == Backend.CPU_RAYON
    assert Backend.CPU_SINGLE in ad.available_backends()


def test_batch_decode_is_faithful_small_and_large():
    code = codes.rotated_surface_code(5)
    H = code.parity_check_matrix()
    cfg = BackendConfig(allow_gpu=False)
    ad = AutoDecoder(code.check_to_qubits, code.n_qubits, cfg)
    rng = np.random.default_rng(0)
    for n in (1, 5, 64, 500):
        syns = (rng.random((n, code.n_checks)) < 0.08).astype(np.uint8)
        out = np.asarray(ad.batch_decode(syns))
        assert out.shape == (n, code.n_qubits)
        assert _faithful(out, H, syns)


def test_single_decode_matches_reference():
    code = codes.repetition_code(15)
    H = code.parity_check_matrix()
    ad = AutoDecoder(code.check_to_qubits, code.n_qubits, BackendConfig(allow_gpu=False))
    rng = np.random.default_rng(2)
    for _ in range(50):
        e = (rng.random(code.n_qubits) < 0.1).astype(np.uint8)
        s = (H @ e) & 1
        c = np.asarray(ad.decode(s.astype(np.uint8))).astype(np.uint8)
        assert np.array_equal((H @ c) & 1, s)


def test_force_backend_override():
    code = codes.repetition_code(9)
    cfg = BackendConfig(force=Backend.CPU_RAYON, allow_gpu=False)
    ad = AutoDecoder(code.check_to_qubits, code.n_qubits, cfg)
    assert ad.select(1) == Backend.CPU_RAYON


def test_diagnostics_structure():
    code = codes.repetition_code(9)
    ad = AutoDecoder(code.check_to_qubits, code.n_qubits, BackendConfig(allow_gpu=False))
    syns = (np.random.default_rng(0).random((20, code.n_checks)) < 0.1).astype(np.uint8)
    ad.batch_decode(syns)
    d = ad.diagnostics()
    assert "available_backends" in d
    assert d["last_backend"] in Backend.ALL
    assert d["calls"] >= 1


def test_gpu_path_faithful_if_available():
    if not (qd.cuda_is_available() or qd.opencl_is_available()):
        pytest.skip("no GPU backend available")
    code = codes.rotated_surface_code(5)
    H = code.parity_check_matrix()
    cfg = BackendConfig(gpu_threshold=16, allow_gpu=True)
    ad = AutoDecoder(code.check_to_qubits, code.n_qubits, cfg)
    syns = (np.random.default_rng(1).random((512, code.n_checks)) < 0.08).astype(np.uint8)
    out = np.asarray(ad.batch_decode(syns))
    assert _faithful(out, H, syns)
    assert ad.last_backend in (Backend.CUDA, Backend.OPENCL, Backend.CPU_RAYON)


def test_calibrate_sets_threshold_and_diagnostics():
    code = codes.rotated_surface_code(5)
    ad = AutoDecoder(code.check_to_qubits, code.n_qubits)
    info = ad.calibrate(sizes=(64, 256), repeats=1)
    assert "gpu_threshold" in info
    assert ad.diagnostics()["calibrated"] is True
    # after calibration the decoder must still decode faithfully
    H = code.parity_check_matrix()
    syns = (np.random.default_rng(0).random((300, code.n_checks)) < 0.08).astype(np.uint8)
    out = np.asarray(ad.batch_decode(syns))
    assert _faithful(out, H, syns)
