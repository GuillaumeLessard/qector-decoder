"""
Test suite for AutoDecoder Self Auto Debugging & Multi-Tier Fallback Recovery
"""

import numpy as np
import pytest
from unittest.mock import patch

import qector_decoder_v3 as qd
from qector_decoder_v3 import codes
from qector_decoder_v3.backend import AutoDecoder, Backend, BackendConfig


def test_auto_debug_fallback_recovery_on_exception():
    code = codes.rotated_surface_code(5)
    H = code.parity_check_matrix()

    cfg = BackendConfig(force=Backend.CUDA, allow_gpu=True, enable_auto_debug=True)
    ad = AutoDecoder(code.check_to_qubits, code.n_qubits, cfg)

    # Mock CUDA decoder to raise a simulated hardware/driver exception
    mock_cuda = ad._get_cuda()
    if mock_cuda is not None:
        with patch.object(mock_cuda, "batch_decode", side_effect=RuntimeError("Simulated CUDA OOM Error")):
            syns = (np.random.default_rng(42).random((32, code.n_checks)) < 0.08).astype(np.uint8)

            # The auto-debug engine should catch the CUDA exception and gracefully fall back to CPU_RAYON / CPU_SINGLE
            out = ad.batch_decode(syns)
            assert out.shape == (32, code.n_qubits)
            assert all(np.array_equal((H @ out[i]) & 1, syns[i]) for i in range(32))

            diag = ad.diagnostics()
            assert diag["backend_health"][Backend.CUDA] is False
            assert len(diag["debug_log"]) >= 1
            assert "Simulated CUDA OOM Error" in diag["debug_log"][0]["error"]


def test_reset_backend_health():
    code = codes.repetition_code(7)
    ad = AutoDecoder(code.check_to_qubits, code.n_qubits)
    ad._diag.backend_health[Backend.CUDA] = False

    ad.reset_backend_health()
    assert ad._diag.backend_health[Backend.CUDA] is True
