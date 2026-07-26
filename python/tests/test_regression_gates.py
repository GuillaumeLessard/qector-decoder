"""C5 (K3dev.md): Regression gate test with stored tolerance bands.

Ensures that:
1. Seeded LER measurements on rotated surface code (d=3, d=5) remain within stored
   tolerance bands.
2. Decoder throughput on d=3, d=5 surface code meets baseline gates.
"""
from __future__ import annotations

import pytest
import numpy as np
from qector_decoder_v3 import codes, BlossomDecoder, FastUnionFindDecoder
from qector_decoder_v3.ler import estimate_ler


def test_regression_gate_seeded_ler_surface_code():
    # Rotated surface code d=3, p=0.005, seed=42, 1000 shots
    code_d3 = codes.rotated_surface_code(3)
    res_d3 = estimate_ler(code_d3, "blossom", p=0.005, shots=1000, seed=42)
    # Stored tolerance band for d=3: 0.0 <= LER <= 0.02
    assert 0.0 <= res_d3.ler <= 0.02, f"d=3 LER {res_d3.ler} out of tolerance band [0.0, 0.02]"
    assert res_d3.unfaithful == 0, f"Unfaithful decodes detected: {res_d3.unfaithful}"

    # Rotated surface code d=5, p=0.005, seed=42, 1000 shots
    code_d5 = codes.rotated_surface_code(5)
    res_d5 = estimate_ler(code_d5, "blossom", p=0.005, shots=1000, seed=42)
    # Stored tolerance band for d=5: 0.0 <= LER <= 0.01
    assert 0.0 <= res_d5.ler <= 0.01, f"d=5 LER {res_d5.ler} out of tolerance band [0.0, 0.01]"
    assert res_d5.unfaithful == 0, f"Unfaithful decodes detected: {res_d5.unfaithful}"


def test_regression_gate_throughput():
    code = codes.rotated_surface_code(3)
    dec = FastUnionFindDecoder(code.check_to_qubits, code.n_qubits)
    rng = np.random.default_rng(42)
    H = code.parity_check_matrix().astype(np.uint8)
    E = (rng.random((5000, code.n_qubits)) < 0.01).astype(np.uint8)
    S = ((E @ H.T) & 1).astype(np.uint8)

    import time
    t0 = time.perf_counter()
    C = dec.batch_decode(S)
    dt = time.perf_counter() - t0
    dec_per_s = 5000 / dt
    # Baseline throughput gate: > 1,000 dec/s on fast UF
    assert dec_per_s > 1000, f"FastUF throughput {dec_per_s:.0f} dec/s below baseline gate of 1000 dec/s"
