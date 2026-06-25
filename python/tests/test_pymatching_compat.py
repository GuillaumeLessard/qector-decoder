"""Tests for qector_decoder_v3.pymatching_compat.

The API-shape tests run unconditionally.  The cross-validation tests run against
the real ``pymatching`` package when installed and assert the strong invariant
that QECTOR's matching is **never heavier** than PyMatching's (both find a
minimum-weight matching; ties may pick different representatives).
"""

import numpy as np
import pytest

from qector_decoder_v3 import codes
from qector_decoder_v3.pymatching_compat import Matching


def test_from_check_matrix_shapes():
    code = codes.rotated_surface_code(5)
    H = code.parity_check_matrix()
    m = Matching.from_check_matrix(H)
    assert m.num_detectors == code.n_checks
    assert m.num_edges == code.n_qubits
    assert m.num_fault_ids == code.n_qubits  # default fault id per edge


def test_decode_returns_edge_correction_by_default():
    code = codes.repetition_code(11)
    H = code.parity_check_matrix()
    m = Matching.from_check_matrix(H)
    rng = np.random.default_rng(0)
    for _ in range(50):
        e = (rng.random(code.n_qubits) < 0.1).astype(np.uint8)
        s = (H @ e) & 1
        corr = m.decode(s)
        assert corr.shape == (code.n_qubits,)
        assert np.array_equal((H @ corr.astype(np.uint8)) & 1, s)


def test_faults_matrix_returns_observables():
    code = codes.repetition_code(9)
    H = code.parity_check_matrix()
    L = code.logicals_matrix()
    m = Matching.from_check_matrix(H, faults_matrix=L)
    assert m.num_fault_ids == 1
    s = code.syndrome(code.random_error(0.1, np.random.default_rng(1)))
    pred = m.decode(s)
    assert pred.shape == (1,)


def test_incremental_construction():
    m = Matching()
    m.add_edge(0, 1, fault_ids=0)
    m.add_edge(1, 2, fault_ids={1})
    m.add_boundary_edge(0, fault_ids=2)
    assert m.num_detectors == 3
    assert m.num_edges == 3
    edges = m.edges()
    assert edges[2][1] is None  # boundary edge


def test_decode_batch():
    code = codes.rotated_surface_code(5)
    H = code.parity_check_matrix()
    m = Matching.from_check_matrix(H)
    rng = np.random.default_rng(3)
    shots = (rng.random((64, code.n_checks)) < 0.08).astype(np.uint8)
    out = m.decode_batch(shots)
    assert out.shape == (64, code.n_qubits)
    for i in range(64):
        assert np.array_equal((H @ out[i].astype(np.uint8)) & 1, shots[i])


@pytest.mark.parametrize(
    "code",
    [
        codes.repetition_code(11),
        codes.rotated_surface_code(5),
        codes.rotated_surface_code(7),
        codes.toric_code(4),
    ],
    ids=lambda c: c.name,
)
def test_weight_optimal_vs_pymatching(code):
    pymatching = pytest.importorskip("pymatching")
    H = code.parity_check_matrix()
    qm = Matching.from_check_matrix(H)
    rm = pymatching.Matching.from_check_matrix(H)
    rng = np.random.default_rng(11)
    worse = 0
    n = 300
    for _ in range(n):
        e = (rng.random(code.n_qubits) < 0.07).astype(np.uint8)
        s = (H @ e) & 1
        cq = np.asarray(qm.decode(s)).astype(np.uint8)
        cr = np.asarray(rm.decode(s)).astype(np.uint8)
        assert np.array_equal((H @ cq) & 1, s)
        if int(cq.sum()) > int(cr.sum()):
            worse += 1
    assert worse == 0, f"QECTOR heavier than PyMatching {worse}/{n} on {code.name}"


def test_observable_agreement_vs_pymatching():
    pymatching = pytest.importorskip("pymatching")
    code = codes.rotated_surface_code(5)
    H = code.parity_check_matrix()
    L = code.logicals_matrix()
    qm = Matching.from_check_matrix(H, faults_matrix=L)
    rm = pymatching.Matching.from_check_matrix(H, faults_matrix=L)
    rng = np.random.default_rng(7)
    agree = 0
    n = 400
    for _ in range(n):
        e = (rng.random(code.n_qubits) < 0.05).astype(np.uint8)
        s = (H @ e) & 1
        if np.array_equal(qm.decode(s), rm.decode(s)):
            agree += 1
    # equal-weight ties may resolve to different cosets; agreement should still be high.
    assert agree / n > 0.9


def test_from_detector_error_model_text():
    text = "error(0.1) D0 L0\nerror(0.1) D0 D1\nerror(0.1) D1 L0\n"
    m = Matching.from_detector_error_model(text)
    assert m.num_detectors == 2
    assert m.num_fault_ids == 1
    pred = m.decode(np.array([1, 0], dtype=np.uint8))
    assert pred.shape == (1,)
