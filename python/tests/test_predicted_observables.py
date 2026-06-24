"""Predicted-observable convention for DEM models and the PyMatching-compatible API.

Section 6 (logical-observable proof). A decoder's *logical outcome* is the set of
observable flips it predicts, not the raw correction vector. These tests pin that
convention down two ways:

* ``DemModel.predicted_observables(correction)`` must equal the GF(2) product
  ``(observables_matrix() @ correction) & 1`` for arbitrary corrections, both for a
  hand-written DEM and a stim-derived one.
* ``pymatching_compat.Matching.from_check_matrix(H, faults_matrix=L)`` must return,
  from ``.decode(s)``, exactly the observable flips ``(L @ edge_correction) & 1`` where
  ``edge_correction = .decode_to_edges_array(s)``.
"""
import numpy as np
import pytest

from qector_decoder_v3 import dem, pymatching_compat


HAND_DEM = """
error(0.1) D0 L0
error(0.1) D0 D1
error(0.1) D1 L0
error(0.05) D0 D1 L0
error(0.08) D1 L0 L1
error(0.07) D0 L1
"""


def test_predicted_observables_hand_dem():
    model = dem.parse_dem(HAND_DEM)
    OM = np.asarray(model.observables_matrix(), np.uint8)
    assert OM.shape == (model.num_observables, model.num_errors)
    ne = model.num_errors
    rng = np.random.default_rng(0)
    for _ in range(500):
        corr = (rng.random(ne) < 0.4).astype(np.uint8)
        predicted = np.asarray(model.predicted_observables(corr), np.uint8)
        manual = (OM @ corr) & 1
        assert predicted.shape == (model.num_observables,)
        assert np.array_equal(predicted, manual)


def test_predicted_observables_stim_dem():
    stim = pytest.importorskip("stim")
    circ = stim.Circuit.generated(
        "surface_code:rotated_memory_x", distance=3, rounds=3,
        after_clifford_depolarization=0.01,
        before_measure_flip_probability=0.01,
        after_reset_flip_probability=0.01,
    )
    sdem = circ.detector_error_model(decompose_errors=True)
    model = dem.from_stim(sdem)
    OM = np.asarray(model.observables_matrix(), np.uint8)
    assert OM.shape == (model.num_observables, model.num_errors)
    ne = model.num_errors
    rng = np.random.default_rng(1)
    for _ in range(300):
        corr = (rng.random(ne) < 0.15).astype(np.uint8)
        predicted = np.asarray(model.predicted_observables(corr), np.uint8)
        manual = (OM @ corr) & 1
        assert np.array_equal(predicted, manual)


def test_matching_decode_equals_faults_matrix_times_edge_correction():
    # A small repetition-style matching graph with two logical observables defined
    # over the qubit (=column / edge) index by the faults_matrix L.
    H = np.array(
        [[1, 1, 0, 0, 0],
         [0, 1, 1, 0, 0],
         [0, 0, 1, 1, 0],
         [0, 0, 0, 1, 1]],
        dtype=np.uint8,
    )
    L = np.array(
        [[1, 0, 0, 0, 0],
         [0, 0, 1, 0, 1]],
        dtype=np.uint8,
    )
    m = pymatching_compat.Matching.from_check_matrix(H, faults_matrix=L)
    assert m.num_fault_ids == L.shape[0]
    # decode_to_edges_array returns a correction with one bit per edge, and edges are
    # ordered to match the columns of H, so L (column-indexed) applies directly.
    assert m.num_edges == H.shape[1]

    n_checks = H.shape[0]
    rng = np.random.default_rng(2)
    for _ in range(400):
        # Build an achievable syndrome from a random edge set so decoding is well posed.
        true_edges = (rng.random(H.shape[1]) < 0.3).astype(np.uint8)
        s = (H @ true_edges) & 1
        s = s.astype(np.uint8)
        assert s.shape == (n_checks,)

        edge_corr = np.asarray(m.decode_to_edges_array(s), np.uint8)
        assert edge_corr.shape == (m.num_edges,)
        # The edge correction must reproduce the syndrome.
        assert np.array_equal((H @ edge_corr) & 1, s)

        predicted = np.asarray(m.decode(s), np.uint8)
        manual = (L @ edge_corr) & 1
        assert predicted.shape == (L.shape[0],)
        assert np.array_equal(predicted, manual)
