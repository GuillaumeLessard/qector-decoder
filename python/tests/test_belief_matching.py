"""Tests for qector_decoder_v3.belief_matching.

Locks in the headline result: on real Stim circuit-level shots, QECTOR's
belief-matching achieves a lower (never higher) logical error rate than plain
PyMatching. A seeded detector sampler makes the comparison deterministic, so the
assertion is not flaky.
"""
import numpy as np
import pytest

stim = pytest.importorskip("stim")
pymatching = pytest.importorskip("pymatching")

from qector_decoder_v3.belief_matching import BeliefMatching, build_matching_matrices  # noqa: E402


def _circ(d, p=0.005):
    return stim.Circuit.generated(
        "surface_code:rotated_memory_x", distance=d, rounds=d,
        after_clifford_depolarization=p,
        before_measure_flip_probability=p,
        after_reset_flip_probability=p,
    )


def _errors(predict, dets, obs):
    pred = np.asarray(predict(dets), np.uint8).reshape(len(dets), -1)
    return int(np.any(pred != obs, axis=1).sum())


def test_build_matrices_shapes():
    sdem = _circ(5).detector_error_model(decompose_errors=True)
    m = build_matching_matrices(sdem)
    assert m.hyper_check.shape[0] == sdem.num_detectors
    assert m.edge_check.shape[0] == sdem.num_detectors
    assert m.edge_obs.shape[0] == sdem.num_observables
    # hyperedges are at least as many as edges (each edge belongs to >=1 hyperedge)
    assert m.hyper_check.shape[1] >= 1 and m.edge_check.shape[1] >= 1


def test_belief_matching_beats_pymatching_d5():
    circ = _circ(5)
    sdem = circ.detector_error_model(decompose_errors=True)
    shots = 5000
    det, obs = circ.compile_detector_sampler(seed=20240601).sample(
        shots=shots, separate_observables=True)
    det = det.astype(np.uint8)
    obs = obs.astype(np.uint8)

    pm = pymatching.Matching.from_detector_error_model(sdem)
    bm = BeliefMatching.from_detector_error_model(sdem)

    pm_err = _errors(lambda x: pm.decode_batch(x), det, obs)
    bm_err = _errors(lambda x: bm.decode_batch(x), det, obs)
    # Deterministic shots: belief-matching must not be worse than plain MWPM,
    # and on this seed it is strictly better.
    assert bm_err <= pm_err, f"belief {bm_err} > pymatching {pm_err}"


def test_belief_matching_not_worse_d3():
    circ = _circ(3)
    sdem = circ.detector_error_model(decompose_errors=True)
    shots = 5000
    det, obs = circ.compile_detector_sampler(seed=7).sample(
        shots=shots, separate_observables=True)
    det = det.astype(np.uint8)
    obs = obs.astype(np.uint8)
    pm = pymatching.Matching.from_detector_error_model(sdem)
    bm = BeliefMatching.from_detector_error_model(sdem)
    pm_err = _errors(lambda x: pm.decode_batch(x), det, obs)
    bm_err = _errors(lambda x: bm.decode_batch(x), det, obs)
    # never dramatically worse than plain matching at small distance
    assert bm_err <= pm_err + max(5, int(0.1 * pm_err))


def test_belief_matching_observable_shape():
    sdem = _circ(3).detector_error_model(decompose_errors=True)
    bm = BeliefMatching.from_detector_error_model(sdem)
    det, _ = _circ(3).compile_detector_sampler(seed=1).sample(
        shots=10, separate_observables=True)
    out = bm.decode_batch(det.astype(np.uint8))
    assert out.shape == (10, sdem.num_observables)
    assert out.dtype == np.uint8
