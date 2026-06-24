"""Belief-matching proof (section 4): BP convergence behaviour at d=3.

Exercises the belief-propagation inner loop:
  (a) belief predictions are valid 0/1 vectors of the right shape with both
      a small (``max_iter=5``) and a large (``max_iter=40``) iteration budget;
  (b) belief decoding is deterministic (same shots -> same output on repeat);
  (c) more BP iterations do not degrade accuracy: pooled LER at max_iter=40
      is <= pooled LER at max_iter=5 plus a small slack;
  (d) the raw BP posterior LLRs are finite (no NaN/inf) on a small case.

Distance fixed to 3 and shots small (QECTOR rebuilds matching per shot).
"""

import numpy as np
import pytest

stim = pytest.importorskip("stim")
pymatching = pytest.importorskip("pymatching")

from qector_decoder_v3.belief_matching import BeliefMatching, build_matching_matrices
from qector_decoder_v3._bp_core import build_incidence, sum_product_bp


def _circuit(d, p):
    return stim.Circuit.generated(
        "surface_code:rotated_memory_x",
        distance=d,
        rounds=d,
        after_clifford_depolarization=p,
        before_measure_flip_probability=p,
        after_reset_flip_probability=p,
    )


def test_belief_bp_convergence():
    d, p, N, seed = 3, 0.01, 400, 0

    circ = _circuit(d, p)
    sdem = circ.detector_error_model(decompose_errors=True)

    bm_small = BeliefMatching.from_detector_error_model(sdem, max_iter=5)
    bm_large = BeliefMatching.from_detector_error_model(sdem, max_iter=40)
    no = bm_small.num_observables
    assert bm_large.num_observables == no

    det, obs = circ.compile_detector_sampler(seed=seed).sample(shots=N, separate_observables=True)
    det = det.astype(np.uint8)
    obs = obs.astype(np.uint8)

    pred_small = np.asarray(bm_small.decode_batch(det), np.uint8).reshape(N, -1)
    pred_large = np.asarray(bm_large.decode_batch(det), np.uint8).reshape(N, -1)

    # (a) Valid 0/1 of correct shape in both cases.
    for pred in (pred_small, pred_large):
        assert pred.shape == (N, no)
        assert set(np.unique(pred).tolist()).issubset({0, 1})

    # (b) Determinism: decoding the same shots again yields identical output.
    pred_small_again = np.asarray(bm_small.decode_batch(det), np.uint8).reshape(N, -1)
    pred_large_again = np.asarray(bm_large.decode_batch(det), np.uint8).reshape(N, -1)
    assert np.array_equal(pred_small, pred_small_again)
    assert np.array_equal(pred_large, pred_large_again)

    # (c) More BP iterations must not degrade accuracy (pooled, small slack).
    err_small = int(np.any(pred_small != obs, axis=1).sum())
    err_large = int(np.any(pred_large != obs, axis=1).sum())
    print(f"[bp_conv] shots={N} err(max_iter=5)={err_small} err(max_iter=40)={err_large}")
    slack = max(3, int(0.10 * err_small))
    assert err_large <= err_small + slack

    # (d) Raw BP posterior LLRs are finite on a handful of small cases.
    M = build_matching_matrices(sdem)
    hic, hie = build_incidence(M.hyper_check)
    n_checks = M.num_detectors
    n_hyper = M.hyper_check.shape[1]
    p_clip = np.clip(M.hyper_priors, 1e-15, 1 - 1e-15)
    prior_llr = np.log((1.0 - p_clip) / p_clip)
    assert np.all(np.isfinite(prior_llr))

    for i in range(min(8, N)):
        posterior = sum_product_bp(hic, hie, n_checks, n_hyper, prior_llr, det[i].astype(np.uint8), 20)
        assert posterior.shape == (n_hyper,)
        assert np.all(np.isfinite(posterior)), f"non-finite BP posterior on shot {i}"
