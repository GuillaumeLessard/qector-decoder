"""Numerical-stability regression tests for the BP core.

Guards the failure that A10-01's strict `set_edge_weights` validator exposed:
`sum_product_bp` produced NaN posteriors whenever a variable-to-check message
was exactly zero, because the log-domain leave-one-out update computed
`log(0) - log(0)`. The NaN propagated into `-log(p_e)` and the matcher then
rejected the weights, so `BeliefMatching.decode` raised on a 2x3 code.
"""

import numpy as np
import pytest
from qector_decoder_v3._bp_core import (
    batch_sum_product_bp,
    build_incidence,
    sum_product_bp,
)
from qector_decoder_v3.belief_matching import BeliefMatching

# The minimal code that triggers exact cancellation: at iteration 2 the middle
# edge's v2c is prior + 0 - prior == 0, so tanh(0) == 0 and log|t| == -inf.
_H = np.array([[1, 1, 0], [0, 1, 1]], dtype=np.uint8)
_PRIORS = [0.08, 0.08, 0.08]
_SYNDROME = np.array([1, 0], dtype=np.uint8)


def _bp_args(max_iter=20):
    bm = BeliefMatching.from_numpy_h(_H, _PRIORS, max_iter=max_iter)
    return bm, (bm._hic, bm._hie, bm.n_checks, bm._n_hyper, bm._prior_llr)


def test_sum_product_bp_posteriors_are_finite():
    """The single-shot path must never emit NaN or inf."""
    _, (ic, ie, n_checks, n_hyper, prior_llr) = _bp_args()
    post = sum_product_bp(ic, ie, n_checks, n_hyper, prior_llr, _SYNDROME, 20)
    assert np.all(np.isfinite(post)), f"non-finite posterior: {post}"


def test_batch_sum_product_bp_posteriors_are_finite():
    """The A10-03 batch path shares the update, so it shares the guard."""
    _, (ic, ie, n_checks, n_hyper, prior_llr) = _bp_args()
    shots = np.repeat(_SYNDROME[None, :], 4, axis=0)
    post = batch_sum_product_bp(ic, ie, n_checks, n_hyper, prior_llr, shots, 20)
    assert post.shape == (4, n_hyper)
    assert np.all(np.isfinite(post)), f"non-finite posterior: {post}"


def test_batch_bp_matches_single_shot():
    """Batch and single-shot BP must agree bit-for-bit on the same syndrome."""
    _, (ic, ie, n_checks, n_hyper, prior_llr) = _bp_args()
    single = sum_product_bp(ic, ie, n_checks, n_hyper, prior_llr, _SYNDROME, 20)
    batch = batch_sum_product_bp(ic, ie, n_checks, n_hyper, prior_llr, _SYNDROME[None, :], 20)
    np.testing.assert_allclose(batch[0], single, rtol=0, atol=1e-12)


def test_belief_matching_decode_does_not_raise_on_cancelling_code():
    """The end-to-end symptom: decode() must return, not raise on NaN weights."""
    bm = BeliefMatching.from_numpy_h(_H, _PRIORS)
    corr = bm.decode(_SYNDROME)
    assert len(corr) == _H.shape[1]
    assert np.all(np.isin(np.asarray(corr), (0, 1)))


def test_belief_matching_decode_batch_does_not_raise():
    """Same guarantee on the batched entry point."""
    bm = BeliefMatching.from_numpy_h(_H, _PRIORS)
    out = bm.decode_batch(np.repeat(_SYNDROME[None, :], 8, axis=0))
    assert out.shape[0] == 8
    assert np.all(np.isin(out, (0, 1)))


@pytest.mark.parametrize("p", [1e-12, 1e-6, 0.02, 0.2, 0.49])
def test_finite_across_prior_magnitudes(p):
    """Extreme priors must not reintroduce a non-finite posterior."""
    bm = BeliefMatching.from_numpy_h(_H, [p, p, p])
    post = sum_product_bp(bm._hic, bm._hie, bm.n_checks, bm._n_hyper, bm._prior_llr, _SYNDROME, 30)
    assert np.all(np.isfinite(post)), f"non-finite at p={p}: {post}"
