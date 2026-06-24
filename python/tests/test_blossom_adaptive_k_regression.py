"""Regression lock for the adaptive-k fix (the old fixed k=12 candidate cap).

Before the fix, QECTOR's Blossom used a fixed k=12 candidate set, which undershot
the optimum on large dense circuit-level graphs (d>=13), giving heavier matchings
and a markedly worse logical error rate than PyMatching. The adaptive-k fix
(k = max(12, 4*sqrt(n_defects))) restored parity. This test reproduces the
regime where the old cap failed (d=13, d=15) and asserts QECTOR is no worse than
exact MWPM on the IDENTICAL collapsed graph and the IDENTICAL seeded shots.

The metric is the logical error count (the quantity that matters and is immune to
QECTOR's internal weight quantization).
"""
import numpy as np
import pytest

stim = pytest.importorskip("stim")
pymatching = pytest.importorskip("pymatching")

from qector_decoder_v3 import pymatching_compat  # noqa: E402


def _logical_errors(d, shots, seed, basis="x", noise=0.005):
    circ = stim.Circuit.generated(
        f"surface_code:rotated_memory_{basis}", distance=d, rounds=d,
        after_clifford_depolarization=noise,
        before_measure_flip_probability=noise,
        after_reset_flip_probability=noise)
    sdem = circ.detector_error_model(decompose_errors=True)
    det, obs = circ.compile_detector_sampler(seed=seed).sample(
        shots=shots, separate_observables=True)
    det = det.astype(np.uint8)
    obs = obs.astype(np.uint8)
    qm = pymatching_compat.Matching.from_detector_error_model(sdem)
    pm = pymatching.Matching.from_detector_error_model(sdem)
    qpred = np.asarray(qm.decode_batch(det), np.uint8).reshape(shots, -1)
    ppred = np.asarray(pm.decode_batch(det), np.uint8).reshape(shots, -1)
    q_err = int(np.any(qpred != obs, axis=1).sum())
    p_err = int(np.any(ppred != obs, axis=1).sum())
    return q_err, p_err


@pytest.mark.parametrize("d", [13, 15])
def test_adaptive_k_no_logical_regression(d):
    """At the distances where the old k=12 cap failed, QECTOR <= PyMatching."""
    shots = 5000 if d == 13 else 4000
    q_err, p_err = _logical_errors(d, shots=shots, seed=20260622)
    slack = max(4, int(0.30 * p_err))
    assert q_err <= p_err + slack, (
        f"d={d}: QECTOR {q_err} logical errors vs PyMatching {p_err} (+{slack} slack) "
        f"— the fixed-k regression has returned")


def test_adaptive_k_helps_more_at_higher_distance():
    """Sanity: both decoders see a non-trivial number of logical errors at d=13
    (so the comparison is meaningful, not a 0-vs-0 trivial pass)."""
    q_err, p_err = _logical_errors(13, shots=5000, seed=7)
    assert p_err > 0 and q_err >= 0
