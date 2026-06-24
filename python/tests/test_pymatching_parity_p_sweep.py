"""Extended PyMatching parity (section 3): physical noise sweep.

Holds QECTOR vs PyMatching parity across a range of physical error rates
(0.002..0.010) at d in {5, 7}. QECTOR's ``pymatching_compat.Matching`` (collapsed
graphlike DEM) decodes the SAME seeded Stim shots as the reference
``pymatching.Matching``; at every noise point the logical error rates must agree
(overlapping Wilson 95% intervals AND QECTOR no worse than PyMatching beyond a
small tie/Poisson slack). This guards against accuracy drift as the DEM edge
weights shift with p.
"""
import math

import numpy as np
import pytest

stim = pytest.importorskip("stim")
pymatching = pytest.importorskip("pymatching")

from qector_decoder_v3 import pymatching_compat  # noqa: E402


def wilson(k, n, z=1.959963985):
    """Wilson score 95% interval for k successes out of n."""
    if n == 0:
        return (0.0, 1.0)
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    w = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return (max(0.0, c - w), min(1.0, c + w))


def _logical_errors(d, shots, seed, basis="x", noise=0.005):
    circ = stim.Circuit.generated(
        f"surface_code:rotated_memory_{basis}", distance=d, rounds=d,
        after_clifford_depolarization=noise,
        before_measure_flip_probability=noise,
        after_reset_flip_probability=noise,
    )
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


def _assert_parity(d, shots, seed, basis, noise):
    q_err, p_err = _logical_errors(d, shots, seed, basis=basis, noise=noise)
    ql, qh = wilson(q_err, shots)
    pl, ph = wilson(p_err, shots)
    overlap = not (qh < pl or ph < ql)
    assert overlap, (
        f"d={d} p={noise}: Wilson intervals disjoint — "
        f"QECTOR {q_err}/{shots} {ql:.4f}-{qh:.4f} vs "
        f"PyMatching {p_err}/{shots} {pl:.4f}-{ph:.4f}"
    )
    slack = max(4, int(0.3 * p_err))
    assert q_err <= p_err + slack, (
        f"d={d} p={noise}: QECTOR {q_err} logical errors vs "
        f"PyMatching {p_err} (+{slack} slack)"
    )


@pytest.mark.parametrize("noise", [0.002, 0.004, 0.006, 0.008, 0.010])
@pytest.mark.parametrize("d", [5, 7])
def test_pymatching_parity_p_sweep(d, noise):
    """QECTOR/PyMatching LER parity at each physical error rate."""
    seed = 20260625 + int(round(noise * 1000)) + d
    _assert_parity(d, shots=3000, seed=seed, basis="x", noise=noise)
