"""Extended PyMatching parity (section 3): ``memory_x`` basis.

QECTOR's ``pymatching_compat.Matching`` decodes the *collapsed* graphlike DEM;
the reference ``pymatching.Matching`` decodes the same DEM. We feed both the
IDENTICAL seeded Stim shots and require their logical error rates to agree:
their Wilson 95% intervals must overlap AND QECTOR's logical-error count must
not exceed PyMatching's beyond a small tie/Poisson slack. This proves QECTOR is
a drop-in MWPM decoder on rotated-surface-code ``memory_x`` circuits across
distances 3..13.
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
        f"surface_code:rotated_memory_{basis}",
        distance=d,
        rounds=d,
        after_clifford_depolarization=noise,
        before_measure_flip_probability=noise,
        after_reset_flip_probability=noise,
    )
    sdem = circ.detector_error_model(decompose_errors=True)
    det, obs = circ.compile_detector_sampler(seed=seed).sample(
        shots=shots, separate_observables=True
    )
    det = det.astype(np.uint8)
    obs = obs.astype(np.uint8)
    qm = pymatching_compat.Matching.from_detector_error_model(sdem)
    pm = pymatching.Matching.from_detector_error_model(sdem)
    qpred = np.asarray(qm.decode_batch(det), np.uint8).reshape(shots, -1)
    ppred = np.asarray(pm.decode_batch(det), np.uint8).reshape(shots, -1)
    q_err = int(np.any(qpred != obs, axis=1).sum())
    p_err = int(np.any(ppred != obs, axis=1).sum())
    return q_err, p_err


def _assert_parity(d, shots, seed, basis="x", noise=0.005):
    q_err, p_err = _logical_errors(d, shots, seed, basis=basis, noise=noise)
    ql, qh = wilson(q_err, shots)
    pl, ph = wilson(p_err, shots)
    overlap = not (qh < pl or ph < ql)
    assert overlap, (
        f"d={d} basis={basis}: Wilson intervals disjoint — "
        f"QECTOR {q_err}/{shots} {ql:.4f}-{qh:.4f} vs "
        f"PyMatching {p_err}/{shots} {pl:.4f}-{ph:.4f}"
    )
    slack = max(4, int(0.3 * p_err))
    assert q_err <= p_err + slack, (
        f"d={d} basis={basis}: QECTOR {q_err} logical errors vs PyMatching {p_err} (+{slack} slack)"
    )


@pytest.mark.parametrize("d", [3, 5, 7, 9, 11])
def test_pymatching_parity_memory_x(d):
    """QECTOR LER overlaps PyMatching Wilson CI on identical memory_x shots."""
    _assert_parity(d, shots=3000, seed=20260622, basis="x", noise=0.005)


def test_pymatching_parity_memory_x_d13():
    """memory_x parity holds at d=13 (kept at 2500 shots, <=4000 cap)."""
    _assert_parity(13, shots=2500, seed=20260623, basis="x", noise=0.005)
