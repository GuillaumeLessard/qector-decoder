"""d=15: QECTOR-Blossom has no logical-error gap vs PyMatching.

The report claims exact-MWPM LER parity through d=15 after the adaptive-k fix.
This test locks that specific claim: on identical seeded d=15 shots decoded over
the identical collapsed graph, QECTOR's logical error count overlaps PyMatching's
Wilson 95% interval.
"""

import math

import numpy as np
import pytest

stim = pytest.importorskip("stim")
pymatching = pytest.importorskip("pymatching")

from qector_decoder_v3 import pymatching_compat  # noqa: E402


def _wilson(k, n, z=1.959963985):
    if n == 0:
        return 0.0, 1.0
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    w = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return max(0.0, c - w), min(1.0, c + w)


def test_d15_logical_error_parity():
    d, shots = 15, 4000
    circ = stim.Circuit.generated(
        "surface_code:rotated_memory_x",
        distance=d,
        rounds=d,
        after_clifford_depolarization=0.005,
        before_measure_flip_probability=0.005,
        after_reset_flip_probability=0.005,
    )
    sdem = circ.detector_error_model(decompose_errors=True)
    det, obs = circ.compile_detector_sampler(seed=20260622).sample(shots=shots, separate_observables=True)
    det = det.astype(np.uint8)
    obs = obs.astype(np.uint8)

    qm = pymatching_compat.Matching.from_detector_error_model(sdem)
    pm = pymatching.Matching.from_detector_error_model(sdem)
    q_err = int(np.any(np.asarray(qm.decode_batch(det), np.uint8).reshape(shots, -1) != obs, axis=1).sum())
    p_err = int(np.any(np.asarray(pm.decode_batch(det), np.uint8).reshape(shots, -1) != obs, axis=1).sum())

    lo_p, hi_p = _wilson(p_err, shots)
    lo_q, hi_q = _wilson(q_err, shots)
    # Wilson intervals overlap -> statistically no gap
    assert lo_q <= hi_p and lo_p <= hi_q, (
        f"d=15 LER gap: QECTOR {q_err}/{shots} CI[{lo_q:.4f},{hi_q:.4f}] vs "
        f"PyMatching {p_err}/{shots} CI[{lo_p:.4f},{hi_p:.4f}]"
    )
    # and QECTOR not materially worse in raw count
    assert q_err <= p_err + max(4, int(0.30 * p_err))
