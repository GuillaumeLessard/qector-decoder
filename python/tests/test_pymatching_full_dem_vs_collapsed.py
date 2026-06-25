"""Extended PyMatching parity (section 3): full DEM vs collapsed graph.

The strongest collapse-preservation check. The REFERENCE here is
``pymatching.Matching.from_detector_error_model(sdem)`` built on the *full*
(uncollapsed) Stim DEM. QECTOR's ``pymatching_compat.Matching
.from_detector_error_model(sdem)`` internally *collapses* parallel error
mechanisms (same detector set) into a single graphlike edge. We decode IDENTICAL
seeded Stim shots through both and require their logical error rates to agree:
overlapping Wilson 95% intervals AND QECTOR no worse than PyMatching beyond a
small tie/Poisson slack. Agreement proves the graph collapse preserves logical
decoding accuracy — it does not throw away information PyMatching uses on the
full model.
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


def _full_vs_collapsed(d, shots, seed, basis="x", noise=0.005):
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
    # Reference: PyMatching on the FULL Stim DEM.
    pm = pymatching.Matching.from_detector_error_model(sdem)
    # QECTOR: collapses parallel mechanisms into a graphlike DEM.
    qm = pymatching_compat.Matching.from_detector_error_model(sdem)
    ppred = np.asarray(pm.decode_batch(det), np.uint8).reshape(shots, -1)
    qpred = np.asarray(qm.decode_batch(det), np.uint8).reshape(shots, -1)
    p_err = int(np.any(ppred != obs, axis=1).sum())
    q_err = int(np.any(qpred != obs, axis=1).sum())
    return q_err, p_err


@pytest.mark.parametrize("d", [3, 5, 7])
def test_pymatching_full_dem_vs_collapsed(d):
    """Collapsed-DEM QECTOR LER overlaps full-DEM PyMatching Wilson CI."""
    shots = 4000
    q_err, p_err = _full_vs_collapsed(
        d, shots=shots, seed=20260627 + d, basis="x", noise=0.005
    )
    ql, qh = wilson(q_err, shots)
    pl, ph = wilson(p_err, shots)
    overlap = not (qh < pl or ph < ql)
    assert overlap, (
        f"d={d}: Wilson intervals disjoint — collapsed QECTOR "
        f"{q_err}/{shots} {ql:.4f}-{qh:.4f} vs full-DEM PyMatching "
        f"{p_err}/{shots} {pl:.4f}-{ph:.4f} (collapse lost accuracy)"
    )
    slack = max(4, int(0.3 * p_err))
    assert q_err <= p_err + slack, (
        f"d={d}: collapsed QECTOR {q_err} logical errors vs full-DEM "
        f"PyMatching {p_err} (+{slack} slack) — collapse regression"
    )
