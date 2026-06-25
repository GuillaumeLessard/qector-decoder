"""DEM collapse: full DEM vs collapsed DEM are logically equivalent.

PyMatching decoding the *full* (uncollapsed) detector error model must give the
same logical error rate as QECTOR decoding the *collapsed* graph on identical
shots — collapsing parallel mechanisms is a speed optimisation that must not
change logical outcomes.
"""

import math

import numpy as np
import pytest

stim = pytest.importorskip("stim")
pymatching = pytest.importorskip("pymatching")

from qector_decoder_v3 import dem, pymatching_compat  # noqa: E402


def _wilson(k, n, z=1.959963985):
    if n == 0:
        return 0.0, 1.0
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    w = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return max(0.0, c - w), min(1.0, c + w)


@pytest.mark.parametrize("d", [3, 5, 7])
def test_full_vs_collapsed_ler_equivalent(d):
    shots = 4000
    circ = stim.Circuit.generated(
        "surface_code:rotated_memory_x",
        distance=d,
        rounds=d,
        after_clifford_depolarization=0.005,
        before_measure_flip_probability=0.005,
        after_reset_flip_probability=0.005,
    )
    sdem = circ.detector_error_model(decompose_errors=True)
    det, obs = circ.compile_detector_sampler(seed=7).sample(shots=shots, separate_observables=True)
    det = det.astype(np.uint8)
    obs = obs.astype(np.uint8)

    # PyMatching on the FULL DEM (PyMatching collapses internally but is the
    # reference for the full model).
    pm_full = pymatching.Matching.from_detector_error_model(sdem)
    pm_pred = np.asarray(pm_full.decode_batch(det), np.uint8).reshape(shots, -1)
    pm_err = int(np.any(pm_pred != obs, axis=1).sum())

    # QECTOR on the explicitly COLLAPSED graph.
    qm = pymatching_compat.Matching.from_detector_error_model(sdem)
    q_pred = np.asarray(qm.decode_batch(det), np.uint8).reshape(shots, -1)
    q_err = int(np.any(q_pred != obs, axis=1).sum())

    lo_p, hi_p = _wilson(pm_err, shots)
    lo_q, hi_q = _wilson(q_err, shots)
    # the two Wilson 95% intervals must overlap (statistically equivalent LER)
    assert lo_q <= hi_p and lo_p <= hi_q, (
        f"d={d}: collapsed LER {q_err}/{shots} vs full LER {pm_err}/{shots} disjoint CIs"
    )


def test_collapsed_graph_decode_is_faithful():
    circ = stim.Circuit.generated(
        "surface_code:rotated_memory_x", distance=5, rounds=5, after_clifford_depolarization=0.006
    )
    sdem = circ.detector_error_model(decompose_errors=True)
    model = dem.from_stim(sdem).collapse_to_graph()
    H = model.check_matrix()
    dec = model.make_decoder("blossom")
    det, _ = circ.compile_detector_sampler(seed=3).sample(shots=1000, separate_observables=True)
    det = det.astype(np.uint8)
    for i in range(len(det)):
        c = np.asarray(dec.decode(det[i]), np.uint8)
        assert np.array_equal((H @ c) & 1, det[i])
