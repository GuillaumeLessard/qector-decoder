"""The adaptive-k candidate set contains the optimal matching.

We measure the QECTOR-Blossom matching weight against the exact optimum
(PyMatching's ``decode(..., return_weight=True)``) on the IDENTICAL weighted
collapsed graph. If the candidate set always contains the optimal matching, the
weight gap is zero. At small/medium distance QECTOR is exactly optimal; we lock
that and require the median gap to be zero at every tested distance.
"""

import numpy as np
import pytest

stim = pytest.importorskip("stim")
pymatching = pytest.importorskip("pymatching")

from qector_decoder_v3 import dem, pymatching_compat  # noqa: E402

# Tolerance absorbing float summation-order noise (~1e-7); a real sub-optimal
# matching differs by ~1.0 in weight, far above this.
TOL = 1e-3


def _gaps(d, shots, seed=123, noise=0.005):
    circ = stim.Circuit.generated(
        "surface_code:rotated_memory_x",
        distance=d,
        rounds=d,
        after_clifford_depolarization=noise,
        before_measure_flip_probability=noise,
        after_reset_flip_probability=noise,
    )
    sdem = circ.detector_error_model(decompose_errors=True)
    model = dem.from_stim(sdem).collapse_to_graph()
    H = np.asarray(model.check_matrix())
    w = np.asarray(model.weights(), float)
    qm = pymatching_compat.Matching.from_detector_error_model(sdem)
    pmc = pymatching.Matching.from_check_matrix(H, weights=w)
    det, _ = circ.compile_detector_sampler(seed=seed).sample(
        shots=shots, separate_observables=True
    )
    det = det.astype(np.uint8)
    gaps = np.empty(shots)
    for i in range(shots):
        cq = np.asarray(qm.decode_to_edges_array(det[i])).astype(bool)
        wq = float(w[cq].sum())
        _, wc = pmc.decode(det[i], return_weight=True)
        gaps[i] = wq - wc
    return gaps


@pytest.mark.parametrize("d", [3, 5])
def test_exactly_optimal_at_small_distance(d):
    gaps = _gaps(d, shots=2000)
    # candidate set contains the optimal matching on every shot
    assert np.all(gaps <= TOL), f"d={d}: {(gaps > TOL).sum()} sub-optimal shots"


def test_near_exact_at_d7():
    gaps = _gaps(7, shots=2000)
    assert np.median(gaps) <= TOL
    assert (gaps <= TOL).mean() >= 0.99  # >=99% of shots optimal


def test_median_gap_zero_at_d9():
    gaps = _gaps(9, shots=2000)
    assert np.median(gaps) <= TOL
    assert (gaps <= TOL).mean() >= 0.95
