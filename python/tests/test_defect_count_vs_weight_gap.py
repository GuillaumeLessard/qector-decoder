"""Defect count vs excess matching weight: the gap does NOT grow with density.

A failing candidate-set heuristic shows up as the weight gap increasing with the
number of defects (denser syndromes => more chances to miss the optimal pairing).
This test builds the (defect_count, excess_weight) scatter and asserts there is no
meaningful positive correlation and the gap stays tiny across the whole defect
range — i.e. adaptive-k scales with defect count as intended.
"""

import numpy as np
import pytest

stim = pytest.importorskip("stim")
pymatching = pytest.importorskip("pymatching")

from qector_decoder_v3 import dem, pymatching_compat  # noqa: E402


def _scatter(d, shots, seed=20260622, noise=0.005):
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
    det, _ = circ.compile_detector_sampler(seed=seed).sample(shots=shots, separate_observables=True)
    det = det.astype(np.uint8)
    defects = det.sum(1).astype(float)
    excess = np.empty(shots)
    for i in range(shots):
        cq = np.asarray(qm.decode_to_edges_array(det[i])).astype(bool)
        excess[i] = max(float(w[cq].sum()) - pmc.decode(det[i], return_weight=True)[1], 0.0)
    return defects, excess


@pytest.mark.parametrize("d", [9, 13])
def test_no_correlation_between_defects_and_excess_weight(d):
    defects, excess = _scatter(d, shots=2500)
    assert defects.std() > 0  # there is a real spread of defect counts
    if excess.std() == 0:
        return  # no excess at all -> trivially no correlation
    corr = float(np.corrcoef(defects, excess)[0, 1])
    assert abs(corr) < 0.30, f"d={d}: defect-count vs excess-weight corr={corr:.3f}"


def test_excess_weight_bounded_across_defect_range_d13():
    defects, excess = _scatter(13, shots=2500)
    # even the densest syndromes have a tiny excess
    assert excess.max() <= 6.0
    # the high-defect half is not systematically worse than the low-defect half
    med = np.median(defects)
    lo = excess[defects <= med].mean()
    hi = excess[defects > med].mean()
    assert hi <= lo + 0.2
