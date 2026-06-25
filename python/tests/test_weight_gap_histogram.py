"""Weight-gap histogram: QECTOR-Blossom vs exact MWPM on the collapsed graph.

Builds the per-shot distribution of ``weight(QECTOR) - weight(PyMatching optimal)``
on the identical weighted collapsed graph and asserts the gap distribution is
concentrated at zero: median ~0, small mean, and the overwhelming majority of
shots exactly optimal. This is the histogram the report shows; it locks the
"never materially heavier" property.

Weights are float sums whose summation order differs between QECTOR and
PyMatching, so a shot is "optimal" when the gap is within TOL (1e-3) — far above
that floating-point noise (~1e-7) and far below a real sub-optimality (~1.0).
"""

import numpy as np
import pytest

stim = pytest.importorskip("stim")
pymatching = pytest.importorskip("pymatching")

from qector_decoder_v3 import dem, pymatching_compat  # noqa: E402

TOL = 1e-3


def _gaps(d, shots, seed=20260622, noise=0.005):
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
        gaps[i] = float(w[cq].sum()) - pmc.decode(det[i], return_weight=True)[1]
    return gaps


@pytest.mark.parametrize("d", [5, 9, 13])
def test_weight_gap_concentrated_at_zero(d):
    gaps = _gaps(d, shots=2500)
    assert np.median(gaps) <= TOL, f"d={d}: median weight gap {np.median(gaps):.2e}"
    assert gaps.mean() < 0.1, f"d={d}: mean weight gap {gaps.mean():.4f} too large"
    # QECTOR can't beat the exact optimum beyond float noise
    assert gaps.min() >= -TOL
    optimal_frac = float((gaps <= TOL).mean())
    assert optimal_frac >= 0.70, f"d={d}: only {optimal_frac:.2%} shots optimal"
    # histogram concentrated in the zero bin
    edges = [-TOL, TOL, 1, 2, 4, 8, 16, 1e9]
    hist, _ = np.histogram(gaps, bins=edges)
    assert hist[0] >= 0.70 * len(gaps)


def test_histogram_p99_small_at_d13():
    gaps = _gaps(13, shots=2500)
    assert np.percentile(gaps, 99) <= 2.0  # tiny tail even at d=13
