"""S1 (K3dev.md wave 3): DEM parse integrity + observable-map change.

``from_stim`` must produce a DemModel identical to the explicit text path,
and the batched ``corr @ faults.T`` observable map must yield predictions
identical to the per-shot path.  (A native Stim-instruction parse path was
benchmarked at 0.60x the text path and removed; see K3dev.md S1.)
"""
from __future__ import annotations

import numpy as np
import pytest

stim = pytest.importorskip("stim")

from qector_decoder_v3.dem import from_stim, parse_dem


def _models(d: int, basis: str, decompose: bool):
    circ = stim.Circuit.generated(
        f"surface_code:rotated_memory_{basis}",
        distance=d,
        rounds=d,
        after_clifford_depolarization=0.005,
        before_measure_flip_probability=0.005,
        after_reset_flip_probability=0.005,
    )
    dem = circ.detector_error_model(decompose_errors=decompose)
    via_from_stim = from_stim(dem)
    text = parse_dem(str(dem.flattened()))
    return dem, via_from_stim, text


@pytest.mark.parametrize("basis", ["x", "z"])
@pytest.mark.parametrize("decompose", [True, False])
def test_from_stim_matches_explicit_text_path(basis, decompose):
    dem, fast, text = _models(3, basis, decompose)
    assert len(fast.errors) == len(text.errors)
    for fe, te in zip(fast.errors, text.errors):
        assert fe.probability == te.probability
        assert fe.detectors == te.detectors
        assert fe.observables == te.observables
    assert fast.num_detectors == max(text.num_detectors, int(dem.num_detectors))
    assert fast.num_observables == max(text.num_observables, int(dem.num_observables))
    assert fast.detector_coords == text.detector_coords


def test_from_stim_matrices_identical_to_textpath():
    _dem, fast, text = _models(3, "x", True)
    f = fast.collapse_to_graph()
    t = text.collapse_to_graph()
    np.testing.assert_array_equal(f.check_matrix(), t.check_matrix())
    np.testing.assert_allclose(f.weights(), t.weights(), rtol=0, atol=0)
    np.testing.assert_array_equal(f.observables_matrix(), t.observables_matrix())


def test_observable_map_batched_matches_single():
    """The batched ``corr @ faults.T`` path must equal the single-shot map."""
    from qector_decoder_v3.pymatching_compat import Matching

    circ = stim.Circuit.generated(
        "surface_code:rotated_memory_x", distance=3, rounds=3,
        after_clifford_depolarization=0.005,
    )
    dem = circ.detector_error_model(decompose_errors=True)
    m = Matching.from_detector_error_model(dem)
    det, _obs = circ.compile_detector_sampler(seed=5).sample(shots=64, separate_observables=True)
    det = det.astype(np.uint8)
    batched = np.asarray(m.decode_batch(det))
    singles = np.stack([np.asarray(m.decode(row)) for row in det])
    np.testing.assert_array_equal(batched, singles)
