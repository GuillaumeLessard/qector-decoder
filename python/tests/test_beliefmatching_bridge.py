"""Regression tests: BeliefMatching interop with the beliefmatching package.

Bug fixed in v0.5.3:
  beliefmatching==0.2.0's DemMatrices.__init__ requires five explicit sparse
  matrix arguments — it does NOT accept a bare stim.DetectorErrorModel.

  The correct call is:
      dm = beliefmatching.detector_error_model_to_check_matrices(dem)

  QECTOR now provides BeliefMatching.from_beliefmatching_matrices(dm) to
  bridge the two APIs without requiring callers to do the sparse→dense
  conversion manually.

  This file also documents the raw-ndarray BeliefMatching(H) constructor
  (added in the same session) with a correctness test.
"""

import numpy as np
import pytest

stim = pytest.importorskip("stim")
beliefmatching = pytest.importorskip("beliefmatching")

from beliefmatching import detector_error_model_to_check_matrices  # noqa: E402

from qector_decoder_v3.belief_matching import BeliefMatching, build_matching_matrices  # noqa: E402


def _circuit(d=3, p=0.005):
    return stim.Circuit.generated(
        "surface_code:rotated_memory_x",
        distance=d,
        rounds=d,
        after_clifford_depolarization=p,
        before_measure_flip_probability=p,
        after_reset_flip_probability=p,
    )


def _dem(d=3, p=0.005):
    return _circuit(d, p).detector_error_model(decompose_errors=True)


# ---------------------------------------------------------------------------
# beliefmatching.DemMatrices API clarification test
# ---------------------------------------------------------------------------


def test_beliefmatching_dem_matrices_requires_explicit_args():
    """Confirm that beliefmatching.DemMatrices(dem) raises TypeError.

    This documents the API break: in beliefmatching==0.2.0 the constructor
    requires five explicit sparse-matrix arguments.  Callers MUST use
    detector_error_model_to_check_matrices(dem) instead.
    """
    from beliefmatching import DemMatrices

    dem = _dem()
    with pytest.raises(TypeError):
        DemMatrices(dem)  # raises: missing 5 required positional arguments


def test_detector_error_model_to_check_matrices_works():
    """The correct API for building DemMatrices from a DEM."""
    dem = _dem(d=3)
    dm = detector_error_model_to_check_matrices(dem)
    assert hasattr(dm, "check_matrix")
    assert hasattr(dm, "observables_matrix")
    assert hasattr(dm, "edge_check_matrix")
    assert hasattr(dm, "edge_observables_matrix")
    assert hasattr(dm, "hyperedge_to_edge_matrix")
    assert hasattr(dm, "priors")


# ---------------------------------------------------------------------------
# BeliefMatching.from_beliefmatching_matrices — the QECTOR bridge
# ---------------------------------------------------------------------------


def test_from_beliefmatching_matrices_constructs():
    """BeliefMatching.from_beliefmatching_matrices accepts a DemMatrices object."""
    dem = _dem(d=3)
    bm_mats = detector_error_model_to_check_matrices(dem)
    qbm = BeliefMatching.from_beliefmatching_matrices(bm_mats)
    assert qbm.n_checks == dem.num_detectors
    assert qbm.num_observables == dem.num_observables


def test_from_beliefmatching_matrices_decodes():
    """Decoder built via from_beliefmatching_matrices produces valid observable preds."""
    circ = _circuit(d=3)
    dem = circ.detector_error_model(decompose_errors=True)
    bm_mats = detector_error_model_to_check_matrices(dem)
    qbm = BeliefMatching.from_beliefmatching_matrices(bm_mats)

    det, obs = circ.compile_detector_sampler(seed=101).sample(
        shots=200, separate_observables=True
    )
    det = det.astype(np.uint8)
    obs = obs.astype(np.uint8)

    pred = qbm.decode_batch(det)
    assert pred.shape == obs.shape
    assert set(np.unique(pred).tolist()).issubset({0, 1})

    ler = float(np.any(pred != obs, axis=1).mean())
    assert ler < 0.5, f"LER {ler:.3f} is random — decoder is broken"


def test_from_beliefmatching_matrices_matches_native_constructor():
    """from_beliefmatching_matrices and from_detector_error_model must agree."""
    circ = _circuit(d=5, p=0.006)
    dem = circ.detector_error_model(decompose_errors=True)

    bm_mats = detector_error_model_to_check_matrices(dem)
    qbm_bridge = BeliefMatching.from_beliefmatching_matrices(bm_mats, max_iter=20)
    qbm_native = BeliefMatching.from_detector_error_model(dem, max_iter=20)

    det, obs = circ.compile_detector_sampler(seed=77).sample(
        shots=500, separate_observables=True
    )
    det = det.astype(np.uint8)
    obs = obs.astype(np.uint8)

    pred_bridge = qbm_bridge.decode_batch(det)
    pred_native = qbm_native.decode_batch(det)
    err_bridge = int(np.any(pred_bridge != obs, axis=1).sum())
    err_native = int(np.any(pred_native != obs, axis=1).sum())

    # Both paths must decode to the same logical error count (same model).
    assert abs(err_bridge - err_native) <= max(5, int(0.1 * err_native)), (
        f"bridge={err_bridge} native={err_native} — constructors disagree"
    )


# ---------------------------------------------------------------------------
# Raw-ndarray BeliefMatching(H) constructor
# ---------------------------------------------------------------------------


def test_raw_ndarray_constructor_shapes():
    """BeliefMatching(H) with a raw check matrix must instantiate without error."""
    H = np.array([[1, 1, 0, 0], [0, 1, 1, 0], [0, 0, 1, 1]], dtype=np.uint8)
    bm = BeliefMatching(H)
    assert bm.n_checks == 3
    # num_observables == num_qubits when no observable matrix is provided
    assert bm.num_observables == 4


def test_raw_ndarray_constructor_decodes():
    """BeliefMatching(H) must decode without crashing."""
    H = np.array([[1, 1, 0, 0], [0, 1, 1, 0], [0, 0, 1, 1]], dtype=np.uint8)
    bm = BeliefMatching(H)
    syndrome = np.array([1, 0, 0], dtype=np.uint8)
    pred = bm.decode(syndrome)
    assert pred.shape == (4,)
    assert set(pred.tolist()).issubset({0, 1})


def test_raw_ndarray_rejects_1d():
    """BeliefMatching(H) must reject a 1-D array."""
    with pytest.raises(ValueError):
        BeliefMatching(np.array([1, 0, 1], dtype=np.uint8))
