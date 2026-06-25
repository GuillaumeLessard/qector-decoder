"""Regression tests: stim_circuit_to_check_matrix accepts a stim.Circuit.

Bug fixed in v0.5.3:
  stim_circuit_to_check_matrix was a bare alias for
  from_stim_detector_error_model.  Both stim.Circuit and
  stim.DetectorErrorModel expose num_detectors, so the type-guard passed.
  circuit.flattened() also exists (returns a flattened Circuit, not a DEM),
  so from_stim() silently called parse_dem(str(circuit)) — which finds no
  error(...) lines — and returned num_errors == 0.

  The fix distinguishes the two types via stim.Circuit.detector_error_model(),
  which stim.DetectorErrorModel does not expose.
"""

import numpy as np
import pytest

stim = pytest.importorskip("stim")

from qector_decoder_v3.stim_compat import (
    stim_circuit_to_check_matrix,
    from_stim_detector_error_model,
)
from qector_decoder_v3.dem import from_stim


# ---------------------------------------------------------------------------
# Core regression: circuit path returns the right num_errors
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("d", [3, 5, 7])
def test_circuit_alias_num_errors_is_nonzero(d):
    """stim_circuit_to_check_matrix(circuit) must return num_errors > 0."""
    circ = stim.Circuit.generated(
        "surface_code:rotated_memory_x",
        distance=d,
        rounds=d,
        after_clifford_depolarization=0.005,
    )
    c2q, n_errors = stim_circuit_to_check_matrix(circ)
    assert n_errors > 0, (
        f"stim_circuit_to_check_matrix(circuit) returned num_errors={n_errors} "
        f"for d={d}; was silently returning 0 in v0.5.2"
    )


def test_circuit_alias_matches_dem_path():
    """Circuit alias and DEM path must produce identical (check_to_qubits, n_errors)."""
    circ = stim.Circuit.generated(
        "surface_code:rotated_memory_x",
        distance=3,
        rounds=3,
        after_clifford_depolarization=0.001,
    )
    dem = circ.detector_error_model(decompose_errors=True)

    c2q_circ, nq_circ = stim_circuit_to_check_matrix(circ)
    c2q_dem, nq_dem = from_stim_detector_error_model(dem)

    assert nq_circ == nq_dem
    assert c2q_circ == c2q_dem


def test_dem_path_still_works_through_alias():
    """Passing a DetectorErrorModel to stim_circuit_to_check_matrix still works."""
    dem = stim.Circuit.generated(
        "surface_code:rotated_memory_x",
        distance=3,
        rounds=3,
        after_clifford_depolarization=0.005,
    ).detector_error_model(decompose_errors=True)

    c2q, n_errors = stim_circuit_to_check_matrix(dem)
    assert n_errors > 0
    assert len(c2q) == dem.num_detectors


def test_dem_from_stim_circuit_direct():
    """dem.from_stim(circuit) must also produce a valid DemModel now."""
    circ = stim.Circuit.generated(
        "surface_code:rotated_memory_x",
        distance=3,
        rounds=3,
        after_clifford_depolarization=0.005,
    )
    model = from_stim(circ)
    assert model.num_errors > 0
    assert model.num_detectors == circ.num_detectors
    assert model.num_observables == circ.num_observables


def test_circuit_num_errors_matches_dem_num_errors():
    """The check count from a circuit equals the one from its DEM."""
    circ = stim.Circuit.generated(
        "surface_code:rotated_memory_z",
        distance=5,
        rounds=5,
        after_clifford_depolarization=0.003,
    )
    dem = circ.detector_error_model(decompose_errors=True)
    _, n_circ = stim_circuit_to_check_matrix(circ)
    _, n_dem = from_stim_detector_error_model(dem)
    assert n_circ == n_dem


def test_circuit_decoder_is_syndrome_faithful():
    """Decoders built from a circuit-derived check matrix produce valid corrections."""
    import qector_decoder_v3 as qd

    circ = stim.Circuit.generated(
        "surface_code:rotated_memory_x",
        distance=3,
        rounds=3,
        after_clifford_depolarization=0.005,
    )
    c2q, n_errors = stim_circuit_to_check_matrix(circ)
    dec = qd.UnionFindDecoder(c2q, n_errors)

    # Sample real syndromes from the circuit and confirm corrections are valid.
    det, _ = circ.compile_detector_sampler(seed=42).sample(
        shots=50, separate_observables=True
    )
    det = det.astype(np.uint8)
    from qector_decoder_v3.dem import from_stim

    model = from_stim(circ)
    H = model.check_matrix()
    for i in range(det.shape[0]):
        s = det[i]
        corr = np.asarray(dec.decode(s), dtype=np.uint8)
        assert np.array_equal((H @ corr) & 1, s), f"shot {i} not syndrome-faithful"
