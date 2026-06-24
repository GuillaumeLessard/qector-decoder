"""DEM collapse regression fixture: d=15 must stay 132,426 -> 17,862.

Locks the exact circuit-level mechanism / collapsed-edge counts the report
quotes for the distance-15 rotated surface code.
"""
import pytest

stim = pytest.importorskip("stim")

from qector_decoder_v3 import dem  # noqa: E402


def _model(basis):
    circ = stim.Circuit.generated(
        f"surface_code:rotated_memory_{basis}", distance=15, rounds=15,
        after_clifford_depolarization=0.005,
        before_measure_flip_probability=0.005,
        after_reset_flip_probability=0.005)
    sdem = circ.detector_error_model(decompose_errors=True)
    return dem.from_stim(sdem)


def test_d15_memory_x_collapse_counts():
    m = _model("x")
    assert m.num_detectors == 3360
    assert m.num_errors == 132426
    assert m.collapse_to_graph().num_errors == 17862


def test_d15_memory_z_collapse_counts():
    m = _model("z")
    assert m.num_detectors == 3360
    assert m.num_errors == 132418
    assert m.collapse_to_graph().num_errors == 17862
