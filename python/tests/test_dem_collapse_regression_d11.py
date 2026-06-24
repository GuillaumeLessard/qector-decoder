"""DEM collapse regression fixture: d=11 must stay 50,484 -> 6,718.

Locks the exact circuit-level mechanism / collapsed-edge counts the report
quotes for the distance-11 rotated surface code, so any future change to the DEM
parser or collapse rule that alters the graph size is caught immediately.
"""
import pytest

stim = pytest.importorskip("stim")

from qector_decoder_v3 import dem  # noqa: E402


def _model(basis):
    circ = stim.Circuit.generated(
        f"surface_code:rotated_memory_{basis}", distance=11, rounds=11,
        after_clifford_depolarization=0.005,
        before_measure_flip_probability=0.005,
        after_reset_flip_probability=0.005)
    sdem = circ.detector_error_model(decompose_errors=True)
    return dem.from_stim(sdem)


def test_d11_memory_x_collapse_counts():
    m = _model("x")
    assert m.num_detectors == 1320
    assert m.num_errors == 50484
    assert m.collapse_to_graph().num_errors == 6718


def test_d11_memory_z_collapse_counts():
    m = _model("z")
    assert m.num_detectors == 1320
    assert m.num_errors == 50476
    assert m.collapse_to_graph().num_errors == 6718
