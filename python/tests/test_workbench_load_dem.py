"""Workbench: loading a Stim DEM (file path, text, and stim object)."""

import pytest

from qector_decoder_v3.workbench import Workbench, WorkbenchError

DEM_TEXT = """
error(0.1) D0 L0
error(0.1) D0 D1
error(0.1) D1 D2
error(0.1) D2 L0
"""


def test_load_dem_from_text():
    wb = Workbench()
    desc = wb.load_dem(DEM_TEXT)
    assert desc["kind"] == "dem"
    assert desc["num_detectors"] == 3
    assert desc["num_observables"] == 1
    assert desc["num_errors"] == 4
    assert desc["graphlike"] is True
    assert desc["collapsed_edges"] <= desc["num_errors"]


def test_load_dem_from_file_with_spaces(tmp_path):
    d = tmp_path / "dem dir"
    d.mkdir()
    p = d / "model file.dem"
    p.write_text(DEM_TEXT, encoding="utf-8")
    wb = Workbench()
    desc = wb.load_dem(str(p))
    assert desc["num_errors"] == 4


def test_load_dem_from_stim_object():
    stim = pytest.importorskip("stim")
    circ = stim.Circuit.generated(
        "surface_code:rotated_memory_x", distance=3, rounds=3, after_clifford_depolarization=0.01
    )
    sdem = circ.detector_error_model(decompose_errors=True)
    wb = Workbench()
    desc = wb.load_dem(sdem)
    assert desc["num_detectors"] == sdem.num_detectors
    assert desc["num_errors"] > 0


def test_load_dem_bad_input_raises():
    wb = Workbench()
    with pytest.raises(WorkbenchError):
        wb.load_dem(3.14)
