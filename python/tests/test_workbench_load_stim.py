"""Workbench: loading a Stim circuit (file path, text, and Circuit object)."""
import numpy as np
import pytest

from qector_decoder_v3.workbench import Workbench, WorkbenchError

stim = pytest.importorskip("stim")


def _circuit(d=3):
    return stim.Circuit.generated(
        "surface_code:rotated_memory_x", distance=d, rounds=d,
        after_clifford_depolarization=0.01)


def test_load_stim_from_circuit_object():
    wb = Workbench()
    desc = wb.load_stim(_circuit(3))
    assert desc["kind"] == "stim"
    assert desc["num_detectors"] > 0
    assert desc["num_observables"] == 1
    assert desc["dem_num_errors"] is not None and desc["dem_num_errors"] > 0
    # the loaded problem is stored
    assert wb.loaded is not None and wb.loaded["kind"] == "stim"


def test_load_stim_from_file_with_spaces(tmp_path):
    """Windows-style path with spaces must load correctly."""
    d = tmp_path / "a folder with spaces"
    d.mkdir()
    p = d / "my circuit.stim"
    p.write_text(str(_circuit(3)), encoding="utf-8")
    wb = Workbench()
    desc = wb.load_stim(str(p))
    assert desc["num_detectors"] > 0


def test_load_stim_from_text():
    wb = Workbench()
    desc = wb.load_stim(str(_circuit(3)))
    assert desc["num_qubits"] > 0


def test_load_stim_bad_input_raises():
    wb = Workbench()
    with pytest.raises(WorkbenchError):
        wb.load_stim(12345)
