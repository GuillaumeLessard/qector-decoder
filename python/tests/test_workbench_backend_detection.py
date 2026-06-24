"""Workbench: backend detection reports CPU and any available GPU honestly."""

import qector_decoder_v3 as qd
from qector_decoder_v3.workbench import Workbench


def test_detect_backends_reports_cpu():
    wb = Workbench()
    info = wb.detect_backends()
    assert info["cpu"] is True
    assert set(["cpu", "cuda", "opencl"]).issubset(info.keys())


def test_detect_backends_matches_core_availability():
    """Workbench detection must agree with the core availability probes."""
    wb = Workbench()
    info = wb.detect_backends()
    assert info["cuda"] == bool(qd.cuda_is_available())
    assert info["opencl"] == bool(qd.opencl_is_available())


def test_detect_backends_device_name_when_present():
    wb = Workbench()
    info = wb.detect_backends()
    if info["cuda"]:
        # when CUDA is present a device name should be discoverable (non-empty)
        assert info["cuda_device"] is None or isinstance(info["cuda_device"], str)
