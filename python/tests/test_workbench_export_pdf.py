"""Workbench: PDF export with charts built from a real artifact."""

import os

import pytest

from qector_decoder_v3.workbench import Workbench

pytest.importorskip("matplotlib")


def _artifact(wb):
    return wb.run_benchmark(
        {
            "code": "rotated_surface",
            "distances": [3, 5, 7],
            "decoders": ["blossom"],
            "trials": 200,
            "warmup": 30,
        }
    )


def test_export_pdf_produces_nonempty_file(tmp_path):
    wb = Workbench()
    art = _artifact(wb)
    out = tmp_path / "report.pdf"
    path = wb.export_pdf(art, str(out))
    assert os.path.exists(path)
    # a real multi-page PDF is comfortably > 1 KB
    assert os.path.getsize(path) > 1000
    with open(path, "rb") as fh:
        assert fh.read(5) == b"%PDF-"


def test_export_pdf_path_with_spaces(tmp_path):
    wb = Workbench()
    art = _artifact(wb)
    d = tmp_path / "export dir with spaces"
    out = d / "my report.pdf"
    path = wb.export_pdf(art, str(out))
    assert os.path.exists(path)


def test_export_pdf_is_reproducible_from_stored_artifact(tmp_path):
    """Re-exporting the SAME artifact twice yields a valid PDF each time."""
    wb = Workbench()
    art = _artifact(wb)
    a = wb.export_pdf(art, str(tmp_path / "a.pdf"))
    b = wb.export_pdf(art, str(tmp_path / "b.pdf"))
    assert os.path.getsize(a) > 1000 and os.path.getsize(b) > 1000
