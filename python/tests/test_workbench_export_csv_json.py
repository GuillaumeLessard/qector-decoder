"""Workbench: CSV and JSON export are reproducible and faithful to the artifact."""

import json
import os

from qector_decoder_v3.workbench import Workbench


def _artifact(wb):
    return wb.run_benchmark(
        {
            "code": "rotated_surface",
            "distances": [3, 5],
            "decoders": ["blossom", "union_find"],
            "trials": 200,
            "warmup": 30,
        }
    )


def test_export_json_roundtrip(tmp_path):
    wb = Workbench()
    art = _artifact(wb)
    out = tmp_path / "art.json"
    wb.export_json(art, str(out))
    loaded = json.loads(out.read_text(encoding="utf-8"))
    assert len(loaded["results"]) == len(art["results"])
    assert loaded["environment"]["git_commit"] == art["environment"]["git_commit"]


def test_export_csv_has_one_row_per_result(tmp_path):
    wb = Workbench()
    art = _artifact(wb)
    out = tmp_path / "art.csv"
    wb.export_csv(art, str(out))
    lines = [l for l in out.read_text(encoding="utf-8").splitlines() if l.strip()]
    assert len(lines) == 1 + len(art["results"])  # header + rows
    header = lines[0].split(",")
    assert "decoder" in header and "syndrome_faithful" in header


def test_export_csv_json_path_with_spaces(tmp_path):
    wb = Workbench()
    art = _artifact(wb)
    d = tmp_path / "out dir spaces"
    j = wb.export_json(art, str(d / "a b.json"))
    c = wb.export_csv(art, str(d / "a b.csv"))
    assert os.path.exists(j) and os.path.exists(c)
