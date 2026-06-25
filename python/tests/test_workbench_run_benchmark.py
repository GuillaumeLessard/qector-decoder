"""Workbench: running real benchmark jobs (no fake data)."""

import pytest

from qector_decoder_v3.workbench import Workbench, WorkbenchError


def test_run_benchmark_generated_code():
    wb = Workbench()
    art = wb.run_benchmark(
        {
            "code": "rotated_surface",
            "distances": [3, 5],
            "decoders": ["blossom", "union_find"],
            "trials": 300,
            "warmup": 50,
        }
    )
    assert len(art["results"]) == 4
    assert "environment" in art and art["environment"].get("git_commit") is not None
    for r in art["results"]:
        # every number is from a real decode: faithful + has latency percentiles
        assert r["syndrome_faithful"] is True
        assert r["latency_us"]["p50"] >= 0.0
        assert r["latency_us"]["p99"] >= r["latency_us"]["p50"]


def test_run_benchmark_real_ler_from_loaded_stim():
    stim = pytest.importorskip("stim")
    pytest.importorskip("pymatching")
    circ = stim.Circuit.generated(
        "surface_code:rotated_memory_x",
        distance=3,
        rounds=3,
        after_clifford_depolarization=0.02,
        before_measure_flip_probability=0.02,
    )
    wb = Workbench()
    wb.load_stim(circ)
    art = wb.run_benchmark({"source": "loaded", "decoders": ["blossom"], "trials": 800})
    ler = art["results"][0]["logical_error_rate"]
    assert ler is not None and 0.0 <= ler <= 1.0


def test_run_benchmark_unknown_decoder_raises():
    wb = Workbench()
    with pytest.raises(WorkbenchError):
        wb.run_benchmark(
            {
                "code": "rotated_surface",
                "distances": [3],
                "decoders": ["totally_fake"],
                "trials": 10,
            }
        )


def test_run_benchmark_unknown_code_raises():
    wb = Workbench()
    with pytest.raises(WorkbenchError):
        wb.run_benchmark(
            {
                "code": "not_a_code",
                "distances": [3],
                "decoders": ["blossom"],
                "trials": 10,
            }
        )


def test_submit_job_completes():
    wb = Workbench()
    jid = wb.submit_job(
        {
            "code": "rotated_surface",
            "distances": [3],
            "decoders": ["blossom"],
            "trials": 200,
        }
    )
    final = wb.wait(jid, timeout=30)
    assert final["status"] == "completed"
    art = wb.job_artifact(jid)
    assert art is not None and len(art["results"]) == 1
    wb.shutdown()
