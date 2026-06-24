"""Tests for qector_decoder_v3.benchmarking — reproducible harness."""

import json

import pytest

from qector_decoder_v3 import codes
from qector_decoder_v3 import benchmarking as bm


def test_capture_environment_has_required_fields():
    env = bm.capture_environment()
    for key in (
        "python_version",
        "platform",
        "cpu_count_logical",
        "numpy_version",
        "cuda_available",
        "opencl_available",
        "qector_decoder_v3_version",
    ):
        assert key in env


def test_percentiles_and_summary():
    samples = [i * 1e-6 for i in range(1, 101)]  # 1..100 us
    summ = bm.summarize(samples)
    assert summ["n"] == 100
    assert summ["min"] <= summ["p50"] <= summ["p99"] <= summ["max"]
    assert summ["ci95_low"] <= summ["mean"] <= summ["ci95_high"]
    pcts = bm.percentiles([1, 2, 3, 4, 5])
    assert pcts["p50"] == 3.0


def test_time_iterations_runs_warmup_and_trials():
    counter = {"n": 0}

    def fn():
        counter["n"] += 1

    samples = bm.time_iterations(fn, n_trials=10, warmup=3)
    assert len(samples) == 10
    assert counter["n"] == 13


@pytest.mark.parametrize("kind", ["union_find", "blossom", "sparse_blossom", "cpu_batch"])
def test_benchmark_decoder_report(kind):
    code = codes.rotated_surface_code(5)
    r = bm.benchmark_decoder(kind, code, n_trials=200, warmup=20, seed=1)
    assert r["decoder"] == kind
    assert r["syndrome_faithful"] is True
    assert r["n_trials"] == 200
    lat = r["latency_us"]
    assert lat["min"] <= lat["p50"] <= lat["p99"] <= lat["max"]
    assert r["cold_path_us"]["median"] >= 0
    assert r["throughput_decodes_per_s"] > 0


def test_report_json_and_csv(tmp_path):
    code = codes.repetition_code(11)
    results = [
        bm.benchmark_decoder("blossom", code, n_trials=100, warmup=10, seed=2),
        bm.benchmark_decoder("union_find", code, n_trials=100, warmup=10, seed=2),
    ]
    report = bm.BenchmarkReport(results)
    blob = report.to_json()
    parsed = json.loads(blob)
    assert "environment" in parsed and "results" in parsed
    assert len(parsed["results"]) == 2

    csv = report.to_csv()
    assert csv.splitlines()[0].startswith("decoder,code,")
    assert len(csv.splitlines()) == 3  # header + 2 rows

    jpath = tmp_path / "bench.json"
    cpath = tmp_path / "bench.csv"
    report.save(str(jpath), str(cpath))
    assert jpath.exists() and cpath.exists()


def test_memory_measurement_optional():
    code = codes.repetition_code(9)
    r = bm.benchmark_decoder("blossom", code, n_trials=50, warmup=5, measure_memory=True)
    assert r["peak_python_alloc_kib"] is not None
    r2 = bm.benchmark_decoder("blossom", code, n_trials=50, warmup=5, measure_memory=False)
    assert r2["peak_python_alloc_kib"] is None
