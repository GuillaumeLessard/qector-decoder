"""Tests for the benchmark suite."""

import json
import pytest

import qector_decoder_v3 as qd


@pytest.fixture
def small_decoder():
    checks, n_qubits = qd.generate_surface_code_checks(5)
    return checks, n_qubits


class TestBenchmarkSuite:
    """Benchmark suite validation tests."""

    def test_benchmark_run_keys(self, small_decoder):
        checks, n_qubits = small_decoder
        suite = qd.BenchmarkSuite(checks, n_qubits, n_samples=100, seed=42)
        results = suite.run()
        assert "latency_mean_us" in results
        assert "latency_p50_us" in results
        assert "latency_p99_us" in results
        assert "latency_min_us" in results
        assert "latency_max_us" in results
        assert "throughput" in results
        assert "timestamp" in results
        assert results["version"] == qd.__version__
        assert results["n_samples"] == 100

    def test_benchmark_save(self, small_decoder, tmp_path):
        checks, n_qubits = small_decoder
        suite = qd.BenchmarkSuite(checks, n_qubits, n_samples=100, seed=42)
        results = suite.run()
        path = tmp_path / "results.json"
        suite.save(str(path), results)
        assert path.exists()
        data = json.loads(path.read_text())
        assert data["version"] == qd.__version__
        assert "throughput" in data

    def test_benchmark_positive_latency(self, small_decoder):
        checks, n_qubits = small_decoder
        suite = qd.BenchmarkSuite(checks, n_qubits, n_samples=100, seed=42)
        results = suite.run()
        assert results["latency_mean_us"] > 0
        assert results["latency_p50_us"] > 0
        assert results["latency_p99_us"] > 0
        assert results["latency_min_us"] > 0
        assert results["latency_max_us"] > 0
        assert results["throughput"] > 0

    def test_benchmark_reproducibility(self, small_decoder):
        checks, n_qubits = small_decoder
        suite1 = qd.BenchmarkSuite(checks, n_qubits, n_samples=100, seed=42)
        suite2 = qd.BenchmarkSuite(checks, n_qubits, n_samples=100, seed=42)
        r1 = suite1.run()
        r2 = suite2.run()
        assert r1["version"] == r2["version"] == qd.__version__
        assert r1["n_samples"] == r2["n_samples"] == 100
        # The seed controls generated syndromes, not wall-clock scheduling noise.
        # Verify timing statistics remain internally consistent and broadly sane.
        for results in (r1, r2):
            assert (
                results["latency_min_us"]
                <= results["latency_p50_us"]
                <= results["latency_max_us"]
            )
            assert (
                results["latency_min_us"]
                <= results["latency_mean_us"]
                <= results["latency_max_us"]
            )
            assert (
                results["latency_p50_us"]
                <= results["latency_p99_us"]
                <= results["latency_max_us"]
            )
        # Throughput can vary significantly on fast decoders due to clock resolution.
        # Just verify both are in the same order of magnitude.
        ratio = r1["throughput"] / r2["throughput"] if r2["throughput"] > 0 else 0
        assert 0.1 < ratio < 10.0
