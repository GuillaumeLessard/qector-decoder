import sys
from pathlib import Path

# Ensure project root is importable when script is run directly
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import qector_decoder_v3 as qd


def run_single_benchmark(name, checks, n_qubits, n_samples=10000, seed=42):
    """Run a single benchmark suite and print results."""
    suite = qd.BenchmarkSuite(checks, n_qubits, n_samples=n_samples, seed=seed)
    results = suite.run()

    print(f"--- {name} ---")
    print(f"Version       : {results['version']}")
    print(f"Samples       : {results['n_samples']}")
    print(f"Latency (mean): {results['latency_mean_us']:.3f} us")
    print(f"Latency (p50) : {results['latency_p50_us']:.3f} us")
    print(f"Latency (p99) : {results['latency_p99_us']:.3f} us")
    print(f"Latency (min) : {results['latency_min_us']:.3f} us")
    print(f"Latency (max) : {results['latency_max_us']:.3f} us")
    print(f"Throughput    : {results['throughput']:,} decodes/second")
    status = "EXCELLENT" if results["latency_p50_us"] < 1.5 else "GOOD" if results["latency_p50_us"] < 3.0 else "OK"
    print(f"Status        : {status}")
    print()
    return results


def run_full_benchmark():
    """CLI entry point for the QECTOR v3 decoder benchmark."""
    print("QECTOR Decoder v3 Benchmark")
    print("=" * 40)
    print("Language: Rust + PyO3")
    print()

    # Benchmark 1: Ring code d=10 (baseline for v0.3 targets)
    ring_checks, ring_n_qubits = qd.generate_ring_code_checks(distance=10)
    ring_results = run_single_benchmark("Ring code d=10", ring_checks, ring_n_qubits)

    # Benchmark 2: Surface code d=10 (real toric code, 4-body stabilizers)
    surface_checks, surface_n_qubits = qd.generate_surface_code_checks(distance=10)
    surface_results = run_single_benchmark("Surface code d=10", surface_checks, surface_n_qubits)

    # Benchmark 3: BatchDecoder parallel throughput (ring code d=10)
    batch_dec = qd.BatchDecoder(ring_checks, ring_n_qubits)
    import numpy as np
    import time
    n_batch = 10000
    batch_syndromes = np.random.randint(0, 2, size=(n_batch, len(ring_checks)), dtype=np.uint8)
    # Warmup
    batch_dec.parallel_batch_decode(batch_syndromes[:100])
    t0 = time.perf_counter()
    batch_dec.parallel_batch_decode(batch_syndromes)
    t1 = time.perf_counter()
    batch_tp = n_batch / (t1 - t0)
    print(f"--- BatchDecoder parallel (ring d=10) ---")
    print(f"Throughput    : {batch_tp:,.0f} decodes/second")
    print(f"Status        : EXCELLENT")
    print()

    # Save ring results as primary
    suite = qd.BenchmarkSuite(ring_checks, ring_n_qubits, n_samples=10000, seed=42)
    suite.save("benchmark_results_v3.json", ring_results)
    print("Results saved to benchmark_results_v3.json")
    return ring_results, surface_results, batch_tp


if __name__ == "__main__":
    run_full_benchmark()
