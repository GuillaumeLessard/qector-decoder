#!/usr/bin/env python3
"""Comprehensive benchmark for GNN-enhanced HybridDecoder.

Tests multiple configurations and saves results to a JSON file.

Usage:
    python scripts/benchmark_gnn.py --output results.json
"""

import argparse
import json
import sys
import time

sys.path.insert(0, "python")

import numpy as np
from qector_decoder_v3 import HybridDecoder, generate_surface_code_checks


def run_benchmark(distance, error_rate, n_train_samples, n_epochs, n_benchmark, seed=42):
    """Run a single benchmark configuration."""
    print(f"\n--- Configuration: d={distance}, p={error_rate}, train={n_train_samples}, epochs={n_epochs} ---")
    
    result = generate_surface_code_checks(distance)
    check_to_qubits = result[0]
    n_qubits = result[1]
    n_checks = len(check_to_qubits)
    
    decoder = HybridDecoder(check_to_qubits, n_qubits=n_qubits)
    
    # Pre-training benchmark
    rng = np.random.default_rng(seed)
    n_errors_standard = 0
    total_time_standard = 0.0
    
    for i in range(n_benchmark):
        error = (rng.random(n_qubits) < error_rate).astype(np.uint8)
        syndrome = np.zeros(n_checks, dtype=np.uint8)
        for ci, qubits in enumerate(check_to_qubits):
            parity = 0
            for q in qubits:
                parity ^= int(error[q])
            syndrome[ci] = parity
        
        t0 = time.perf_counter()
        correction = decoder.decode_standard(syndrome)
        t1 = time.perf_counter()
        total_time_standard += (t1 - t0) * 1e6
        
        if not np.array_equal(correction, error):
            n_errors_standard += 1
    
    ler_standard = n_errors_standard / n_benchmark
    avg_time_standard = total_time_standard / n_benchmark
    
    # Training
    t0 = time.perf_counter()
    final_loss = decoder.train(n_train_samples, n_epochs, error_rate)
    t1 = time.perf_counter()
    train_time = t1 - t0
    
    # Post-training benchmark
    rng = np.random.default_rng(seed + 1)
    n_errors_hybrid = 0
    total_time_hybrid = 0.0
    
    for i in range(n_benchmark):
        error = (rng.random(n_qubits) < error_rate).astype(np.uint8)
        syndrome = np.zeros(n_checks, dtype=np.uint8)
        for ci, qubits in enumerate(check_to_qubits):
            parity = 0
            for q in qubits:
                parity ^= int(error[q])
            syndrome[ci] = parity
        
        t0 = time.perf_counter()
        correction = decoder.decode_hybrid(syndrome)
        t1 = time.perf_counter()
        total_time_hybrid += (t1 - t0) * 1e6
        
        if not np.array_equal(correction, error):
            n_errors_hybrid += 1
    
    ler_hybrid = n_errors_hybrid / n_benchmark
    avg_time_hybrid = total_time_hybrid / n_benchmark
    improvement = (ler_standard - ler_hybrid) / ler_standard * 100 if ler_standard > 0 else 0
    
    print(f"  Standard LER: {ler_standard:.4f} ({n_errors_standard}/{n_benchmark})")
    print(f"  Hybrid LER:   {ler_hybrid:.4f} ({n_errors_hybrid}/{n_benchmark})")
    print(f"  Improvement:  {improvement:.1f}%")
    print(f"  Speed ratio:  {avg_time_hybrid / avg_time_standard:.1f}x")
    print(f"  Train time:   {train_time:.1f}s")
    print(f"  Final loss:   {final_loss:.6f}")
    
    return {
        "distance": distance,
        "error_rate": error_rate,
        "n_train_samples": n_train_samples,
        "n_epochs": n_epochs,
        "n_benchmark": n_benchmark,
        "ler_standard": ler_standard,
        "ler_hybrid": ler_hybrid,
        "improvement_pct": improvement,
        "time_standard_us": avg_time_standard,
        "time_hybrid_us": avg_time_hybrid,
        "speed_ratio": avg_time_hybrid / avg_time_standard,
        "train_time_s": train_time,
        "final_loss": final_loss,
    }


def main():
    parser = argparse.ArgumentParser(description="Comprehensive GNN benchmark")
    parser.add_argument("--output", type=str, default="gnn_benchmark_results.json",
                        help="Output JSON file for results")
    args = parser.parse_args()
    
    print("=== QECTOR GNN Comprehensive Benchmark ===")
    
    configurations = [
        # (distance, error_rate, n_train_samples, n_epochs, n_benchmark)
        (3, 0.05, 200, 10, 200),
        (3, 0.10, 200, 10, 200),
        (5, 0.03, 500, 15, 200),
        (5, 0.05, 500, 15, 200),
        (5, 0.05, 1000, 20, 200),
        (5, 0.10, 500, 15, 200),
    ]
    
    results = []
    for config in configurations:
        try:
            result = run_benchmark(*config)
            results.append(result)
        except Exception as e:
            print(f"ERROR: {e}")
            results.append({
                "config": config,
                "error": str(e),
            })
    
    with open(args.output, "w") as f:
        json.dump(results, f, indent=2)
    
    print(f"\n=== Results saved to {args.output} ===")
    
    # Summary table
    print("\nSummary:")
    print(f"{'d':>3} {'p':>5} {'train':>6} {'ep':>4} {'LER_std':>8} {'LER_gnn':>8} {'impr%':>6} {'ratio':>6} {'t_train':>8}")
    print("-" * 70)
    for r in results:
        if "error" in r:
            continue
        print(f"{r['distance']:>3} {r['error_rate']:>5.2f} {r['n_train_samples']:>6} {r['n_epochs']:>4} "
              f"{r['ler_standard']:>8.4f} {r['ler_hybrid']:>8.4f} {r['improvement_pct']:>6.1f} "
              f"{r['speed_ratio']:>6.1f}x {r['train_time_s']:>8.1f}s")


if __name__ == "__main__":
    main()
