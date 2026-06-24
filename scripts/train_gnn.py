#!/usr/bin/env python3
"""Training script for GNN-enhanced HybridDecoder.

Pipeline:
  1. Create a surface code with given distance
  2. Create a HybridDecoder (GNN + SparseBlossom)
  3. Benchmark LER before training (standard SparseBlossom)
  4. Train the GNN using Blossom teacher model
  5. Benchmark LER after training (hybrid GNN + SparseBlossom)
  6. Report improvement

Usage:
    python scripts/train_gnn.py --distance 5 --n-samples 5000 --n-epochs 50
"""

import argparse
import sys
import time

sys.path.insert(0, "python")

import numpy as np
from qector_decoder_v3 import (
    HybridDecoder,
    generate_surface_code_checks,
)


def benchmark_ler(decoder, n_samples, error_rate, seed=42):
    """Benchmark logical error rate on random syndromes.

    Returns dict with n_errors, n_total, ler, and avg_time_us.
    """
    rng = np.random.default_rng(seed)
    n_qubits = decoder.n_qubits
    n_checks = decoder.n_checks

    n_errors = 0
    total_time_us = 0.0

    for i in range(n_samples):
        # Generate random error vector
        error = (rng.random(n_qubits) < error_rate).astype(np.uint8)

        # Compute syndrome
        syndrome = np.zeros(n_checks, dtype=np.uint8)
        # We need check_to_qubits to compute syndrome, but it's not exposed on decoder
        # Instead, we use the decoder's decode methods and check validity indirectly
        # For LER benchmarking, we compare correction to actual error

        t0 = time.perf_counter()
        correction = decoder.decode_hybrid(syndrome)
        t1 = time.perf_counter()
        total_time_us += (t1 - t0) * 1e6

        # Check if correction matches error (simplified LER metric)
        # In practice, we need the stabilizer checks to compute syndrome properly
        # This is a placeholder - we'll use the generate_surface_code_checks function

    # For a proper LER benchmark, we need the check_to_qubits structure
    # Let's restructure to use the generator directly
    return {
        "n_errors": n_errors,
        "n_total": n_samples,
        "ler": n_errors / n_samples if n_samples > 0 else 0.0,
        "avg_time_us": total_time_us / n_samples if n_samples > 0 else 0.0,
    }


def main():
    parser = argparse.ArgumentParser(description="Train GNN for HybridDecoder")
    parser.add_argument("--distance", type=int, default=5, help="Surface code distance (3, 5, 7, 9)")
    parser.add_argument("--n-train-samples", type=int, default=500, help="Training samples")
    parser.add_argument("--n-epochs", type=int, default=15, help="Training epochs")
    parser.add_argument("--error-rate", type=float, default=0.05, help="Physical error rate")
    parser.add_argument("--n-benchmark", type=int, default=200, help="Benchmark samples")
    parser.add_argument("--gnn-hidden-size", type=int, default=16, help="GNN hidden size (smaller = faster)")
    parser.add_argument("--gnn-n-layers", type=int, default=2, help="GNN number of layers (fewer = faster)")
    parser.add_argument("--use-batch", action="store_true", help="Use batch decode for benchmark")
    args = parser.parse_args()

    print(f"=== QECTOR GNN Training Pipeline ===")
    print(f"Surface code distance: {args.distance}")
    print(f"GNN hidden size: {args.gnn_hidden_size}, layers: {args.gnn_n_layers}")
    print(f"Training samples: {args.n_train_samples}")
    print(f"Training epochs: {args.n_epochs}")
    print(f"Physical error rate: {args.error_rate}")
    print(f"Benchmark samples: {args.n_benchmark}")
    print()

    # 1. Generate surface code checks
    result = generate_surface_code_checks(args.distance)
    check_to_qubits = result[0]
    n_qubits = result[1]
    n_checks = len(check_to_qubits)
    print(f"Generated surface code: {n_qubits} qubits, {n_checks} checks")

    # 2. Create HybridDecoder with small GNN for speed
    decoder = HybridDecoder(
        check_to_qubits, n_qubits=n_qubits,
        gnn_hidden_size=args.gnn_hidden_size, gnn_n_layers=args.gnn_n_layers
    )
    print(f"HybridDecoder created: {decoder.n_qubits} qubits, {decoder.n_checks} checks")
    print()

    # 3. Benchmark before training (standard SparseBlossom)
    print("Benchmarking standard SparseBlossom (before training)...")
    rng = np.random.default_rng(42)
    n_errors_standard = 0
    total_time_standard = 0.0

    for i in range(args.n_benchmark):
        error = (rng.random(n_qubits) < args.error_rate).astype(np.uint8)
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

        # Logical error if correction differs from error (simplified)
        if not np.array_equal(correction, error):
            n_errors_standard += 1

    ler_standard = n_errors_standard / args.n_benchmark
    avg_time_standard = total_time_standard / args.n_benchmark
    print(f"  LER (standard): {ler_standard:.4f} ({n_errors_standard}/{args.n_benchmark})")
    print(f"  Avg time: {avg_time_standard:.1f} µs")
    print()

    # 4. Train GNN
    print(f"Training GNN with {args.n_train_samples} samples, {args.n_epochs} epochs...")
    t0 = time.perf_counter()
    final_loss = decoder.train(args.n_train_samples, args.n_epochs, args.error_rate)
    t1 = time.perf_counter()
    train_time = t1 - t0
    print(f"  Final loss: {final_loss:.6f}")
    print(f"  Training time: {train_time:.1f}s")
    print()

    # 5. Benchmark after training (hybrid)
    print("Benchmarking HybridDecoder (after training)...")
    rng = np.random.default_rng(43)  # Different seed
    n_errors_hybrid = 0
    total_time_hybrid = 0.0

    for i in range(args.n_benchmark):
        error = (rng.random(n_qubits) < args.error_rate).astype(np.uint8)
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

    ler_hybrid = n_errors_hybrid / args.n_benchmark
    avg_time_hybrid = total_time_hybrid / args.n_benchmark
    print(f"  LER (hybrid): {ler_hybrid:.4f} ({n_errors_hybrid}/{args.n_benchmark})")
    print(f"  Avg time: {avg_time_hybrid:.1f} µs")
    print()

    # 6. Report improvement
    improvement = (ler_standard - ler_hybrid) / ler_standard * 100 if ler_standard > 0 else 0
    print("=== Results ===")
    print(f"Standard SparseBlossom LER: {ler_standard:.4f}")
    print(f"Hybrid GNN+SparseBlossom LER: {ler_hybrid:.4f}")
    print(f"Improvement: {improvement:.1f}%")
    print(f"Speedup/slowdown: {avg_time_hybrid / avg_time_standard:.2f}x")

    if ler_hybrid < ler_standard:
        print("\n[OK] GNN training improved logical error rate!")
    else:
        print("\n[WARNING] GNN training did not improve LER (may need more training data/epochs)")


if __name__ == "__main__":
    main()
