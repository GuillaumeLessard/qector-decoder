#!/usr/bin/env python3
"""Comprehensive training and benchmark for larger surface codes (d >= 7).

Trains GNN with reduced samples/epochs for practicality, then benchmarks LER.
"""
import sys
import time
import numpy as np

from qector_decoder_v3 import (
    HybridDecoder,
    BlossomDecoder,
    SparseBlossomDecoder,
    generate_surface_code_checks,
)


def compute_syndrome(error, check_to_qubits):
    """Compute syndrome from error pattern."""
    n_checks = len(check_to_qubits)
    syndrome = np.zeros(n_checks, dtype=np.uint8)
    for ci, qubits in enumerate(check_to_qubits):
        parity = 0
        for q in qubits:
            parity ^= int(error[q])
        syndrome[ci] = parity
    return syndrome


def benchmark_ler(decoder_fn, check_to_qubits, n_qubits, n_samples, error_rate, seed=42):
    """Benchmark logical error rate (proxy: exact correction match)."""
    rng = np.random.default_rng(seed)
    n_errors = 0
    total_time = 0.0
    for i in range(n_samples):
        error = (rng.random(n_qubits) < error_rate).astype(np.uint8)
        syndrome = compute_syndrome(error, check_to_qubits)
        t0 = time.perf_counter()
        correction = decoder_fn(syndrome)
        t1 = time.perf_counter()
        total_time += (t1 - t0) * 1e6
        if not np.array_equal(correction, error):
            n_errors += 1
    return n_errors / n_samples, total_time / n_samples


def main():
    distances = [5]
    error_rate = 0.05
    n_train = 200
    n_epochs = 5
    n_test = 500

    for distance in distances:
        print(f"\n{'='*60}")
        print(f"Surface code d={distance}")
        print(f"{'='*60}")

        result = generate_surface_code_checks(distance)
        check_to_qubits = result[0]
        n_qubits = result[1]
        n_checks = len(check_to_qubits)
        print(f"Qubits: {n_qubits}, Checks: {n_checks}")

        # Standard SparseBlossom
        sparse = SparseBlossomDecoder(check_to_qubits, n_qubits=n_qubits)
        ler_std, time_std = benchmark_ler(
            lambda s: sparse.decode(s), check_to_qubits, n_qubits, n_test, error_rate, seed=42
        )
        print(f"Standard SparseBlossom: LER={ler_std:.4f}, Avg time={time_std:.1f} µs")

        # Hybrid decoder (untrained GNN)
        hybrid = HybridDecoder(
            check_to_qubits,
            n_qubits=n_qubits,
            gnn_hidden_size=64,
            gnn_n_layers=3,
        )
        ler_untrained, time_untrained = benchmark_ler(
            lambda s: hybrid.decode_hybrid(s), check_to_qubits, n_qubits, n_test, error_rate, seed=42
        )
        print(f"Hybrid (untrained):     LER={ler_untrained:.4f}, Avg time={time_untrained:.1f} µs")

        # Heuristic decoder
        ler_heur, time_heur = benchmark_ler(
            lambda s: hybrid.decode_heuristic(s), check_to_qubits, n_qubits, n_test, error_rate, seed=42
        )
        print(f"Heuristic decoder:      LER={ler_heur:.4f}, Avg time={time_heur:.1f} µs")

        # Train GNN
        print(f"\nTraining GNN: {n_train} samples, {n_epochs} epochs, p={error_rate}...")
        t0 = time.perf_counter()
        final_loss = hybrid.train(n_train, n_epochs, error_rate)
        t1 = time.perf_counter()
        print(f"Training complete in {t1-t0:.1f}s, final loss={final_loss:.8f}")

        # Benchmark trained GNN
        ler_trained, time_trained = benchmark_ler(
            lambda s: hybrid.decode_hybrid(s), check_to_qubits, n_qubits, n_test, error_rate, seed=42
        )
        print(f"Hybrid (trained):       LER={ler_trained:.4f}, Avg time={time_trained:.1f} µs")

        # Summary
        print(f"\n--- Summary d={distance} ---")
        print(f"Standard:      LER={ler_std:.4f}  time={time_std:.1f} µs")
        print(f"Untrained:     LER={ler_untrained:.4f}  time={time_untrained:.1f} µs")
        print(f"Heuristic:     LER={ler_heur:.4f}  time={time_heur:.1f} µs")
        print(f"Trained GNN:   LER={ler_trained:.4f}  time={time_trained:.1f} µs")
        if ler_std > 0:
            improvement = (ler_std - ler_trained) / ler_std * 100
            print(f"Improvement over standard: {improvement:+.1f}%")


if __name__ == "__main__":
    main()
