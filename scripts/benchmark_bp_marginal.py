#!/usr/bin/env python3
"""Compare GNN training with Blossom soft targets vs BP marginal targets."""
import time
import numpy as np
from qector_decoder_v3 import (
    HybridDecoder,
    SparseBlossomDecoder,
    generate_surface_code_checks,
)


def compute_syndrome(error, check_to_qubits):
    n_checks = len(check_to_qubits)
    syndrome = np.zeros(n_checks, dtype=np.uint8)
    for ci, qubits in enumerate(check_to_qubits):
        parity = 0
        for q in qubits:
            parity ^= int(error[q])
        syndrome[ci] = parity
    return syndrome


def benchmark_ler(decoder_fn, check_to_qubits, n_qubits, n_samples, error_rate, seed=42):
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
    distance = 5
    error_rate = 0.05
    n_train = 200
    n_epochs = 5
    n_test = 500

    result = generate_surface_code_checks(distance)
    check_to_qubits = result[0]
    n_qubits = result[1]
    print(f"Surface code d={distance}: {n_qubits} qubits, {len(check_to_qubits)} checks")
    print(f"Training: {n_train} samples x {n_epochs} epochs, p={error_rate}")
    print(f"Test: {n_test} samples")
    print()

    # Standard SparseBlossom
    sparse = SparseBlossomDecoder(check_to_qubits, n_qubits=n_qubits)
    ler_std, time_std = benchmark_ler(
        lambda s: sparse.decode(s), check_to_qubits, n_qubits, n_test, error_rate, seed=42
    )
    print(f"Standard SparseBlossom:  LER={ler_std:.4f}, Avg time={time_std:.1f} µs")

    # --- Blossom soft target training ---
    hybrid_soft = HybridDecoder(check_to_qubits, n_qubits=n_qubits)
    t0 = time.perf_counter()
    loss_soft = hybrid_soft.train(n_train, n_epochs, error_rate)
    t1 = time.perf_counter()
    print(f"\nSoft target training:    loss={loss_soft:.8f}, time={t1-t0:.1f}s")

    ler_soft, time_soft = benchmark_ler(
        lambda s: hybrid_soft.decode_hybrid(s), check_to_qubits, n_qubits, n_test, error_rate, seed=42
    )
    print(f"Soft target (trained):   LER={ler_soft:.4f}, Avg time={time_soft:.1f} µs")

    # --- BP marginal target training ---
    hybrid_bp = HybridDecoder(check_to_qubits, n_qubits=n_qubits)
    t0 = time.perf_counter()
    loss_bp = hybrid_bp.train_bp(n_train, n_epochs, error_rate, max_bp_iter=10)
    t1 = time.perf_counter()
    print(f"\nBP marginal training:    loss={loss_bp:.8f}, time={t1-t0:.1f}s")

    ler_bp, time_bp = benchmark_ler(
        lambda s: hybrid_bp.decode_hybrid(s), check_to_qubits, n_qubits, n_test, error_rate, seed=42
    )
    print(f"BP marginal (trained):   LER={ler_bp:.4f}, Avg time={time_bp:.1f} µs")

    # Summary
    print(f"\n{'='*50}")
    print(f"Summary d={distance}")
    print(f"{'='*50}")
    print(f"Standard SparseBlossom:  LER={ler_std:.4f}  time={time_std:.1f} µs")
    print(f"Soft target trained:     LER={ler_soft:.4f}  time={time_soft:.1f} µs")
    print(f"BP marginal trained:     LER={ler_bp:.4f}  time={time_bp:.1f} µs")
    if ler_std > 0:
        print(f"Soft improvement:        {(ler_std-ler_soft)/ler_std*100:+.1f}%")
        print(f"BP improvement:          {(ler_std-ler_bp)/ler_std*100:+.1f}%")


if __name__ == "__main__":
    main()
