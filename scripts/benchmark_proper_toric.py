#!/usr/bin/env python3
"""Benchmark with proper toric code using CSS-aware syndrome computation."""
import time
import numpy as np
from qector_decoder_v3 import (
    HybridDecoder,
    SparseBlossomDecoder,
    generate_surface_code_checks,
)


def compute_x_syndrome(error, check_to_qubits, d):
    """Compute syndrome for X errors (only Z-stabilizers detect X errors)."""
    n_checks = len(check_to_qubits)
    syndrome = np.zeros(n_checks, dtype=np.uint8)
    # Z-stabilizers are the last d*d checks
    for ci in range(d * d, n_checks):
        qubits = check_to_qubits[ci]
        parity = 0
        for q in qubits:
            parity ^= int(error[q])
        syndrome[ci] = parity
    return syndrome


def check_valid(correction, syndrome, check_to_qubits, d):
    """Check if correction is valid for X-error syndrome."""
    n_checks = len(check_to_qubits)
    for ci in range(d * d, n_checks):
        qubits = check_to_qubits[ci]
        parity = 0
        for q in qubits:
            parity ^= int(correction[q])
        if parity != syndrome[ci]:
            return False
    return True


def benchmark_decoder(decoder, check_to_qubits, d, n_samples, error_rate, seed=42):
    rng = np.random.default_rng(seed)
    n_qubits = decoder.n_qubits
    n_valid = 0
    total_time = 0.0
    for _ in range(n_samples):
        error = (rng.random(n_qubits) < error_rate).astype(np.uint8)
        full_syndrome = compute_x_syndrome(error, check_to_qubits, d)
        # Extract Z-stabilizer syndrome for decoder
        z_syndrome = full_syndrome[d * d:]
        t0 = time.perf_counter()
        if hasattr(decoder, 'decode'):
            correction = decoder.decode(z_syndrome)
        else:
            # HybridDecoder uses decode_hybrid or decode_standard
            correction = decoder.decode_hybrid(z_syndrome)
        t1 = time.perf_counter()
        total_time += (t1 - t0) * 1e6
        if check_valid(correction, full_syndrome, check_to_qubits, d):
            n_valid += 1
    return n_valid / n_samples, total_time / n_samples


def main():
    d = 5
    error_rate = 0.05
    n_test = 500
    n_train = 200
    n_epochs = 5

    result = generate_surface_code_checks(d)
    check_to_qubits = result[0]
    n_qubits = result[1]
    print(f"Proper toric code d={d}: {n_qubits} qubits, {len(check_to_qubits)} checks")
    print(f"Z-stabilizers: {d*d} (used for X-error decoding)")
    print()

    # Standard decoder (only Z-stabilizers)
    z_checks = check_to_qubits[d * d:]
    sparse = SparseBlossomDecoder(z_checks, n_qubits=n_qubits)
    valid_rate, avg_time = benchmark_decoder(sparse, check_to_qubits, d, n_test, error_rate)
    print(f"Standard SparseBlossom: {valid_rate*100:.1f}% valid, Avg time={avg_time:.1f} µs")

    # Hybrid with GNN training (only Z-stabilizers for decoder)
    hybrid = HybridDecoder(z_checks, n_qubits=n_qubits)
    t0 = time.perf_counter()
    loss = hybrid.train(n_train, n_epochs, error_rate)
    t1 = time.perf_counter()
    print(f"\nGNN training: {n_train}×{n_epochs}, loss={loss:.8f}, time={t1-t0:.1f}s")

    # Benchmark trained GNN
    valid_rate_gnn, avg_time_gnn = benchmark_decoder(
        hybrid, check_to_qubits, d, n_test, error_rate
    )
    print(f"Trained GNN: {valid_rate_gnn*100:.1f}% valid, Avg time={avg_time_gnn:.1f} µs")


if __name__ == "__main__":
    main()
