#!/usr/bin/env python3
"""Quick benchmark of heuristic decoder using SparseBlossomDecoder directly."""
import sys
import time
import numpy as np

sys.path.insert(0, "python")
from qector_decoder_v3 import (
    SparseBlossomDecoder,
    generate_surface_code_checks,
)


def benchmark(decoder, check_to_qubits, n_samples, error_rate, seed, use_heuristic=False):
    rng = np.random.default_rng(seed)
    n_qubits = decoder.n_qubits
    n_checks = decoder.n_checks
    n_errors = 0
    total_time = 0.0

    # Build qubit -> checks mapping for heuristic
    qubit_to_checks = [[] for _ in range(n_qubits)]
    for ci, qubits in enumerate(check_to_qubits):
        for q in qubits:
            if q < n_qubits:
                qubit_to_checks[q].append(ci)

    for i in range(n_samples):
        error = (rng.random(n_qubits) < error_rate).astype(np.uint8)
        syndrome = np.zeros(n_checks, dtype=np.uint8)
        for ci, qubits in enumerate(check_to_qubits):
            parity = 0
            for q in qubits:
                parity ^= int(error[q])
            syndrome[ci] = parity

        t0 = time.perf_counter()
        if use_heuristic:
            # Compute heuristic weights
            dynamic_weights = []
            for q in range(n_qubits):
                checks = qubit_to_checks[q]
                n_violated = sum(1 for ci in checks if syndrome[ci] == 1)
                weight = {0: 1.0, 1: 3.0, 2: 10.0}.get(n_violated, 10.0)
                dynamic_weights.append((q, weight))
            correction = decoder.decode_with_weights(syndrome, dynamic_weights)
        else:
            correction = decoder.decode(syndrome)
        t1 = time.perf_counter()
        total_time += (t1 - t0) * 1e6

        if not np.array_equal(correction, error):
            n_errors += 1

    return n_errors / n_samples, total_time / n_samples


def main():
    distance = 5
    error_rate = 0.05
    n_samples = 200

    result = generate_surface_code_checks(distance)
    check_to_qubits = result[0]
    n_qubits = result[1]
    n_checks = len(check_to_qubits)

    print(f"Surface code d={distance}: {n_qubits} qubits, {n_checks} checks")
    print(f"Benchmarking {n_samples} samples at p={error_rate}")
    print()

    decoder = SparseBlossomDecoder(check_to_qubits, n_qubits=n_qubits)

    ler_std, time_std = benchmark(decoder, check_to_qubits, n_samples, error_rate, 42, False)
    print(f"Standard SparseBlossom: LER={ler_std:.4f}, Avg time={time_std:.1f} µs")

    ler_heur, time_heur = benchmark(decoder, check_to_qubits, n_samples, error_rate, 42, True)
    print(f"Heuristic decoder:      LER={ler_heur:.4f}, Avg time={time_heur:.1f} µs")

    improvement = (ler_std - ler_heur) / ler_std * 100 if ler_std > 0 else 0
    print(f"Improvement: {improvement:.1f}%")


if __name__ == "__main__":
    main()
