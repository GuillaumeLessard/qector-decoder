#!/usr/bin/env python3
"""Quick benchmark of heuristic decoder vs standard decoder."""
import sys
import time
import numpy as np

sys.path.insert(0, "python")
from qector_decoder_v3 import HybridDecoder, generate_surface_code_checks


def benchmark(decoder, check_to_qubits, n_samples, error_rate, seed, method="standard"):
    rng = np.random.default_rng(seed)
    n_qubits = decoder.n_qubits
    n_checks = decoder.n_checks
    n_errors = 0
    total_time = 0.0

    for i in range(n_samples):
        error = (rng.random(n_qubits) < error_rate).astype(np.uint8)
        syndrome = np.zeros(n_checks, dtype=np.uint8)
        for ci, qubits in enumerate(check_to_qubits):
            parity = 0
            for q in qubits:
                parity ^= int(error[q])
            syndrome[ci] = parity

        t0 = time.perf_counter()
        if method == "standard":
            correction = decoder.decode_standard(syndrome)
        elif method == "heuristic":
            correction = decoder.decode_heuristic(syndrome)
        elif method == "hybrid":
            correction = decoder.decode_hybrid(syndrome)
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

    decoder = HybridDecoder(check_to_qubits, n_qubits=n_qubits, gnn_hidden_size=16, gnn_n_layers=2)

    ler_std, time_std = benchmark(decoder, check_to_qubits, n_samples, error_rate, 42, "standard")
    print(f"Standard SparseBlossom: LER={ler_std:.4f}, Avg time={time_std:.1f} µs")

    ler_heur, time_heur = benchmark(decoder, check_to_qubits, n_samples, error_rate, 42, "heuristic")
    print(f"Heuristic decoder:      LER={ler_heur:.4f}, Avg time={time_heur:.1f} µs")

    improvement = (ler_std - ler_heur) / ler_std * 100 if ler_std > 0 else 0
    print(f"Improvement: {improvement:.1f}%")


if __name__ == "__main__":
    main()
