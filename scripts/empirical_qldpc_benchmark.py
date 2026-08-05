"""
qLDPC / Hypergraph Code Real Empirical Benchmark for QECTOR v0.9.0.

Evaluates MatrixBPOSDDecoder on general qLDPC codes where MWPM / PyMatching fails.
"""

import time

import numpy as np
import qector_decoder_v3 as qd
from qector_decoder_v3 import MatrixBPOSDDecoder


def run_qldpc_benchmark():
    print("==========================================================================")
    print("         EMPIRICAL QECTOR v0.9.0 qLDPC / HYPERGRAPH REAL BENCHMARK         ")
    print("==========================================================================")

    code_x, code_z = qd.codes.bicycle_code(20)
    H = code_x.parity_check_matrix()
    n_qubits = code_x.n_qubits
    print(f"qLDPC Bicycle Code: {H.shape[0]} checks, {n_qubits} qubits")

    bposd = MatrixBPOSDDecoder(H, error_rate=0.05, max_iter=30, osd_order=0)

    # Generate 5,000 physical error shots
    num_shots = 5000
    rng = np.random.default_rng(42)
    errors = (rng.random((num_shots, n_qubits)) < 0.05).astype(np.uint8)

    # Compute syndromes s = H @ e (mod 2)
    syndromes = ((errors @ H.T) % 2).astype(np.uint8)

    t0 = time.perf_counter()
    corrections = bposd.batch_decode(syndromes)
    t_total = time.perf_counter() - t0

    # Verify syndrome validity: H @ c == s (mod 2)
    residual_syndromes = ((corrections @ H.T) % 2).astype(np.uint8)
    syndrome_validity = np.mean(residual_syndromes == syndromes)

    # Exact error recovery rate (c == e)
    exact_recovery = np.mean(np.all(corrections == errors, axis=1))
    fps = num_shots / t_total

    print(f"Shots Decoded        : {num_shots}")
    print(f"Syndrome Validity    : {syndrome_validity * 100:.2f}% (H @ c == s)")
    print(f"Exact Word Recovery  : {exact_recovery * 100:.2f}%")
    print(f"BP-OSD Throughput    : {fps:.1f} shots/sec")
    print("==========================================================================")


if __name__ == "__main__":
    run_qldpc_benchmark()
