"""Benchmark Logical Error Rate (LER) -- QECTOR v3.

Compare le taux d'erreur logique entre :
- UnionFindDecoder (quasi-optimal, rapide)
- BlossomDecoder (optimal MWPM, exact DP)
- PyMatching (si installe, reference externe)

Pour un ring code en 1D, une erreur logique survient si le nombre total de
qubits errones (erreur injectee XOR correction) est impair, i.e. la chaine
ne s'annule pas dans le sous-espace stabilisateur.
"""

import json
import sys
import time
import numpy as np

import qector_decoder_v3 as qd


def logical_error_ring(correction: np.ndarray, error: np.ndarray) -> bool:
    """Retourne True si une erreur logique persiste (nombre de qubits 1 impair)."""
    residual = correction ^ error
    return int(np.sum(residual)) % 2 == 1


def run_ler_benchmark(decoder_cls, checks, n_qubits, distances, error_rates, n_trials=5000):
    """Execute le benchmark LER pour un decodeur donne."""
    results = {}
    for d in distances:
        checks_d, n_qubits_d = qd.generate_ring_code_checks(d)
        dec = decoder_cls(checks_d, n_qubits_d)
        results[d] = {}
        for p in error_rates:
            errors = 0
            t0 = time.perf_counter()
            for _ in range(n_trials):
                error = np.random.rand(n_qubits_d) < p
                error = error.astype(np.uint8)
                # Syndrome = parite des erreurs autour de chaque check
                syndrome = np.zeros(len(checks_d), dtype=np.uint8)
                for ci, qubits in enumerate(checks_d):
                    syndrome[ci] = int(np.sum(error[qubits]) % 2)
                correction = dec.decode(syndrome)
                if logical_error_ring(correction, error):
                    errors += 1
            t1 = time.perf_counter()
            ler = errors / n_trials
            latency = (t1 - t0) / n_trials * 1e6  # us
            results[d][p] = {"ler": ler, "latency_us": latency}
            print(f"  {decoder_cls.__name__} d={d} p={p:.3f} -> LER={ler:.4f}, latency={latency:.2f} us")
    return results


def main():
    distances = [5, 7, 10, 15]
    error_rates = [0.05, 0.10, 0.15]
    n_trials = 2000

    print("=" * 60)
    print("QECTOR v3 - Logical Error Rate Benchmark")
    print(f"Distances: {distances}")
    print(f"Error rates: {error_rates}")
    print(f"Trials per point: {n_trials}")
    print("=" * 60)

    all_results = {}

    # UnionFind
    print("\n--- UnionFindDecoder ---")
    all_results["UnionFindDecoder"] = run_ler_benchmark(
        qd.UnionFindDecoder, None, None, distances, error_rates, n_trials
    )

    # Blossom DP
    print("\n--- BlossomDecoder ---")
    all_results["BlossomDecoder"] = run_ler_benchmark(
        qd.BlossomDecoder, None, None, distances, error_rates, n_trials
    )

    # PyMatching (optionnel)
    try:
        import pymatching
        import scipy.sparse
        print("\n--- PyMatching (reference) ---")
        pymatching_results = {}
        for d in distances:
            checks_d, n_qubits_d = qd.generate_ring_code_checks(d)
            # Build check matrix H (n_checks x n_qubits)
            H = np.zeros((len(checks_d), n_qubits_d), dtype=np.uint8)
            for ci, qubits in enumerate(checks_d):
                for q in qubits:
                    H[ci, q] = 1
            H_sparse = scipy.sparse.csr_matrix(H)
            m = pymatching.Matching(H_sparse)
            pymatching_results[d] = {}
            for p in error_rates:
                errors = 0
                t0 = time.perf_counter()
                for _ in range(n_trials):
                    error = np.random.rand(n_qubits_d) < p
                    error = error.astype(np.uint8)
                    syndrome = np.zeros(len(checks_d), dtype=np.uint8)
                    for ci, qubits in enumerate(checks_d):
                        syndrome[ci] = int(np.sum(error[qubits]) % 2)
                    correction = m.decode(syndrome)
                    if logical_error_ring(correction, error):
                        errors += 1
                t1 = time.perf_counter()
                ler = errors / n_trials
                latency = (t1 - t0) / n_trials * 1e6
                pymatching_results[d][p] = {"ler": ler, "latency_us": latency}
                print(f"  PyMatching d={d} p={p:.3f} -> LER={ler:.4f}, latency={latency:.2f} us")
        all_results["PyMatching"] = pymatching_results
    except ImportError:
        print("\n--- PyMatching not installed (pip install pymatching) -- skip ---")
    except Exception as e:
        print(f"\n--- PyMatching error: {e} -- skip ---")

    # Sauvegarde
    out_path = "benchmarks/ler_results.json"
    with open(out_path, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\nResultats sauvegardes dans {out_path}")

    # Resume
    print("\n" + "=" * 60)
    print("Resume LER - UnionFind vs Blossom")
    print("=" * 60)
    for d in distances:
        for p in error_rates:
            ler_uf = all_results["UnionFindDecoder"][d][p]["ler"]
            ler_bl = all_results["BlossomDecoder"][d][p]["ler"]
            delta = ler_uf - ler_bl
            print(f"d={d:2d} p={p:.2f}  UF={ler_uf:.4f}  Blossom={ler_bl:.4f}  Delta={delta:+.5f}")


if __name__ == "__main__":
    main()
