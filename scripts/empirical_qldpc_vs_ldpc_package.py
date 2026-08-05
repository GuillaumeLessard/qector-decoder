"""
DEDICATED qLDPC REAL EMPIRICAL BENCHMARK (QECTOR v1.0.0 vs Reference `ldpc` Package)

Evaluates QECTOR MatrixBPOSDDecoder and GraphBPOSDDecoder on Bivariate Bicycle BB72
and Bicycle qLDPC codes across physical error rates p in {0.03, 0.05, 0.08}.
"""

import json
import os
import time

import numpy as np
from qector_decoder_v3 import GraphBPOSDDecoder, MatrixBPOSDDecoder, codes

try:
    from ldpc.bposd_decoder import BpOsdDecoder as ReferenceBpOsdDecoder
    HAS_LDPC_PKG = True
except ImportError:
    HAS_LDPC_PKG = False


def run_qldpc_empirical_benchmark():
    print("==========================================================================================")
    print("      QECTOR v1.0.0 DEDICATED qLDPC / HYPERGRAPH EMPIRICAL BENCHMARK vs `ldpc`             ")
    print("==========================================================================================")
    print(f"Reference `ldpc` package installed: {HAS_LDPC_PKG}")

    os.makedirs("benchmark_results", exist_ok=True)
    results = []

    # 1. Bivariate Bicycle BB72 Code
    # a_terms = [("x", 3), ("y", 1), ("y", 2)], b_terms = [("y", 3), ("x", 1), ("x", 2)]
    a_terms = [("x", 3), ("y", 1), ("y", 2)]
    b_terms = [("y", 3), ("x", 1), ("x", 2)]
    code_bb72, _ = codes.bivariate_bicycle_code(6, 6, a_terms, b_terms)

    # 2. Bicycle Code (20 circulants -> 40 qubits)
    code_bic, _ = codes.bicycle_code(20)

    test_codes = [
        ("qLDPC BB72 Code", code_bb72),
        ("qLDPC Bicycle Code", code_bic),
    ]

    p_values = [0.03, 0.05, 0.08]
    num_shots = 10000

    for code_name, code in test_codes:
        H = code.parity_check_matrix()
        c2q = code.check_to_qubits
        nq = code.n_qubits
        n_checks = H.shape[0]

        for p in p_values:
            print(f"\n>>> Code: {code_name} ({n_checks} checks, {nq} qubits) | Noise p={p} | Shots={num_shots}")

            # Generate physical errors e ~ Bernoulli(p)
            rng = np.random.default_rng(42 + int(p * 100))
            errors = (rng.random((num_shots, nq)) < p).astype(np.uint8)
            syndromes = ((errors @ H.T) % 2).astype(np.uint8)

            # -----------------------------------------------------------------
            # 1. QECTOR MatrixBPOSDDecoder (Dense GF2)
            # -----------------------------------------------------------------
            q_matrix = MatrixBPOSDDecoder(H, error_rate=p, max_iter=30, osd_order=0)
            t0 = time.perf_counter()
            q_corr_m = q_matrix.batch_decode(syndromes)
            t_m = time.perf_counter() - t0
            m_fps = num_shots / t_m
            m_valid = float(np.mean(((q_corr_m @ H.T) % 2) == syndromes)) * 100.0
            m_rec = float(np.mean(np.all(q_corr_m == errors, axis=1))) * 100.0

            # Single-shot latency
            lat_m = []
            for s in syndromes[:500]:
                t_s = time.perf_counter()
                _ = q_matrix.decode(s)
                lat_m.append((time.perf_counter() - t_s) * 1e6)
            m_p50 = float(np.percentile(lat_m, 50))
            m_p99 = float(np.percentile(lat_m, 99))

            print(f"  [QECTOR MatrixBPOSD]  Validity = {m_valid:6.2f}% | Word Rec = {m_rec:5.2f}% | FPS = {m_fps:8.1f} | p50 = {m_p50:5.1f} us")
            results.append({
                "code": code_name,
                "decoder": "QECTOR MatrixBPOSD",
                "p": p,
                "shots": num_shots,
                "validity_pct": m_valid,
                "word_recovery_pct": m_rec,
                "fps": m_fps,
                "p50_us": m_p50,
                "p99_us": m_p99,
            })

            # -----------------------------------------------------------------
            # 2. QECTOR GraphBPOSDDecoder (Native Adjacency Rust Core)
            # -----------------------------------------------------------------
            q_graph = GraphBPOSDDecoder(c2q, nq, p)
            t0 = time.perf_counter()
            q_corr_g = q_graph.batch_decode(syndromes)
            t_g = time.perf_counter() - t0
            g_fps = num_shots / t_g
            g_valid = float(np.mean(((q_corr_g @ H.T) % 2) == syndromes)) * 100.0
            g_rec = float(np.mean(np.all(q_corr_g == errors, axis=1))) * 100.0

            lat_g = []
            for s in syndromes[:500]:
                t_s = time.perf_counter()
                _ = q_graph.decode(s)
                lat_g.append((time.perf_counter() - t_s) * 1e6)
            g_p50 = float(np.percentile(lat_g, 50))
            g_p99 = float(np.percentile(lat_g, 99))

            print(f"  [QECTOR GraphBPOSD]   Validity = {g_valid:6.2f}% | Word Rec = {g_rec:5.2f}% | FPS = {g_fps:8.1f} | p50 = {g_p50:5.1f} us")
            results.append({
                "code": code_name,
                "decoder": "QECTOR GraphBPOSD",
                "p": p,
                "shots": num_shots,
                "validity_pct": g_valid,
                "word_recovery_pct": g_rec,
                "fps": g_fps,
                "p50_us": g_p50,
                "p99_us": g_p99,
            })

            # -----------------------------------------------------------------
            # 3. Reference `ldpc` Package (if installed)
            # -----------------------------------------------------------------
            if HAS_LDPC_PKG:
                try:
                    ref_dec = ReferenceBpOsdDecoder(
                        H,
                        error_rate=p,
                        max_iter=30,
                        bp_method="ms",
                        osd_method="osd_cs",
                        osd_order=0,
                    )
                    t0 = time.perf_counter()
                    ref_corr = np.zeros((num_shots, nq), dtype=np.uint8)
                    for i in range(num_shots):
                        ref_corr[i] = ref_dec.decode(syndromes[i])
                    t_ref = time.perf_counter() - t0
                    ref_fps = num_shots / t_ref
                    ref_valid = float(np.mean(((ref_corr @ H.T) % 2) == syndromes)) * 100.0
                    ref_rec = float(np.mean(np.all(ref_corr == errors, axis=1))) * 100.0

                    print(f"  [Reference ldpc pkg]  Validity = {ref_valid:6.2f}% | Word Rec = {ref_rec:5.2f}% | FPS = {ref_fps:8.1f}")
                    results.append({
                        "code": code_name,
                        "decoder": "Reference ldpc pkg",
                        "p": p,
                        "shots": num_shots,
                        "validity_pct": ref_valid,
                        "word_recovery_pct": ref_rec,
                        "fps": ref_fps,
                        "p50_us": 0.0,
                        "p99_us": 0.0,
                    })
                except Exception as ex:
                    print(f"  [Reference ldpc pkg]  Skipped ({ex})")

    # -----------------------------------------------------------------
    # Save Artifacts
    # -----------------------------------------------------------------
    with open("benchmark_results/qldpc_vs_ldpc_empirical.json", "w") as f:
        json.dump(results, f, indent=2)

    import csv
    with open("benchmark_results/qldpc_vs_ldpc_empirical.csv", "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(results[0].keys()))
        writer.writeheader()
        writer.writerows(results)

    print("\n==========================================================================================")
    print("                     qLDPC EMPIRICAL BENCHMARK COMPLETE & SAVED                             ")
    print("  Artifacts: benchmark_results/qldpc_vs_ldpc_empirical.json                              ")
    print("             benchmark_results/qldpc_vs_ldpc_empirical.csv                               ")
    print("==========================================================================================")


if __name__ == "__main__":
    run_qldpc_empirical_benchmark()
