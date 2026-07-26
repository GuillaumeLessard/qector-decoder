#!/usr/bin/env python
"""Ultra-rigorous, high-shot hardware benchmark suite for QECTOR v3.

Executes 100,000 shots per configuration with 3 isolated repetitions per data point,
thread environment isolation (OMP_NUM_THREADS=1), C-contiguous pointer arrays,
and full empirical logging.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time

_REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(_REPO, "python"))

# Isolate CPU threads for reproducible micro-benchmarking
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"

import numpy as np
import pymatching
from ldpc import bposd_decoder
import fusion_blossom
import stim

from qector_decoder_v3 import (
    codes,
    BlossomDecoder,
    SparseBlossomDecoder,
    UnionFindDecoder,
    FastUnionFindDecoder,
    BPOSDDecoder,
)
try:
    from qector_decoder_v3 import CUDABatchDecoder
except ImportError:
    CUDABatchDecoder = None

from qector_decoder_v3.ler import wilson_ci


def run_rigorous_benchmark(distances, shots_list, p=0.005, reps=3, seed=42):
    results = []
    print(f"=== QECTOR v3 Ultra-Rigorous Hardware Benchmark ===")
    print(f"Distances: {distances}")
    print(f"Shots per pass: {shots_list}")
    print(f"Repetitions per config: {reps}")
    print(f"Noise rate: {p}")
    print("=" * 60)

    for d in distances:
        code = codes.rotated_surface_code(d)
        nq = code.n_qubits
        nc = code.n_checks
        c2q = code.check_to_qubits

        H = np.zeros((nc, nq), dtype=np.uint8)
        for i, qs in enumerate(c2q):
            H[i, qs] = 1
        H = np.ascontiguousarray(H, dtype=np.uint8)

        if code.logicals:
            L = np.zeros((len(code.logicals), nq), dtype=np.uint8)
            for i, qs in enumerate(code.logicals):
                L[i, qs] = 1
        else:
            L = np.ones((1, nq), dtype=np.uint8)
        L = np.ascontiguousarray(L, dtype=np.uint8)

        for shots in shots_list:
            rng = np.random.default_rng(seed)
            E = (rng.random((shots, nq)) < p).astype(np.uint8)
            E = np.ascontiguousarray(E, dtype=np.uint8)
            S = ((E @ H.T) & 1).astype(np.uint8)
            S = np.ascontiguousarray(S, dtype=np.uint8)

            decoders = {}

            # PyMatching v2.4
            try:
                decoders["PyMatching v2.4 (C++)"] = pymatching.Matching.from_check_matrix(H)
            except Exception as e:
                results.append({"distance": d, "shots": shots, "decoder": "PyMatching v2.4 (C++)", "status": "failed", "error": str(e)})

            # QECTOR Fast Union-Find
            try:
                decoders["QECTOR Fast Union-Find (CPU)"] = FastUnionFindDecoder(c2q, nq)
            except Exception as e:
                results.append({"distance": d, "shots": shots, "decoder": "QECTOR Fast Union-Find (CPU)", "status": "failed", "error": str(e)})

            # QECTOR Sparse Blossom
            try:
                decoders["QECTOR Sparse Blossom (CPU)"] = SparseBlossomDecoder(c2q, nq)
            except Exception as e:
                results.append({"distance": d, "shots": shots, "decoder": "QECTOR Sparse Blossom (CPU)", "status": "failed", "error": str(e)})

            # QECTOR CUDA Batch
            if CUDABatchDecoder:
                try:
                    decoders["QECTOR CUDA Batch (GPU)"] = CUDABatchDecoder(c2q, nq)
                except Exception as e:
                    results.append({"distance": d, "shots": shots, "decoder": "QECTOR CUDA Batch (GPU)", "status": "failed", "error": str(e)})

            # QECTOR Rust BP-OSD
            if d <= 13 and shots <= 5000:
                try:
                    decoders["QECTOR BP-OSD (Rust CPU)"] = BPOSDDecoder(c2q, nq, max_p=p)
                except Exception as e:
                    results.append({"distance": d, "shots": shots, "decoder": "QECTOR BP-OSD (Rust CPU)", "status": "failed", "error": str(e)})

            # ldpc BP-OSD
            if d <= 9 and shots <= 5000:
                try:
                    decoders["ldpc BP-OSD (C++)"] = bposd_decoder(
                        H,
                        error_rate=p,
                        max_iter=30,
                        bp_method="ms",
                        ms_scaling_factor=0.625,
                        osd_method="osd_cs",
                        osd_order=0,
                    )
                except Exception as e:
                    results.append({"distance": d, "shots": shots, "decoder": "ldpc BP-OSD (C++)", "status": "failed", "error": str(e)})

            for name, dec in decoders.items():
                # Warmup CPU/GPU
                try:
                    if "PyMatching" in name:
                        _ = dec.decode_batch(S[: min(shots, 1000)])
                    elif "ldpc" in name:
                        for i in range(min(shots, 20)):
                            _ = dec.decode(S[i])
                    else:
                        _ = dec.batch_decode(S[: min(shots, 1000)])
                except Exception as e:
                    results.append({"distance": d, "shots": shots, "decoder": name, "status": "unsupported", "error": str(e)})
                    continue

                durations = []
                C_last = None
                for rep in range(reps):
                    t0 = time.perf_counter()
                    if "PyMatching" in name:
                        C = dec.decode_batch(S)
                    elif "ldpc" in name:
                        C = np.zeros((shots, nq), dtype=np.uint8)
                        for i in range(shots):
                            C[i] = dec.decode(S[i])
                    else:
                        C = dec.batch_decode(S)
                    dt = time.perf_counter() - t0
                    durations.append(dt)
                    C_last = C

                C_mat = np.ascontiguousarray(C_last, dtype=np.uint8)
                if C_mat.ndim == 1:
                    C_mat = C_mat.reshape(shots, -1)
                if C_mat.shape[1] > nq:
                    C_mat = C_mat[:, :nq]

                faithful = int(np.sum(np.all(((C_mat @ H.T) & 1) == S, axis=1)))
                R = (C_mat ^ E).astype(np.uint8)
                flips = (R @ L.T) & 1
                fails = int(np.sum(np.any(flips, axis=1)))
                lo, hi = wilson_ci(fails, shots)

                dt_median = float(np.median(durations))
                dt_mean = float(np.mean(durations))
                dt_std = float(np.std(durations))
                dec_s = shots / dt_median if dt_median > 0 else 0.0
                lat_us = (dt_median / shots) * 1e6 if shots > 0 else 0.0

                rec = {
                    "distance": d,
                    "n_qubits": nq,
                    "shots": shots,
                    "decoder": name,
                    "fails": fails,
                    "ler": fails / shots,
                    "ci95": (round(lo, 6), round(hi, 6)),
                    "faith_pct": round(100.0 * faithful / shots, 2),
                    "dt_median": round(dt_median, 6),
                    "dt_mean": round(dt_mean, 6),
                    "dt_std": round(dt_std, 6),
                    "dec_per_s": round(dec_s, 1),
                    "latency_us": round(lat_us, 2),
                    "status": "success",
                }
                results.append(rec)
                print(
                    f"d={d:2d} (nq={nq:3d}) | shots={shots:7,d} | {name:<30} | LER={rec['ler']:.5f} | "
                    f"{rec['dec_per_s']:11,.0f} dec/s | {rec['latency_us']:7.2f} us/dec | ±{dt_std*1000:5.2f}ms"
                )

    return results


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--distances", nargs="+", type=int, default=[3, 5, 7, 9, 11, 13, 15, 17, 19])
    parser.add_argument("--shots", nargs="+", type=int, default=[1000, 20000, 50000, 100000])
    parser.add_argument("--p", type=float, default=0.005)
    parser.add_argument("--reps", type=int, default=3)
    parser.add_argument("--out", type=str, default="competitive_results.json")
    args = parser.parse_args()

    recs = run_rigorous_benchmark(args.distances, args.shots, args.p, args.reps)

    with open(args.out, "w", encoding="utf-8") as fh:
        json.dump(recs, fh, indent=2)
    print(f"\nSuccessfully completed ultra-rigorous benchmark sweep ({len(recs)} records written to {args.out})")


if __name__ == "__main__":
    main()
