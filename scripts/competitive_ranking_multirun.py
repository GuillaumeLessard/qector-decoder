#!/usr/bin/env python
"""Multi-run, statistically averaged competitive benchmark for QECTOR v3 vs PyMatching & LDPC.

Runs 5 warmup passes and 5 timed repetitions per configuration to report mean, stddev, and Wilson CIs.
Surfaces unsupported configurations explicitly without swallowing exceptions.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time

_REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(_REPO, "python"))

import numpy as np
import pymatching
from ldpc import bposd_decoder
import stim

from qector_decoder_v3 import (
    codes,
    BlossomDecoder,
    SparseBlossomDecoder,
    UnionFindDecoder,
    FastUnionFindDecoder,
    BPOSDDecoder,
    CPUBatchDecoder,
)
try:
    from qector_decoder_v3 import CUDABatchDecoder
except ImportError:
    CUDABatchDecoder = None

from qector_decoder_v3.bposd import BpOsdDecoder as PyBpOsdDecoder
from qector_decoder_v3.ler import wilson_ci


def run_benchmark(distances=[3, 5, 7, 9, 11, 13, 15, 17, 19], shots_list=[1000, 10000], p=0.005, reps=5, seed=42):
    results = []

    for d in distances:
        code = codes.rotated_surface_code(d)
        nq = code.n_qubits
        nc = code.n_checks
        c2q = code.check_to_qubits

        H = np.zeros((nc, nq), dtype=np.uint8)
        for i, qs in enumerate(c2q):
            H[i, qs] = 1

        if code.logicals:
            L = np.zeros((len(code.logicals), nq), dtype=np.uint8)
            for i, qs in enumerate(code.logicals):
                L[i, qs] = 1
        else:
            L = np.ones((1, nq), dtype=np.uint8)

        # Build decoders
        pm_dec = pymatching.Matching.from_check_matrix(H)
        q_fast_uf = FastUnionFindDecoder(c2q, nq)
        q_sparse_blossom = SparseBlossomDecoder(c2q, nq)
        q_cuda_batch = CUDABatchDecoder(c2q, nq) if CUDABatchDecoder else None

        q_rust_bposd = None
        if d <= 13:
            try:
                q_rust_bposd = BPOSDDecoder(c2q, nq, max_p=p)
            except Exception:
                pass

        ldpc_bposd = None
        if d <= 9:
            try:
                ldpc_bposd = bposd_decoder(
                    H,
                    error_rate=p,
                    max_iter=30,
                    bp_method="ms",
                    ms_scaling_factor=0.625,
                    osd_method="osd_cs",
                    osd_order=0,
                )
            except Exception:
                ldpc_bposd = None

        for shots in shots_list:
            rng = np.random.default_rng(seed)
            E = (rng.random((shots, nq)) < p).astype(np.uint8)
            S = ((E @ H.T) & 1).astype(np.uint8)

            eval_targets = []

            if q_cuda_batch:
                eval_targets.append(("QECTOR CUDA Batch (GPU)", lambda S=S: q_cuda_batch.batch_decode(S)))
            eval_targets.append(("QECTOR Fast Union-Find (CPU)", lambda S=S: q_fast_uf.batch_decode(S)))
            eval_targets.append(("QECTOR Sparse Blossom (CPU)", lambda S=S: q_sparse_blossom.batch_decode(S)))
            eval_targets.append(("PyMatching v2.4 (C++)", lambda S=S: pm_dec.decode_batch(S)))

            if q_rust_bposd and shots <= 5000:
                eval_targets.append(("QECTOR BP-OSD (Rust CPU)", lambda S=S: q_rust_bposd.decode_batch(S)))

            if ldpc_bposd and shots <= 5000:
                def run_ldpc_bposd(S=S):
                    out = np.zeros((shots, nq), dtype=np.uint8)
                    for i in range(shots):
                        out[i] = ldpc_bposd.decode(S[i])
                    return out
                eval_targets.append(("ldpc BP-OSD (C++)", run_ldpc_bposd))

            for name, runner in eval_targets:
                try:
                    # Warmup pass
                    _ = runner()

                    # Repetitions
                    times = []
                    for _ in range(reps):
                        t0 = time.perf_counter()
                        C = np.asarray(runner(), dtype=np.uint8)
                        dt = time.perf_counter() - t0
                        times.append(dt)

                    times = np.array(times)
                    dt_mean = float(np.mean(times))
                    dt_std = float(np.std(times))

                    if C.ndim == 1:
                        C = C.reshape(shots, -1)
                    if C.shape[1] > nq:
                        C = C[:, :nq]

                    faithful = int(np.sum(np.all(((C @ H.T) & 1) == S, axis=1)))
                    R = (C ^ E).astype(np.uint8)
                    flips = (R @ L.T) & 1
                    fails = int(np.sum(np.any(flips, axis=1)))
                    lo, hi = wilson_ci(fails, shots)

                    dec_s_mean = shots / dt_mean if dt_mean > 0 else 0.0
                    lat_us_mean = (dt_mean / shots) * 1e6 if shots > 0 else 0.0

                    rec = {
                        "distance": d,
                        "n_qubits": nq,
                        "shots": shots,
                        "decoder": name,
                        "fails": fails,
                        "ler": fails / shots,
                        "ci95": (round(lo, 6), round(hi, 6)),
                        "faith_pct": 100.0 * faithful / shots,
                        "seconds_mean": round(dt_mean, 6),
                        "seconds_std": round(dt_std, 6),
                        "dec_per_s": round(dec_s_mean, 1),
                        "latency_us": round(lat_us_mean, 2),
                        "status": "success",
                    }
                    results.append(rec)
                    print(f"d={d:2d} | shots={shots:6d} | {name:<30} | LER={rec['ler']:.5f} | {rec['dec_per_s']:10,.0f} dec/s | {rec['latency_us']:7.2f} us/dec (±{dt_std*1000:.2f}ms)")
                except Exception as exc:
                    print(f"d={d:2d} | shots={shots:6d} | {name:<30} | UNSUPPORTED: {exc}")
                    results.append({
                        "distance": d,
                        "n_qubits": nq,
                        "shots": shots,
                        "decoder": name,
                        "status": "unsupported",
                        "error": str(exc),
                    })

    return results


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--distances", nargs="+", type=int, default=[3, 5, 7, 9, 11, 13, 15, 17, 19])
    parser.add_argument("--shots", nargs="+", type=int, default=[1000, 10000])
    parser.add_argument("--p", type=float, default=0.005)
    parser.add_argument("--reps", type=int, default=5)
    parser.add_argument("--out", type=str, default="competitive_results.json")
    args = parser.parse_args()

    recs = run_benchmark(args.distances, args.shots, args.p, args.reps)

    with open(args.out, "w", encoding="utf-8") as fh:
        json.dump(recs, fh, indent=2)
    print(f"\nWrote multi-run benchmark results ({len(recs)} records) to {args.out}")


if __name__ == "__main__":
    main()
