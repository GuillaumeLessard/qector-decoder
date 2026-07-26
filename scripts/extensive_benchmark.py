#!/usr/bin/env python
"""Extensive multi-family benchmark for QECTOR v3 decoders.

Runs all decoders across surface code distances d=3 to d=19 for 1,000 and 20,000 shots.
Saves structured JSON results for report generation.
"""
from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time

_REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(_REPO, "python"))

import numpy as np
from qector_decoder_v3 import (
    codes,
    BlossomDecoder,
    SparseBlossomDecoder,
    UnionFindDecoder,
    FastUnionFindDecoder,
    BPOSDDecoder,
    CPUBatchDecoder,
)
from qector_decoder_v3.bposd import BpOsdDecoder as PyBpOsdDecoder
from qector_decoder_v3.ler import wilson_ci


def get_gpu_decoders():
    gpu_decs = {}
    try:
        from qector_decoder_v3 import CUDABatchDecoder
        gpu_decs["cuda_batch"] = CUDABatchDecoder
    except (ImportError, RuntimeError):
        pass
    try:
        from qector_decoder_v3 import OpenCLBatchDecoder
        gpu_decs["opencl_batch"] = OpenCLBatchDecoder
    except (ImportError, RuntimeError):
        pass
    try:
        from qector_decoder_v3 import CUDABpOsdDecoder
        gpu_decs["cuda_bp_osd"] = CUDABpOsdDecoder
    except (ImportError, RuntimeError):
        pass
    return gpu_decs


def benchmark_run(distances: list[int], shot_counts: list[int], p: float, seed: int) -> dict:
    gpu_decs = get_gpu_decoders()
    results = []

    for d in distances:
        code = codes.rotated_surface_code(d)
        H = code.parity_check_matrix().astype(np.uint8)
        L = code.logicals_matrix().astype(np.uint8)
        c2q = [list(map(int, c)) for c in code.check_to_qubits]
        nq = code.n_qubits
        nc = code.n_checks

        # Pre-instantiate decoders (warm reuse)
        decoders = {
            "blossom": lambda: BlossomDecoder(c2q, nq),
            "sparse_blossom": lambda: SparseBlossomDecoder(c2q, nq),
            "union_find": lambda: UnionFindDecoder(c2q, nq),
            "fast_union_find": lambda: FastUnionFindDecoder(c2q, nq),
            "bposd_rust": lambda: BPOSDDecoder(c2q, nq, p),
            "bposd_python": lambda: PyBpOsdDecoder(H, error_rate=p, max_iter=30, osd_order=0),
            "cpu_batch": lambda: CPUBatchDecoder(c2q, nq),
        }

        # Add GPU decoders if available
        if "cuda_batch" in gpu_decs:
            try:
                c_dec = gpu_decs["cuda_batch"](c2q, nq)
                decoders["cuda_batch"] = lambda dec=c_dec: dec
            except Exception:
                pass
        if "cuda_bp_osd" in gpu_decs:
            try:
                cbp_dec = gpu_decs["cuda_bp_osd"](c2q, nq, p)
                decoders["cuda_bp_osd"] = lambda dec=cbp_dec: dec
            except Exception:
                pass

        for shots in shot_counts:
            rng = np.random.default_rng(seed)
            E = (rng.random((shots, nq)) < p).astype(np.uint8)
            S = ((E @ H.T) & 1).astype(np.uint8)

            for dec_name, factory in decoders.items():
                # Skip heavy Python BP-OSD / exact Blossom on d >= 15 for 20k shots to finish in reasonable time
                if dec_name == "bposd_python" and shots >= 5000:
                    continue
                if dec_name in ("bposd_rust", "cuda_bp_osd") and d >= 11 and shots >= 50000:
                    continue
                if dec_name == "blossom" and d >= 13 and shots >= 50000:
                    continue

                try:
                    inst = factory()
                    t0 = time.perf_counter()
                    if hasattr(inst, "batch_decode"):
                        C = np.asarray(inst.batch_decode(S), dtype=np.uint8)
                    else:
                        C = np.zeros((shots, nq), dtype=np.uint8)
                        for i in range(shots):
                            C[i] = np.asarray(inst.decode(S[i]), dtype=np.uint8)
                    dt = time.perf_counter() - t0

                    if C.ndim == 1:
                        C = C.reshape(shots, -1)

                    faithful = int(np.sum(np.all(((C @ H.T) & 1) == S, axis=1)))
                    R = (C ^ E).astype(np.uint8)
                    flips = (R @ L.T) & 1
                    fails = int(np.sum(np.any(flips, axis=1)))
                    lo, hi = wilson_ci(fails, shots)
                    dec_per_s = shots / dt if dt > 0 else 0.0
                    latency_us = (dt / shots) * 1e6 if shots > 0 else 0.0

                    entry = {
                        "distance": d,
                        "n_qubits": nq,
                        "n_checks": nc,
                        "shots": shots,
                        "decoder": dec_name,
                        "logical_failures": fails,
                        "ler": fails / shots,
                        "ci95": (round(lo, 6), round(hi, 6)),
                        "faithful_pct": 100.0 * faithful / shots,
                        "seconds": round(dt, 4),
                        "decodes_per_s": round(dec_per_s, 1),
                        "latency_us": round(latency_us, 2),
                    }
                    results.append(entry)
                    print(
                        f"  d={d:2d} ({nq:3d}q) | shots={shots:5d} | {dec_name:<16} | LER={entry['ler']:.5f} | "
                        f"Faithful={entry['faithful_pct']:5.1f}% | {entry['decodes_per_s']:9,.0f} dec/s | {entry['latency_us']:7.2f} us/dec"
                    )
                except Exception as exc:
                    print(f"  d={d:2d} | shots={shots:5d} | {dec_name:<16} | ERROR: {exc}")
                    results.append({"distance": d, "shots": shots, "decoder": dec_name, "error": str(exc)})

    return {"p": p, "seed": seed, "results": results}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--distances", type=int, nargs="+", default=[3, 5, 7, 9, 11, 13, 15, 17, 19])
    ap.add_argument("--shots", type=int, nargs="+", default=[1000, 20000])
    ap.add_argument("--p", type=float, default=0.005)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out", default="benchmark_results.json")
    args = ap.parse_args()

    print(f"Starting Extensive QECTOR v3 Benchmark...")
    print(f"Distances: {args.distances}, Shots: {args.shots}, p={args.p}, seed={args.seed}\n")

    res = benchmark_run(args.distances, args.shots, args.p, args.seed)

    with open(args.out, "w", encoding="utf-8") as fh:
        json.dump(res, fh, indent=2)
    print(f"\nWrote benchmark results to {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
