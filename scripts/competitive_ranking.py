#!/usr/bin/env python
"""Head-to-head empirical ranking benchmark: QECTOR v3 vs PyMatching, LDPC (BP-OSD/LSD), and BeliefMatching."""
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
from beliefmatching import BeliefMatching
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
from qector_decoder_v3.bposd import BpOsdDecoder as PyBpOsdDecoder
from qector_decoder_v3.ler import wilson_ci


def get_cuda_batch():
    try:
        from qector_decoder_v3 import CUDABatchDecoder
        return CUDABatchDecoder
    except Exception:
        return None


def get_cuda_bposd():
    try:
        from qector_decoder_v3 import CUDABpOsdDecoder
        return CUDABpOsdDecoder
    except Exception:
        return None


def benchmark_competitors(distances: list[int], shots_list: list[int], p: float, seed: int):
    CudaBatch = get_cuda_batch()
    CudaBpOsd = get_cuda_bposd()

    results = []

    for d in distances:
        code = codes.rotated_surface_code(d)
        H = code.parity_check_matrix().astype(np.uint8)
        L = code.logicals_matrix().astype(np.uint8)
        c2q = [list(map(int, c)) for c in code.check_to_qubits]
        nq = code.n_qubits
        nc = code.n_checks

        # Build PyMatching matching graph directly from parity check matrix H
        pm_dec = pymatching.Matching.from_check_matrix(H)

        try:
            circuit = stim.Circuit.generated("surface_code:rotated_memory_z", distance=d, rounds=1, after_clifford_depolarization=p)
            dem = circuit.detector_error_model(decompose_errors=True)
            bm_dec = BeliefMatching.from_detector_error_model(dem)
        except Exception:
            bm_dec = None

        # Build ldpc BP-OSD
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

        # Build QECTOR decoders
        q_fast_uf = FastUnionFindDecoder(c2q, nq)
        q_sparse_blossom = SparseBlossomDecoder(c2q, nq)
        q_cuda_batch = CudaBatch(c2q, nq) if CudaBatch else None
        q_cuda_bposd = CudaBpOsd(c2q, nq, p) if CudaBpOsd else None

        for shots in shots_list:
            rng = np.random.default_rng(seed)
            E = (rng.random((shots, nq)) < p).astype(np.uint8)
            S = ((E @ H.T) & 1).astype(np.uint8)

            eval_targets = []

            # QECTOR CUDA Batch
            if q_cuda_batch:
                eval_targets.append(("QECTOR CUDA Batch (GPU)", lambda S=S: q_cuda_batch.batch_decode(S)))

            # QECTOR Fast Union-Find
            eval_targets.append(("QECTOR Fast Union-Find (CPU)", lambda S=S: q_fast_uf.batch_decode(S)))

            # QECTOR Sparse Blossom
            eval_targets.append(("QECTOR Sparse Blossom (CPU)", lambda S=S: q_sparse_blossom.batch_decode(S)))

            # PyMatching v2.4 (C++)
            eval_targets.append(("PyMatching v2.4 (C++)", lambda S=S: pm_dec.decode_batch(S)))

            # QECTOR CUDA BP-OSD
            if q_cuda_bposd and d <= 13:
                eval_targets.append(("QECTOR CUDA BP-OSD (GPU)", lambda S=S: q_cuda_bposd.batch_decode(S)))

            # ldpc BP-OSD (Joschka Roffe)
            if ldpc_bposd and d <= 9 and shots <= 5000:
                def run_ldpc_bposd(S=S):
                    out = np.zeros((shots, nq), dtype=np.uint8)
                    for i in range(shots):
                        out[i] = ldpc_bposd.decode(S[i])
                    return out
                eval_targets.append(("ldpc BP-OSD (C++)", run_ldpc_bposd))

            # BeliefMatching (Requires detector-based syndrome input matching Stim DEM)
            if bm_dec and d <= 9 and shots <= 5000:
                def run_bm(S=S):
                    out = np.zeros((shots, nq), dtype=np.uint8)
                    for i in range(shots):
                        out[i] = bm_dec.decode(S[i])
                    return out
                eval_targets.append(("BeliefMatching (BP+MWPM)", run_bm))

            for name, runner in eval_targets:
                try:
                    t0 = time.perf_counter()
                    C = np.asarray(runner(), dtype=np.uint8)
                    dt = time.perf_counter() - t0

                    if C.ndim == 1:
                        C = C.reshape(shots, -1)
                    if C.shape[1] > nq:
                        C = C[:, :nq]

                    faithful = int(np.sum(np.all(((C @ H.T) & 1) == S, axis=1)))
                    R = (C ^ E).astype(np.uint8)
                    flips = (R @ L.T) & 1
                    fails = int(np.sum(np.any(flips, axis=1)))
                    lo, hi = wilson_ci(fails, shots)
                    dec_s = shots / dt if dt > 0 else 0.0
                    lat = (dt / shots) * 1e6 if shots > 0 else 0.0

                    rec = {
                        "distance": d,
                        "n_qubits": nq,
                        "shots": shots,
                        "decoder": name,
                        "fails": fails,
                        "ler": fails / shots,
                        "ci95": (round(lo, 6), round(hi, 6)),
                        "faith_pct": 100.0 * faithful / shots,
                        "seconds": round(dt, 4),
                        "dec_per_s": round(dec_s, 1),
                        "latency_us": round(lat, 2),
                        "status": "success",
                    }
                    results.append(rec)
                    print(f"d={d:2d} | shots={shots:6d} | {name:<30} | LER={rec['ler']:.5f} | {rec['dec_per_s']:10,.0f} dec/s | {rec['latency_us']:7.2f} us/dec")
                except Exception as exc:
                    print(f"d={d:2d} | shots={shots:6d} | {name:<30} | FAILED/UNSUPPORTED: {exc}")
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
    ap = argparse.ArgumentParser()
    ap.add_argument("--distances", type=int, nargs="+", default=[3, 5, 7, 9, 11, 13, 15, 17, 19])
    ap.add_argument("--shots", type=int, nargs="+", default=[1000, 10000])
    ap.add_argument("--p", type=float, default=0.005)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out", default="competitive_results.json")
    args = ap.parse_args()

    print("Running Competitive Market Ranking Benchmark...")
    res = benchmark_competitors(args.distances, args.shots, args.p, args.seed)

    with open(args.out, "w", encoding="utf-8") as fh:
        json.dump(res, fh, indent=2)
    print(f"\nWrote competitive results to {args.out}")


if __name__ == "__main__":
    main()
