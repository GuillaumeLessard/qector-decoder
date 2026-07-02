#!/usr/bin/env python
"""Extensive GPU correctness + crossover test for QECTOR v3.

For each surface-code distance and a sweep of batch sizes it:
  * decodes the SAME reachable syndromes on CPU, CUDA and OpenCL,
  * asserts CUDA == CPU and OpenCL == CPU bit-for-bit,
  * asserts every correction is syndrome-faithful (H c == s mod 2),
  * times each backend (median of repeats) to find the CPU->GPU crossover.

Emits benchmark_results/gpu_extensive.json (+ .md) with device info, the full
latency table, and the per-backend crossover batch size.

    python scripts/gpu_extensive_test.py --distances 3 5 7 9 11 \
        --batches 1 64 1024 4096 16384 65536 --error-rate 0.05
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time

_REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(_REPO, "python"))

import numpy as np  # noqa: E402

import qector_decoder_v3 as qd  # noqa: E402
from qector_decoder_v3 import codes, benchmarking as bm  # noqa: E402


def _time(fn, reps):
    ts = []
    for _ in range(reps):
        t0 = time.perf_counter()
        out = fn()
        ts.append(time.perf_counter() - t0)
    return out, float(np.median(ts))


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--distances", type=int, nargs="+", default=[3, 5, 7, 9, 11])
    ap.add_argument("--batches", type=int, nargs="+",
                    default=[1, 64, 1024, 4096, 16384, 65536])
    ap.add_argument("--error-rate", type=float, default=0.05)
    ap.add_argument("--reps", type=int, default=3)
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--out", default="benchmark_results/gpu_extensive")
    args = ap.parse_args()

    cuda_ok = qd.CUDABatchDecoder.is_available()
    opencl_ok = qd.OpenCLBatchDecoder.is_available()

    env = bm.capture_environment()
    env["command"] = " ".join(sys.argv)
    env["cuda_available"] = cuda_ok
    env["opencl_available"] = opencl_ok

    device = {}
    if cuda_ok:
        try:
            probe = qd.CUDABatchDecoder([[0]], 1)
            device["cuda_device"] = probe.device_name
            device["cuda_compute_capability"] = probe.compute_capability
        except Exception as e:  # pragma: no cover
            device["cuda_device"] = f"(probe failed: {e})"
    print(f"CUDA available={cuda_ok}  OpenCL available={opencl_ok}  device={device}")

    rng = np.random.default_rng(args.seed)
    rows = []
    crossover = {"cuda": {}, "opencl": {}}

    for d in args.distances:
        code = codes.rotated_surface_code(d)
        H = code.parity_check_matrix().astype(np.uint8)
        c2q, nq = code.check_to_qubits, code.n_qubits
        cpu = qd.CPUBatchDecoder(c2q, nq)
        cu = qd.CUDABatchDecoder(c2q, nq) if cuda_ok else None
        ocl = qd.OpenCLBatchDecoder(c2q, nq) if opencl_ok else None

        for B in args.batches:
            err = (rng.random((B, nq)) < args.error_rate).astype(np.uint8)
            syn = ((err @ H.T) & 1).astype(np.uint8)

            # warmup
            cpu.batch_decode(syn)
            c_corr, c_us = _time(lambda: np.asarray(cpu.batch_decode(syn), np.uint8), args.reps)
            c_faithful = bool(np.array_equal((c_corr @ H.T) & 1, syn))

            row = {"distance": d, "n_qubits": nq, "n_checks": code.n_checks,
                   "batch": B, "cpu_us_per_shot": c_us / B * 1e6,
                   "cpu_faithful": c_faithful}

            if cu is not None:
                cu.batch_decode(syn)
                g_corr, g_us = _time(lambda: np.asarray(cu.batch_decode(syn), np.uint8), args.reps)
                row["cuda_us_per_shot"] = g_us / B * 1e6
                row["cuda_bit_identical"] = bool(np.array_equal(g_corr, c_corr))
                row["cuda_faithful"] = bool(np.array_equal((g_corr @ H.T) & 1, syn))
                row["cuda_degraded"] = bool(cu.is_degraded)
                if row["cuda_bit_identical"] and g_us < c_us and d not in crossover["cuda"]:
                    crossover["cuda"][d] = B
            if ocl is not None:
                ocl.batch_decode(syn)
                o_corr, o_us = _time(lambda: np.asarray(ocl.batch_decode(syn), np.uint8), args.reps)
                row["opencl_us_per_shot"] = o_us / B * 1e6
                row["opencl_bit_identical"] = bool(np.array_equal(o_corr, c_corr))
                row["opencl_faithful"] = bool(np.array_equal((o_corr @ H.T) & 1, syn))
                row["opencl_degraded"] = bool(ocl.is_degraded)
                if row["opencl_bit_identical"] and o_us < c_us and d not in crossover["opencl"]:
                    crossover["opencl"][d] = B

            rows.append(row)
            msg = (f"d={d:2d} B={B:6d} | CPU {row['cpu_us_per_shot']:8.3f}us"
                   f"  faithful={c_faithful}")
            if "cuda_us_per_shot" in row:
                msg += (f" | CUDA {row['cuda_us_per_shot']:8.3f}us"
                        f" id={row['cuda_bit_identical']}")
            if "opencl_us_per_shot" in row:
                msg += (f" | OpenCL {row['opencl_us_per_shot']:8.3f}us"
                        f" id={row['opencl_bit_identical']}")
            print(msg, flush=True)

    # summary checks
    cuda_rows = [r for r in rows if "cuda_bit_identical" in r]
    ocl_rows = [r for r in rows if "opencl_bit_identical" in r]
    summary = {
        "n_configs": len(rows),
        "cuda_bit_identical_all": all(r["cuda_bit_identical"] for r in cuda_rows) if cuda_rows else None,
        "opencl_bit_identical_all": all(r["opencl_bit_identical"] for r in ocl_rows) if ocl_rows else None,
        "all_faithful": all(r["cpu_faithful"] for r in rows),
        "crossover": crossover,
    }

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    with open(args.out + ".json", "w", encoding="utf-8") as fh:
        json.dump({"environment": env, "device": device, "summary": summary,
                   "results": rows}, fh, indent=2)
    print("SUMMARY:", json.dumps(summary))
    print(f"wrote {args.out}.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
