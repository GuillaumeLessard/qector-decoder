#!/usr/bin/env python
"""Native (process-RSS) memory profiling — captures the Rust/native + GPU-host
allocations that Python's tracemalloc cannot see.

For each (decoder, distance, batch) it measures the process resident-set-size
(RSS) before vs the peak during a batch decode, sampled on a background thread.
Reports the native delta per decode, complementing the Python-side
peak_python_alloc_kib in the benchmarking harness.

    python scripts/native_memory_profile.py --distances 5 9 13 --batch 16384
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import threading
import time

_REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(_REPO, "python"))

import numpy as np  # noqa: E402

import qector_decoder_v3 as qd  # noqa: E402
from qector_decoder_v3 import codes, benchmarking as bm  # noqa: E402

try:
    import psutil
    _PROC = psutil.Process()
except Exception:
    psutil = None
    _PROC = None


def _rss_mib():
    return _PROC.memory_info().rss / (1024 * 1024) if _PROC else float("nan")


class _PeakSampler(threading.Thread):
    def __init__(self, interval=0.002):
        super().__init__(daemon=True)
        self.interval = interval
        self._stopev = threading.Event()  # NOT _stop (collides with Thread._stop)
        self.peak = 0.0

    def run(self):
        while not self._stopev.is_set():
            self.peak = max(self.peak, _rss_mib())
            time.sleep(self.interval)

    def stop(self):
        self._stopev.set()
        self.join()


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--distances", type=int, nargs="+", default=[5, 9, 13])
    ap.add_argument("--batch", type=int, default=16384)
    ap.add_argument("--error-rate", type=float, default=0.05)
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--out", default="benchmark_results/native_memory")
    args = ap.parse_args()

    if psutil is None:
        print("psutil not available; cannot measure native RSS")
        return 1

    env = bm.capture_environment()
    env["command"] = " ".join(sys.argv)
    rng = np.random.default_rng(args.seed)
    rows = []

    backends = [
        ("cpu_batch", lambda c2q, nq: qd.CPUBatchDecoder(c2q, nq)),
        ("blossom", lambda c2q, nq: qd.BlossomDecoder(c2q, nq)),
        ("fast_union_find", lambda c2q, nq: qd.FastUnionFindDecoder(c2q, nq)),
    ]
    if qd.CUDABatchDecoder.is_available():
        backends.append(("cuda_batch", lambda c2q, nq: qd.CUDABatchDecoder(c2q, nq)))

    for d in args.distances:
        code = codes.rotated_surface_code(d)
        H = code.parity_check_matrix().astype(np.uint8)
        c2q, nq = code.check_to_qubits, code.n_qubits
        err = (rng.random((args.batch, nq)) < args.error_rate).astype(np.uint8)
        syn = ((err @ H.T) & 1).astype(np.uint8)
        for name, build in backends:
            dec = build(c2q, nq)
            # warm + settle
            if name in ("blossom",):
                _ = dec.batch_decode(syn[:256].reshape(-1), 256) if False else dec.batch_decode(syn[:256])
            else:
                dec.batch_decode(syn[:256])
            time.sleep(0.05)
            base = _rss_mib()
            samp = _PeakSampler()
            samp.start()
            dec.batch_decode(syn)
            samp.stop()
            rows.append({
                "decoder": name, "distance": d, "n_qubits": nq,
                "n_checks": code.n_checks, "batch": args.batch,
                "rss_base_mib": round(base, 2),
                "rss_peak_mib": round(samp.peak, 2),
                "native_delta_mib": round(max(samp.peak - base, 0.0), 2),
            })
            print(f"d={d:2d} {name:16s} base={base:8.1f}MiB peak={samp.peak:8.1f}MiB "
                  f"native_delta={max(samp.peak-base,0):7.2f}MiB", flush=True)

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    with open(args.out + ".json", "w", encoding="utf-8") as fh:
        json.dump({"environment": env, "batch": args.batch, "results": rows}, fh, indent=2)
    print(f"wrote {args.out}.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
