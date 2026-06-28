#!/usr/bin/env python
"""Device VRAM profiling for the CUDA batch decoder (nvidia-smi), complementing
the host-RSS native_memory_profile.py. Samples GPU memory.used before vs during
a sustained large-batch decode to report the device-side memory the kernel uses.

    python scripts/vram_profile.py --distances 7 11 --batch 65536
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import threading
import time

_REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
import importlib.util
if importlib.util.find_spec("qector_decoder_v3.qector_decoder_v3") is None:
    sys.path.insert(0, os.path.join(_REPO, "python"))

import numpy as np  # noqa: E402

import qector_decoder_v3 as qd  # noqa: E402
from qector_decoder_v3 import codes  # noqa: E402


def _vram_used_mib():
    try:
        out = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader,nounits"],
            stderr=subprocess.DEVNULL).decode().strip().splitlines()
        return float(out[0])
    except Exception:
        return float("nan")


class _VramSampler(threading.Thread):
    def __init__(self, interval=0.05):
        super().__init__(daemon=True)
        self.interval = interval
        self._stopev = threading.Event()
        self.peak = 0.0

    def run(self):
        while not self._stopev.is_set():
            v = _vram_used_mib()
            if v == v:  # not NaN
                self.peak = max(self.peak, v)
            time.sleep(self.interval)

    def stop(self):
        self._stopev.set()
        self.join()


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--distances", type=int, nargs="+", default=[7, 11])
    ap.add_argument("--batch", type=int, default=65536)
    ap.add_argument("--error-rate", type=float, default=0.05)
    ap.add_argument("--hold-seconds", type=float, default=2.0)
    ap.add_argument("--out", default="benchmark_results/vram_profile")
    args = ap.parse_args()

    if not qd.CUDABatchDecoder.is_available():
        print("CUDA not available; cannot profile VRAM")
        return 1

    probe = qd.CUDABatchDecoder([[0]], 1)
    device = probe.device_name
    if np.isnan(_vram_used_mib()):
        print("nvidia-smi not available; cannot read VRAM")
        return 1

    rng = np.random.default_rng(1)
    rows = []
    for d in args.distances:
        code = codes.rotated_surface_code(d)
        c2q, nq = code.check_to_qubits, code.n_qubits
        dec = qd.CUDABatchDecoder(c2q, nq)
        err = (rng.random((args.batch, nq)) < args.error_rate).astype(np.uint8)
        H = code.parity_check_matrix().astype(np.uint8)
        syn = ((err @ H.T) & 1).astype(np.uint8)
        dec.batch_decode(syn)  # warm
        time.sleep(0.3)
        base = _vram_used_mib()
        samp = _VramSampler()
        samp.start()
        t_end = time.perf_counter() + args.hold_seconds
        while time.perf_counter() < t_end:
            dec.batch_decode(syn)
        samp.stop()
        rows.append({"distance": d, "n_qubits": nq, "batch": args.batch,
                     "vram_base_mib": round(base, 1), "vram_peak_mib": round(samp.peak, 1),
                     "vram_delta_mib": round(max(samp.peak - base, 0.0), 1)})
        print(f"d={d:2d} batch={args.batch} VRAM base={base:.0f}MiB peak={samp.peak:.0f}MiB "
              f"delta={max(samp.peak-base,0):.0f}MiB", flush=True)

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    with open(args.out + ".json", "w", encoding="utf-8") as fh:
        json.dump({"device": device, "batch": args.batch, "results": rows}, fh, indent=2)
    print(f"wrote {args.out}.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
