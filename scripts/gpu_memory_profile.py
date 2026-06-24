#!/usr/bin/env python
"""GPU host (RSS) + device (VRAM) memory profile for the batch decoders.

Complements ``native_memory_profile.py`` (host only) and ``vram_profile.py``
(device only): for each distance it measures the host RSS delta and the device
VRAM delta during a sustained large-batch decode, then — after releasing the
decoder and a short settle — re-reads VRAM to confirm the device memory returns
toward baseline (no leak across jobs).

    python scripts/gpu_memory_profile.py --distances 5 9 13 --batch 65536
"""
from __future__ import annotations

import argparse
import gc
import json
import os
import subprocess
import sys
import threading
import time

_REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(_REPO, "python"))

import numpy as np  # noqa: E402

import qector_decoder_v3 as qd  # noqa: E402
from qector_decoder_v3 import benchmarking as bm  # noqa: E402
from qector_decoder_v3 import codes  # noqa: E402


def _rss_mib() -> float:
    try:
        import psutil
        return psutil.Process().memory_info().rss / (1024 * 1024)
    except Exception:
        return float("nan")


def _vram_used_mib() -> float:
    try:
        out = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader,nounits"],
            stderr=subprocess.DEVNULL).decode().strip().splitlines()
        return float(out[0])
    except Exception:
        return float("nan")


class _Sampler(threading.Thread):
    def __init__(self, reader, interval=0.05):
        super().__init__(daemon=True)
        self.reader = reader
        self.interval = interval
        self._stopev = threading.Event()
        self.peak = 0.0

    def run(self):
        while not self._stopev.is_set():
            v = self.reader()
            if v == v:
                self.peak = max(self.peak, v)
            time.sleep(self.interval)

    def stop(self):
        self._stopev.set()
        self.join()


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--distances", type=int, nargs="+", default=[5, 9, 13])
    ap.add_argument("--batch", type=int, default=65536)
    ap.add_argument("--error-rate", type=float, default=0.05)
    ap.add_argument("--hold-seconds", type=float, default=2.0)
    ap.add_argument("--backend", default="cuda", choices=["cuda", "opencl"])
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--out", default="benchmark_results/gpu_memory_profile")
    args = ap.parse_args()

    Dec = qd.CUDABatchDecoder if args.backend == "cuda" else qd.OpenCLBatchDecoder
    avail = qd.cuda_is_available() if args.backend == "cuda" else qd.opencl_is_available()
    if not avail:
        print(f"{args.backend} not available; cannot profile GPU memory")
        return 1

    env = bm.capture_environment()
    env["timestamp_unix"] = int(time.time())
    env["command"] = "gpu_memory_profile " + " ".join(sys.argv[1:])
    have_vram = _vram_used_mib() == _vram_used_mib()

    rng = np.random.default_rng(args.seed)
    rows = []
    for d in args.distances:
        code = codes.rotated_surface_code(d)
        c2q, nq = code.check_to_qubits, code.n_qubits
        H = code.parity_check_matrix().astype(np.uint8)
        err = (rng.random((args.batch, nq)) < args.error_rate).astype(np.uint8)
        syn = ((err @ H.T) & 1).astype(np.uint8)

        gc.collect()
        rss_base = _rss_mib()
        vram_base = _vram_used_mib() if have_vram else float("nan")

        dec = Dec(c2q, nq)
        dec.batch_decode(syn)  # warm / allocate
        time.sleep(0.3)

        rss_s = _Sampler(_rss_mib)
        vram_s = _Sampler(_vram_used_mib) if have_vram else None
        rss_s.start()
        if vram_s:
            vram_s.start()
        t_end = time.perf_counter() + args.hold_seconds
        while time.perf_counter() < t_end:
            dec.batch_decode(syn)
        rss_s.stop()
        if vram_s:
            vram_s.stop()

        rss_peak = rss_s.peak
        vram_peak = vram_s.peak if vram_s else float("nan")

        # Release the decoder and re-read VRAM to check it returns toward baseline.
        del dec
        gc.collect()
        time.sleep(0.8)
        vram_after = _vram_used_mib() if have_vram else float("nan")

        row = {
            "distance": d, "n_qubits": nq, "n_checks": code.n_checks, "batch": args.batch,
            "backend": args.backend,
            "rss_base_mib": round(rss_base, 1),
            "rss_peak_mib": round(rss_peak, 1),
            "rss_delta_mib": round(max(rss_peak - rss_base, 0.0), 1),
            "vram_base_mib": None if not have_vram else round(vram_base, 1),
            "vram_peak_mib": None if not have_vram else round(vram_peak, 1),
            "vram_delta_mib": None if not have_vram else round(max(vram_peak - vram_base, 0.0), 1),
            "vram_after_release_mib": None if not have_vram else round(vram_after, 1),
            "vram_returned_to_baseline": None if not have_vram else bool(
                vram_after <= vram_base + max(64.0, 0.25 * max(vram_peak - vram_base, 1.0))),
        }
        rows.append(row)
        print(f"d={d:2d} batch={args.batch} RSS d={row['rss_delta_mib']}MiB "
              f"VRAM d={row['vram_delta_mib']}MiB returned={row['vram_returned_to_baseline']}",
              flush=True)

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    bm.write_json(args.out + ".json", {
        "environment": env, "backend": args.backend, "batch": args.batch,
        "have_vram": have_vram, "results": rows})
    print(f"wrote {args.out}.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
