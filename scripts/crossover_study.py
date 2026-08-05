#!/usr/bin/env python
"""
CPU vs GPU batch-decode break-even study (dev2todo §3.1).

Measures where ``CPUBatchDecoder`` (Rayon) stops being faster than
``CUDABatchDecoder`` as the batch (shot count) grows, for a distance sweep of
repetition-code memory DEMs, and prints the measured/interpolated
break-even table used by :func:`qector_decoder_v3.recommend_backend`.

Run under the Enterprise dev environment for the CUDA path::

    dev.bat .venv\\Scripts\\python.exe scripts\\crossover_study.py
    dev.bat .venv\\Scripts\\python.exe scripts\\crossover_study.py \\
        --distances 3 5 9 13 --ladder 1000 10000 100000 1000000

Without a license the CUDA leg is skipped and the script reports the CPU-only
baseline (the JSON output is still valid; ``cuda`` entries are null).
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
from qector_decoder_v3 import dem  # noqa: E402

_DEFAULT_DISTANCES = [3, 5, 7, 9, 11, 13, 15, 17, 19, 21]
_DEFAULT_LADDER = [1_000, 10_000, 30_000, 100_000, 300_000, 1_000_000]


def _dem_problem(distance: int, p: float = 0.005):
    import stim

    circ = stim.Circuit.generated(
        "repetition_code:memory",
        rounds=1,
        distance=distance,
        before_measure_flip_probability=p,
        after_reset_flip_probability=p,
        after_clifford_depolarization=p,
    )
    sdem = circ.detector_error_model(decompose_errors=True)
    graph = dem.from_stim(sdem).collapse_to_graph()
    return graph.check_to_qubits(), graph.num_errors, graph.weights().tolist()


def _best_time(fn, repeats: int = 3) -> float:
    best = float("inf")
    for _ in range(repeats):
        t0 = time.perf_counter()
        fn()
        best = min(best, time.perf_counter() - t0)
    return best


def main() -> int:
    ap = argparse.ArgumentParser(description="CPU vs GPU batch-decode break-even study")
    ap.add_argument("--distances", type=int, nargs="+", default=_DEFAULT_DISTANCES)
    ap.add_argument("--ladder", type=int, nargs="+", default=_DEFAULT_LADDER)
    ap.add_argument("--repeats", type=int, default=3)
    ap.add_argument("--noise", type=float, default=0.005)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default="benchmark_results/crossover_study.json")
    args = ap.parse_args()

    rng = np.random.default_rng(args.seed)
    gpu_dec = None
    gpu_family = None
    try:
        gpu_dec = qd.CUDABatchDecoder([[0, 1]], 2, [1.0, 1.0])
        gpu_dec.batch_decode(np.zeros((1, 1), dtype=np.uint8))
        gpu_family = "cuda"
    except Exception:
        gpu_dec = None

    timings: dict[str, dict] = {"cpu": {}, "gpu": {}}
    break_even: dict[str, int | None] = {}
    for d in args.distances:
        c2q, nq, w = _dem_problem(d, args.noise)
        n_checks = len(c2q)
        cpu = qd.BatchDecoder(c2q, nq)  # Rayon parallel batch, as Backend.calibrate
        gpu = None
        if gpu_dec is not None:
            try:
                gpu = qd.CUDABatchDecoder(c2q, nq, w)
            except Exception:
                gpu = None
        timings["cpu"][str(d)] = {}
        if gpu is not None:
            timings["gpu"][str(d)] = {}
        first = None
        for n in args.ladder:
            syn = (rng.random((n, n_checks)) < 0.08).astype(np.uint8)
            cpu_t = _best_time(lambda s=syn, _cpu=cpu: _cpu.parallel_batch_decode(s), args.repeats)
            timings["cpu"][str(d)][str(n)] = cpu_t
            if gpu is not None:
                try:
                    gpu_t = _best_time(lambda s=syn, _gpu=gpu: _gpu.batch_decode(s), args.repeats)
                    timings["gpu"][str(d)][str(n)] = gpu_t
                    if first is None and gpu_t < cpu_t:
                        first = n
                except Exception:
                    gpu = None
                    timings["gpu"][str(d)] = {}
                    break
            print(
                f"d={d:>3} shots={n:>9}  cpu={cpu_t * 1e3:8.3f} ms"
                + (f"  gpu={timings['gpu'][str(d)][str(n)] * 1e3:8.3f} ms" if gpu is not None else "  gpu=n/a")
            )
        break_even[str(d)] = first
        print(f"  -> d={d} break-even: {first}")

    out = {
        "machine": os.environ.get("COMPUTERNAME", "unknown"),
        "gpu": qd.detect_hardware().as_dict(),
        "gpu_family": gpu_family,
        "timings_ms": timings,
        "break_even_shots": break_even,
    }
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as fh:
        json.dump(out, fh, indent=2)
    print(f"\nwrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
