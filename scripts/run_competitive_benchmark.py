#!/usr/bin/env python
"""
Reproducible competitive benchmark driver for QECTOR Decoder v3.

Sweeps decoders x code-distances, separates the cold (construction) path from the
hot (decode-only) path, reports full tail-latency statistics, captures the exact
environment, and writes machine-readable JSON + CSV artifacts.  When PyMatching is
installed it also records a weight-optimality cross-check.

Usage
-----
    python scripts/run_competitive_benchmark.py \
        --code rotated_surface --distances 3 5 7 9 \
        --decoders blossom sparse_blossom union_find \
        --trials 5000 --warmup 500 --seed 1 \
        --out benchmark_results/competitive

Outputs ``<out>.json`` and ``<out>.csv``.  Every number is reproducible from the
seed + recorded environment.
"""
from __future__ import annotations

import argparse
import os
import sys
import time

# Make the in-repo package importable without installation.
_REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
import importlib.util
if importlib.util.find_spec("qector_decoder_v3.qector_decoder_v3") is None:
    sys.path.insert(0, os.path.join(_REPO, "python"))

import numpy as np  # noqa: E402

from qector_decoder_v3 import codes  # noqa: E402
from qector_decoder_v3 import benchmarking as bm  # noqa: E402


CODE_BUILDERS = {
    "repetition": codes.repetition_code,
    "ring": codes.ring_code,
    "rotated_surface": codes.rotated_surface_code,
    "unrotated_surface": codes.unrotated_surface_code,
    "toric": codes.toric_code,
    "heavy_hex": codes.heavy_hex_code,
}


def pymatching_optimality(code, p, seed, n) -> dict | None:
    """Fraction of shots where QECTOR's matching weight <= PyMatching's."""
    try:
        import pymatching
        from qector_decoder_v3.pymatching_compat import Matching
    except Exception:
        return None
    H = code.parity_check_matrix()
    qm = Matching.from_check_matrix(H)
    rm = pymatching.Matching.from_check_matrix(H)
    rng = np.random.default_rng(seed)
    worse = 0
    for _ in range(n):
        e = (rng.random(code.n_qubits) < p).astype(np.uint8)
        s = (H @ e) & 1
        cq = np.asarray(qm.decode(s)).astype(np.uint8)
        cr = np.asarray(rm.decode(s)).astype(np.uint8)
        if int(cq.sum()) > int(cr.sum()):
            worse += 1
    return {"shots": n, "qector_heavier": worse, "qector_weight_optimal_fraction": 1 - worse / n}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--code", default="rotated_surface", choices=list(CODE_BUILDERS))
    ap.add_argument("--distances", type=int, nargs="+", default=[3, 5, 7])
    ap.add_argument(
        "--decoders",
        nargs="+",
        default=["union_find", "blossom", "sparse_blossom"],
    )
    ap.add_argument("--trials", type=int, default=2000)
    ap.add_argument("--warmup", type=int, default=200)
    ap.add_argument("--error-rate", type=float, default=0.08)
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--out", default="benchmark_results/competitive")
    ap.add_argument("--no-pymatching", action="store_true")
    args = ap.parse_args()

    env = bm.capture_environment()
    env["timestamp_unix"] = int(time.time())
    env["command"] = " ".join(sys.argv)

    results = []
    for d in args.distances:
        try:
            code = CODE_BUILDERS[args.code](d)
        except ValueError as exc:
            print(f"skip {args.code} d={d}: {exc}")
            continue
        for kind in args.decoders:
            print(f"benchmark {args.code} d={d} {kind} ...", flush=True)
            rep = bm.benchmark_decoder(
                kind,
                code,
                n_trials=args.trials,
                warmup=args.warmup,
                p=args.error_rate,
                seed=args.seed,
            )
            if not args.no_pymatching and code.is_matching_graph():
                opt = pymatching_optimality(code, args.error_rate, args.seed, min(500, args.trials))
                if opt is not None:
                    rep["pymatching_crosscheck"] = opt
            results.append(rep)
            lat = rep["latency_us"]
            print(
                f"  faithful={rep['syndrome_faithful']} "
                f"median={lat['median']:.2f}us p99={lat['p99']:.2f}us "
                f"cold={rep['cold_path_us']['median']:.2f}us"
            )

    report = bm.BenchmarkReport(results, environment=env)
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    report.save(args.out + ".json", args.out + ".csv")
    print(f"\nwrote {args.out}.json and {args.out}.csv ({len(results)} rows)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
