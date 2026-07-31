#!/usr/bin/env python
"""Ultra-rigorous, high-shot hardware benchmark suite via ``run_competitive_suite``.

Executes the canonical circuit-level competitive pipeline with thread environment
isolation (OMP_NUM_THREADS=1) for reproducible micro-benchmarking.
"""
from __future__ import annotations

import argparse
import json
import os
import sys

_REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(_REPO, "python"))

# Isolate CPU threads for reproducible micro-benchmarking
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"

from qector_decoder_v3.ler import run_competitive_suite


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--distances", nargs="+", type=int, default=[3, 5, 7, 9, 11, 13, 15, 17, 19])
    parser.add_argument("--shots", type=int, default=100000)
    parser.add_argument("--p", type=float, default=0.005)
    parser.add_argument("--reps", type=int, default=3)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--out", type=str, default="competitive_results.json")
    parser.add_argument(
        "--decoders",
        nargs="+",
        default=["qector_blossom", "qector_belief", "qector_unionfind", "pymatching"],
    )
    args = parser.parse_args()

    print("=== QECTOR Ultra-Rigorous Hardware Benchmark (via run_competitive_suite) ===")
    print(f"Distances: {args.distances}")
    print(f"Shots: {args.shots}")
    print(f"Repetitions per config: {args.reps}")
    print(f"Noise rate: {args.p}")
    print("=" * 60)

    all_results = []
    for rep in range(args.reps):
        s = args.seed + rep * 101
        res = run_competitive_suite(
            p=args.p,
            shots=args.shots,
            seed=s,
            distances=tuple(args.distances),
            decoders=tuple(args.decoders),
        )
        for r in res:
            r["rep"] = rep + 1
        all_results.extend(res)
        print(f"  rep {rep + 1}/{args.reps} (seed={s}) done ({len(res)} records)")

    with open(args.out, "w", encoding="utf-8") as fh:
        json.dump(all_results, fh, indent=2)
    print(f"\nWrote ultra-rigorous benchmark results ({len(all_results)} records) to {args.out}")


if __name__ == "__main__":
    main()
