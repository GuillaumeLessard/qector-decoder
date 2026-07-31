#!/usr/bin/env python
"""Multi-run, statistically averaged competitive benchmark via ``run_competitive_suite``.

All decoders run under the same circuit-level Stim pipeline — the comparison is honest.
The ``--reps`` argument runs the suite multiple times with incremented seeds.
"""
from __future__ import annotations

import argparse
import json
import os
import sys

_REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(_REPO, "python"))

from qector_decoder_v3.ler import run_competitive_suite


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--distances", nargs="+", type=int, default=[3, 5, 7, 9, 11, 13, 15, 17, 19])
    parser.add_argument("--shots", type=int, default=10000)
    parser.add_argument("--p", type=float, default=0.005)
    parser.add_argument("--reps", type=int, default=5)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--out", type=str, default="competitive_results.json")
    parser.add_argument(
        "--decoders",
        nargs="+",
        default=["qector_blossom", "qector_belief", "qector_unionfind", "pymatching"],
    )
    args = parser.parse_args()

    print("Running Multi-Run Competitive Market Ranking Benchmark...")
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
        all_results.extend(res)
        print(f"  rep {rep + 1}/{args.reps} (seed={s}) done ({len(res)} records)")

    with open(args.out, "w", encoding="utf-8") as fh:
        json.dump(all_results, fh, indent=2)
    print(f"\nWrote multi-run benchmark results ({len(all_results)} records) to {args.out}")


if __name__ == "__main__":
    main()
