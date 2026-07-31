#!/usr/bin/env python
"""Extensive multi-family benchmark for QECTOR decoders via ``run_competitive_suite``.

Runs the canonical circuit-level competitive suite across distances and shot counts.
Saves structured JSON results for report generation.
"""
from __future__ import annotations

import argparse
import json
import os
import sys

_REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(_REPO, "python"))

from qector_decoder_v3.ler import run_competitive_suite


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--distances", type=int, nargs="+", default=[3, 5, 7, 9, 11, 13, 15, 17, 19])
    ap.add_argument("--shots", type=int, default=20000)
    ap.add_argument("--p", type=float, default=0.005)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out", default="benchmark_results.json")
    ap.add_argument(
        "--decoders",
        nargs="+",
        default=["qector_blossom", "qector_belief", "qector_unionfind", "pymatching"],
    )
    args = ap.parse_args()

    print("Starting Extensive QECTOR Benchmark via run_competitive_suite...")
    print(f"Distances: {args.distances}, Shots: {args.shots}, p={args.p}, seed={args.seed}\n")

    res = run_competitive_suite(
        p=args.p,
        shots=args.shots,
        seed=args.seed,
        distances=tuple(args.distances),
        decoders=tuple(args.decoders),
    )

    output = {"p": args.p, "seed": args.seed, "results": res}
    with open(args.out, "w", encoding="utf-8") as fh:
        json.dump(output, fh, indent=2)
    print(f"\nWrote benchmark results to {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
