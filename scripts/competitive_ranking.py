#!/usr/bin/env python
"""Head-to-head empirical ranking benchmark: QECTOR decoders vs PyMatching & BeliefMatching.

Delegates to ``run_competitive_suite`` (circuit-level pipeline, same noise model for all
decoders) so the comparison is honest and defensible.
"""
from __future__ import annotations

import argparse
import os
import sys
import time

_REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(_REPO, "python"))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from _provenance import write_artifact
from qector_decoder_v3.ler import run_competitive_suite


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--distances", type=int, nargs="+", default=[3, 5, 7, 9, 11, 13, 15, 17, 19])
    ap.add_argument("--shots", type=int, default=10000)
    ap.add_argument("--p", type=float, default=0.005)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out", default="competitive_results.json")
    ap.add_argument(
        "--decoders",
        nargs="+",
        default=["qector_blossom", "qector_belief", "qector_unionfind", "pymatching"],
    )
    args = ap.parse_args()

    print("Running Competitive Market Ranking Benchmark...")
    t0 = time.perf_counter()
    res = run_competitive_suite(
        p=args.p,
        shots=args.shots,
        seed=args.seed,
        distances=tuple(args.distances),
        decoders=tuple(args.decoders),
    )
    elapsed = time.perf_counter() - t0

    # Stamped, not bare: an artifact that does not carry its own methodology is what got the
    # pre-v0.7.0 benchmark numbers withdrawn (todo6 A1-03).
    out = write_artifact(
        args.out,
        res,
        parameters={
            "p": args.p,
            "shots": args.shots,
            "seed": args.seed,
            "rounds": "distance (d rounds)",
            "distances": list(args.distances),
            "decoders": list(args.decoders),
        },
        elapsed_seconds=elapsed,
    )
    print(f"\nWrote competitive results ({len(res)} records) to {out} in {elapsed:.1f}s")


if __name__ == "__main__":
    main()
