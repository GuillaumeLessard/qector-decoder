#!/usr/bin/env python
"""Belief-matching seed x p grid: multi-seed robustness across multiple physical
error rates (not only a single p-sweep + a single fixed-p seed sweep).

For a fixed distance, every (p, seed) cell decodes real Stim shots with QECTOR
belief-matching and plain PyMatching and records the LER reduction. This shows
the belief-matching advantage holds across BOTH p and seed, not one slice.

    python scripts/belief_grid.py --distance 5 --probs 0.004 0.005 0.006 \
        --seeds 5 --shots 3000 --out benchmark_results/belief_grid
"""
from __future__ import annotations

import argparse
import json
import math
import os
import sys

_REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(_REPO, "python"))

import numpy as np  # noqa: E402

from qector_decoder_v3 import benchmarking as bm  # noqa: E402
from qector_decoder_v3.belief_matching import BeliefMatching  # noqa: E402


def wilson(k, n, z=1.959963985):
    if n == 0:
        return 0.0, 1.0
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    w = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return max(0.0, c - w), min(1.0, c + w)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--distance", type=int, default=5)
    ap.add_argument("--probs", type=float, nargs="+", default=[0.004, 0.005, 0.006])
    ap.add_argument("--seeds", type=int, default=5)
    ap.add_argument("--shots", type=int, default=3000)
    ap.add_argument("--out", default="benchmark_results/belief_grid")
    args = ap.parse_args()

    import stim
    import pymatching

    env = bm.capture_environment()
    env["command"] = " ".join(sys.argv)
    env["stim_version"] = stim.__version__
    env["pymatching_version"] = pymatching.__version__

    d = args.distance
    cells = []
    summary = []
    for p in args.probs:
        circ = stim.Circuit.generated(
            "surface_code:rotated_memory_x", distance=d, rounds=d,
            after_clifford_depolarization=p,
            before_measure_flip_probability=p,
            after_reset_flip_probability=p)
        sdem = circ.detector_error_model(decompose_errors=True)
        pm = pymatching.Matching.from_detector_error_model(sdem)
        bmd = BeliefMatching.from_detector_error_model(sdem)
        pm_tot = bel_tot = wins = 0
        reds = []
        for seed in range(args.seeds):
            det, obs = circ.compile_detector_sampler(seed=seed).sample(
                shots=args.shots, separate_observables=True)
            det = det.astype(np.uint8)
            obs = obs.astype(np.uint8)
            pe = int(np.any(np.asarray(pm.decode_batch(det), np.uint8).reshape(args.shots, -1) != obs, axis=1).sum())
            be = int(np.any(np.asarray(bmd.decode_batch(det), np.uint8).reshape(args.shots, -1) != obs, axis=1).sum())
            red = 100 * (1 - (be / pe)) if pe else 0.0
            reds.append(red)
            pm_tot += pe
            bel_tot += be
            wins += int(be <= pe)
            cells.append({"p": p, "seed": seed, "pm_errors": pe, "belief_errors": be,
                          "pm_ler": pe / args.shots, "belief_ler": be / args.shots,
                          "reduction_pct": red})
            print(f"p={p:.3f} seed={seed} pm={pe/args.shots:.4f} belief={be/args.shots:.4f} "
                  f"red={red:+.1f}%", flush=True)
        n = args.seeds * args.shots
        summary.append({"p": p, "n_seeds": args.seeds, "belief_le_pm": wins,
                        "mean_reduction_pct": sum(reds) / len(reds),
                        "pooled_pm_ler": pm_tot / n, "pooled_belief_ler": bel_tot / n,
                        "pooled_reduction_pct": 100 * (1 - bel_tot / pm_tot) if pm_tot else 0.0})

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    with open(args.out + ".json", "w", encoding="utf-8") as fh:
        json.dump({"environment": env, "distance": d, "shots": args.shots,
                   "cells": cells, "summary": summary}, fh, indent=2)
    print(f"wrote {args.out}.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
