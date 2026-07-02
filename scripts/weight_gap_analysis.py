#!/usr/bin/env python
"""Weight-gap analysis for QECTOR-Blossom vs exact MWPM at large distance.

For each distance it decodes N shots and records, per shot, the QECTOR matching
weight minus the optimal (PyMatching on the identical collapsed graph) weight,
together with the defect count (syndrome weight). Emits:

  * a binned histogram of the weight gap per distance, and
  * a (defect-count, excess-weight) scatter sample,

so the gap distribution and its growth with defect count are visible. Run it
before and after a decoder change to show the effect.

    python scripts/weight_gap_analysis.py --distances 13 15 17 --shots 20000 \
        --out benchmark_results/weight_gap_analysis
"""
from __future__ import annotations

import argparse
import json
import os
import sys

_REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(_REPO, "python"))

import numpy as np  # noqa: E402

from qector_decoder_v3 import dem, pymatching_compat  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--distances", type=int, nargs="+", default=[13, 15, 17])
    ap.add_argument("--basis", default="x", choices=["x", "z"])
    ap.add_argument("--noise", type=float, default=0.005)
    ap.add_argument("--shots", type=int, default=20000)
    ap.add_argument("--seed", type=int, default=20260622)
    ap.add_argument("--scatter-cap", type=int, default=3000)
    ap.add_argument("--out", default="benchmark_results/weight_gap_analysis")
    args = ap.parse_args()

    import stim
    import pymatching

    out = {"basis": args.basis, "noise": args.noise, "shots": args.shots,
           "seed": args.seed, "per_distance": []}

    for d in args.distances:
        circ = stim.Circuit.generated(
            f"surface_code:rotated_memory_{args.basis}", distance=d, rounds=d,
            after_clifford_depolarization=args.noise,
            before_measure_flip_probability=args.noise,
            after_reset_flip_probability=args.noise)
        sdem = circ.detector_error_model(decompose_errors=True)
        raw = dem.from_stim(sdem)
        model = raw.collapse_to_graph()
        H = np.asarray(model.check_matrix())
        w = np.asarray(model.weights(), float)
        qm = pymatching_compat.Matching.from_detector_error_model(sdem)
        pmc = pymatching.Matching.from_check_matrix(H, weights=w)

        det, _ = circ.compile_detector_sampler(seed=args.seed).sample(
            shots=args.shots, separate_observables=True)
        det = det.astype(np.uint8)

        sw = det.sum(1).astype(int)
        deltas = np.empty(args.shots, float)
        for i in range(args.shots):
            cq = np.asarray(qm.decode_to_edges_array(det[i])).astype(bool)
            wq = float(w[cq].sum())
            _, wc = pmc.decode(det[i], return_weight=True)
            deltas[i] = wq - wc

        pos = deltas[deltas > 1e-6]
        # histogram of the (positive) weight gap
        edges = [0, 1e-6, 1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1e9]
        hist, _ = np.histogram(deltas, bins=edges)
        # scatter sample of (defect_count, excess_weight)
        idx = np.arange(args.shots)
        if args.shots > args.scatter_cap:
            rng = np.random.default_rng(0)
            idx = rng.choice(args.shots, args.scatter_cap, replace=False)
        scatter = [[int(sw[i]), round(float(max(deltas[i], 0.0)), 3)] for i in idx]

        rec = {
            "distance": d, "edges": int(model.num_errors),
            "detectors": int(model.num_detectors),
            "defect_count_mean": float(sw.mean()),
            "defect_count_range": [int(sw.min()), int(sw.max())],
            "heavier_fraction": float(np.mean(deltas > 1e-6)),
            "delta_mean": float(deltas.mean()),
            "delta_median": float(np.median(deltas)),
            "delta_p99": float(np.percentile(deltas, 99)),
            "delta_max": float(deltas.max()),
            "pos_delta_median": float(np.median(pos)) if pos.size else 0.0,
            "hist_bin_edges": edges,
            "hist_counts": [int(x) for x in hist],
            "scatter": scatter,
        }
        out["per_distance"].append(rec)
        print(f"d={d:2d} edges={rec['edges']:6d} defects~{rec['defect_count_mean']:.0f} | "
              f"heavier={rec['heavier_fraction']*100:5.1f}%  median_delta={rec['delta_median']:.3f}  "
              f"max={rec['delta_max']:.1f}", flush=True)

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    with open(args.out + ".json", "w", encoding="utf-8") as fh:
        json.dump(out, fh, indent=2)
    print(f"wrote {args.out}.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
