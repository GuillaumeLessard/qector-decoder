#!/usr/bin/env python
"""Extended belief-matching evidence: multi-seed robustness, p-sweep, and
rotated_memory_z (not only memory_x).

Same methodology as scripts/competitive_belief_matching.py (real Stim
circuit-level shots, QECTOR belief-matching vs plain PyMatching, logical-error
metric = predicted observable mismatch, Wilson 95% intervals) but adds the
breakdowns reviewers asked for:

  * multi-seed : the same (d, p) decoded under many independent sampler seeds,
                 so the LER reduction is shown to reproduce, not be one seed.
  * p-sweep    : d=5 across several physical error rates.
  * memory_z   : the rotated_memory_z basis, mirroring the memory_x headline.

Emits one JSON with all three sections + a Markdown summary.

    python scripts/belief_extended.py --out benchmark_results/belief_extended
"""
from __future__ import annotations

import argparse
import gc
import json
import math
import os
import sys
import time

_REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(_REPO, "python"))

import numpy as np  # noqa: E402

from qector_decoder_v3 import benchmarking as bm  # noqa: E402
from qector_decoder_v3.belief_matching import BeliefMatching  # noqa: E402
from qector_decoder_v3.pymatching_compat import Matching as QMatching  # noqa: E402


def wilson(k, n, z=1.959963985):
    if n == 0:
        return 0.0, 1.0
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    w = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return max(0.0, c - w), min(1.0, c + w)


def ler_count(pred, dets, obs):
    p = np.asarray(pred(dets), np.uint8).reshape(len(dets), -1)
    return int(np.any(p != obs, axis=1).sum())


def make_circ(stim, d, p, basis):
    return stim.Circuit.generated(
        f"surface_code:rotated_memory_{basis}", distance=d, rounds=d,
        after_clifford_depolarization=p,
        before_measure_flip_probability=p,
        after_reset_flip_probability=p,
    )


def build_decoders(sdem, pymatching):
    pm = pymatching.Matching.from_detector_error_model(sdem)
    qbm = BeliefMatching.from_detector_error_model(sdem)
    qpm = QMatching.from_detector_error_model(sdem)
    return pm, qpm, qbm


def eval_point(circ, sdem, pm, qpm, qbm, shots, seed):
    det, obs = circ.compile_detector_sampler(seed=seed).sample(
        shots=shots, separate_observables=True)
    det = det.astype(np.uint8)
    obs = obs.astype(np.uint8)
    out = {}
    for name, fn in (("pm", lambda x: pm.decode_batch(x)),
                     ("qmwpm", lambda x: qpm.decode_batch(x)),
                     ("belief", lambda x: qbm.decode_batch(x))):
        k = ler_count(fn, det, obs)
        lo, hi = wilson(k, shots)
        out[name] = {"errors": k, "ler": k / shots, "ci95": [lo, hi]}
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", default="benchmark_results/belief_extended")
    ap.add_argument("--ms-distances", type=int, nargs="+", default=[3, 5])
    ap.add_argument("--ms-seeds", type=int, default=8)
    ap.add_argument("--ms-shots", type=int, default=3000)
    ap.add_argument("--ms-noise", type=float, default=0.005)
    ap.add_argument("--ps-distance", type=int, default=5)
    ap.add_argument("--ps-probs", type=float, nargs="+",
                    default=[0.002, 0.004, 0.005, 0.006, 0.008, 0.010])
    ap.add_argument("--ps-shots", type=int, default=4000)
    ap.add_argument("--mz-distances", type=int, nargs="+", default=[3, 5, 7])
    ap.add_argument("--mz-shots", type=int, default=8000)
    ap.add_argument("--mz-noise", type=float, default=0.005)
    args = ap.parse_args()

    import stim
    import pymatching

    env = bm.capture_environment()
    env["timestamp_unix"] = int(time.time())
    env["stim_version"] = stim.__version__
    env["pymatching_version"] = pymatching.__version__
    env["command"] = " ".join(sys.argv)

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    blob = {"environment": env, "multiseed": None, "psweep": None, "memz": None}

    def dump():
        # Incremental write: each completed section is persisted immediately so a
        # later crash never loses the earlier sections.
        with open(args.out + ".json", "w", encoding="utf-8") as fh:
            json.dump(blob, fh, indent=2)

    # --- 1. multi-seed robustness -----------------------------------------
    multiseed = {"basis": "x", "noise": args.ms_noise, "shots": args.ms_shots,
                 "n_seeds": args.ms_seeds, "rows": [], "summary": []}
    for d in args.ms_distances:
        circ = make_circ(stim, d, args.ms_noise, "x")
        sdem = circ.detector_error_model(decompose_errors=True)
        pm, qpm, qbm = build_decoders(sdem, pymatching)
        pm_tot = bel_tot = 0
        wins = le = 0
        reds = []
        for seed in range(args.ms_seeds):
            r = eval_point(circ, sdem, pm, qpm, qbm, args.ms_shots, seed)
            pm_l, bel_l = r["pm"]["ler"], r["belief"]["ler"]
            red = 100 * (1 - bel_l / pm_l) if pm_l else 0.0
            reds.append(red)
            pm_tot += r["pm"]["errors"]
            bel_tot += r["belief"]["errors"]
            wins += int(bel_l < pm_l)
            le += int(bel_l <= pm_l)
            multiseed["rows"].append(
                {"d": d, "seed": seed, "pm_ler": pm_l, "belief_ler": bel_l,
                 "reduction_pct": red})
            print(f"[multiseed] d={d} seed={seed} pm={pm_l:.4f} belief={bel_l:.4f} "
                  f"red={red:+.1f}%", flush=True)
        n = args.ms_seeds * args.ms_shots
        pooled_red = 100 * (1 - bel_tot / pm_tot) if pm_tot else 0.0
        multiseed["summary"].append(
            {"d": d, "n_seeds": args.ms_seeds, "belief_strictly_better": wins,
             "belief_le_pm": le, "mean_reduction_pct": sum(reds) / len(reds),
             "pooled_pm_ler": pm_tot / n, "pooled_belief_ler": bel_tot / n,
             "pooled_reduction_pct": pooled_red})
        del pm, qpm, qbm
        gc.collect()
    blob["multiseed"] = multiseed
    dump()

    # --- 2. p-sweep at fixed distance -------------------------------------
    psweep = {"basis": "x", "distance": args.ps_distance, "shots": args.ps_shots,
              "seed": 12345, "rows": []}
    for p in args.ps_probs:
        circ = make_circ(stim, args.ps_distance, p, "x")
        sdem = circ.detector_error_model(decompose_errors=True)
        pm, qpm, qbm = build_decoders(sdem, pymatching)
        r = eval_point(circ, sdem, pm, qpm, qbm, args.ps_shots, 12345)
        pm_l, bel_l = r["pm"]["ler"], r["belief"]["ler"]
        red = 100 * (1 - bel_l / pm_l) if pm_l else 0.0
        psweep["rows"].append(
            {"p": p, "pm_ler": pm_l, "pm_ci95": r["pm"]["ci95"],
             "belief_ler": bel_l, "belief_ci95": r["belief"]["ci95"],
             "reduction_pct": red})
        print(f"[psweep] d={args.ps_distance} p={p} pm={pm_l:.4f} "
              f"belief={bel_l:.4f} red={red:+.1f}%", flush=True)
        del pm, qpm, qbm
        gc.collect()
    blob["psweep"] = psweep
    dump()

    # --- 3. rotated_memory_z ----------------------------------------------
    memz = {"basis": "z", "noise": args.mz_noise, "shots": args.mz_shots,
            "seed": 2026, "rows": []}
    for d in args.mz_distances:
        circ = make_circ(stim, d, args.mz_noise, "z")
        sdem = circ.detector_error_model(decompose_errors=True)
        pm, qpm, qbm = build_decoders(sdem, pymatching)
        r = eval_point(circ, sdem, pm, qpm, qbm, args.mz_shots, 2026)
        pm_l, qm_l, bel_l = r["pm"]["ler"], r["qmwpm"]["ler"], r["belief"]["ler"]
        red = 100 * (1 - bel_l / pm_l) if pm_l else 0.0
        memz["rows"].append(
            {"d": d, "pm_ler": pm_l, "pm_ci95": r["pm"]["ci95"],
             "qmwpm_ler": qm_l, "belief_ler": bel_l,
             "belief_ci95": r["belief"]["ci95"], "reduction_pct": red})
        print(f"[memz] d={d} pm={pm_l:.4f} qmwpm={qm_l:.4f} belief={bel_l:.4f} "
              f"red={red:+.1f}%", flush=True)
        del pm, qpm, qbm
        gc.collect()
    blob["memz"] = memz
    dump()
    print(f"wrote {args.out}.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
