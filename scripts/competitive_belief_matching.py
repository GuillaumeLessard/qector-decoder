#!/usr/bin/env python
"""
Belief-matching head-to-head: QECTOR belief-matching vs plain PyMatching.

Real Stim circuit-level shots. Shows that QECTOR's belief-matching (BP-reweighted
exact MWPM) achieves a LOWER logical error rate than plain MWPM/PyMatching, with
Wilson 95% intervals. The reference `beliefmatching` package is included when
installed as an upper-accuracy reference.

Usage:
    python scripts/competitive_belief_matching.py --distances 3 5 7 \
        --noise 0.005 --shots 20000 --out benchmark_results/competitive_belief
"""
from __future__ import annotations

import argparse
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


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--distances", type=int, nargs="+", default=[3, 5, 7])
    ap.add_argument("--noise", type=float, default=0.005)
    ap.add_argument("--shots", type=int, default=20000)
    ap.add_argument("--out", default="benchmark_results/competitive_belief")
    ap.add_argument("--no-ref", action="store_true",
                    help="skip the reference beliefmatching package (much faster at large d)")
    args = ap.parse_args()

    import stim
    import pymatching

    if args.no_ref:
        have_ref = False
    else:
        try:
            from beliefmatching import BeliefMatching as RefBM
            have_ref = True
        except Exception:
            have_ref = False

    env = bm.capture_environment()
    env["timestamp_unix"] = int(time.time())
    env["stim_version"] = stim.__version__
    env["pymatching_version"] = pymatching.__version__
    env["command"] = " ".join(sys.argv)

    rows = []
    for d in args.distances:
        circ = stim.Circuit.generated(
            "surface_code:rotated_memory_x", distance=d, rounds=d,
            after_clifford_depolarization=args.noise,
            before_measure_flip_probability=args.noise,
            after_reset_flip_probability=args.noise,
        )
        sdem = circ.detector_error_model(decompose_errors=True)
        det, obs = circ.compile_detector_sampler().sample(
            shots=args.shots, separate_observables=True)
        det = det.astype(np.uint8)
        obs = obs.astype(np.uint8)

        pm = pymatching.Matching.from_detector_error_model(sdem)
        qpm = QMatching.from_detector_error_model(sdem)
        qbm = BeliefMatching.from_detector_error_model(sdem)

        row = {"distance": d, "noise": args.noise, "shots": args.shots,
               "detectors": sdem.num_detectors}
        for name, fn in [
            ("pymatching", lambda x: pm.decode_batch(x)),
            ("qector_mwpm", lambda x: qpm.decode_batch(x)),
            ("qector_belief", lambda x: qbm.decode_batch(x)),
        ] + ([("ref_belief", lambda x: RefBM.from_detector_error_model(sdem).decode_batch(x))]
             if have_ref else []):
            t0 = time.perf_counter()
            k = ler_count(fn, det, obs)
            dt = time.perf_counter() - t0
            lo, hi = wilson(k, args.shots)
            row[name] = {"ler": k / args.shots, "ler_ci95": [lo, hi],
                         "us_per_shot": dt / args.shots * 1e6}
        rows.append(row)
        bmr = row["qector_belief"]["ler"]
        pmr = row["pymatching"]["ler"]
        print(f"d={d}: pymatching={pmr:.4f}  QECTOR-belief={bmr:.4f}  "
              f"reduction={100*(1-bmr/pmr) if pmr else 0:.1f}%", flush=True)

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    with open(args.out + ".json", "w", encoding="utf-8") as fh:
        json.dump({"environment": env, "results": rows}, fh, indent=2)
    _write_md(args.out + ".md", env, rows, have_ref)
    print(f"wrote {args.out}.json and {args.out}.md")
    return 0


def _write_md(path, env, rows, have_ref):
    cols = "| d | shots | PyMatching LER | QECTOR-MWPM LER | QECTOR-belief LER | LER reduction |"
    sep = "|---|-------|----------------|-----------------|-------------------|---------------|"
    if have_ref:
        cols = cols[:-1] + " ref-belief LER |"
        sep = sep[:-1] + "----------------|"
    lines = [
        "# Belief-matching head-to-head — QECTOR vs PyMatching (circuit-level)",
        "",
        f"- `surface_code:rotated_memory_x`, rounds=d, circuit noise p={rows[0]['noise'] if rows else '?'}",
        f"- shots/point: {rows[0]['shots'] if rows else '?'}; Wilson 95% intervals",
        f"- {env.get('processor') or env.get('platform')}; Stim {env.get('stim_version')}; "
        f"PyMatching {env.get('pymatching_version')}",
        "",
        cols, sep,
    ]
    for r in rows:
        pm, qm, qb = r["pymatching"], r["qector_mwpm"], r["qector_belief"]
        red = 100 * (1 - qb["ler"] / pm["ler"]) if pm["ler"] else 0.0
        line = (f"| {r['distance']} | {r['shots']} | "
                f"{pm['ler']:.4f} [{pm['ler_ci95'][0]:.4f},{pm['ler_ci95'][1]:.4f}] | "
                f"{qm['ler']:.4f} | "
                f"{qb['ler']:.4f} [{qb['ler_ci95'][0]:.4f},{qb['ler_ci95'][1]:.4f}] | "
                f"{red:.1f}% |")
        if have_ref and "ref_belief" in r:
            line = line[:-1] + f" {r['ref_belief']['ler']:.4f} |"
        lines.append(line)
    with open(path, "w", encoding="utf-8") as fh:
        fh.write("\n".join(lines) + "\n")


if __name__ == "__main__":
    raise SystemExit(main())
