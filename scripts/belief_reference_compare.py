#!/usr/bin/env python
"""Cross-validate QECTOR belief-matching against the reference ``beliefmatching``.

The headline differentiator claim is that QECTOR's belief-matching lowers the
logical error rate vs plain MWPM/PyMatching, *matching the reference
``beliefmatching`` package*.  This script decodes the **same** sampled Stim shots
with all three decoders and reports the logical error rate (with Wilson 95%
intervals) per distance:

  * ``pymatching.Matching``               — plain weighted MWPM (baseline)
  * ``qector ... BeliefMatching``         — QECTOR's BP + exact weighted MWPM
  * ``beliefmatching.BeliefMatching``     — the reference implementation

Output: JSON + a Markdown table.  Use it to prove the QECTOR belief LER tracks
the reference (not just beats PyMatching) beyond d=5.

Usage::

    python scripts/belief_reference_compare.py --distances 3 5 7 --shots 10000
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
from qector_decoder_v3.belief_matching import BeliefMatching as QBelief  # noqa: E402


def wilson(k, n, z=1.959963985):
    if n == 0:
        return 0.0, 1.0
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    w = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return max(0.0, c - w), min(1.0, c + w)


def ler_count(pred, dets, obs):
    pred = np.asarray(pred, dtype=np.uint8).reshape(len(dets), -1)
    actual = obs.astype(np.uint8).reshape(len(dets), -1)
    return int(np.any(pred != actual, axis=1).sum())


def make_circuit(stim, d, noise, basis):
    return stim.Circuit.generated(
        f"surface_code:rotated_memory_{basis}", distance=d, rounds=d,
        after_clifford_depolarization=noise,
        before_measure_flip_probability=noise,
        after_reset_flip_probability=noise,
    )


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--distances", type=int, nargs="+", default=[3, 5, 7])
    ap.add_argument("--basis", default="x", choices=["x", "z"])
    ap.add_argument("--noise", type=float, default=0.005)
    ap.add_argument("--shots", type=int, default=10000)
    ap.add_argument("--seed", type=int, default=20260622)
    ap.add_argument("--max-iter", type=int, default=20)
    ap.add_argument("--out", default="benchmark_results/belief_reference_compare")
    args = ap.parse_args()

    import stim
    import pymatching
    from beliefmatching import BeliefMatching as RefBelief

    env = bm.capture_environment()
    env["timestamp_unix"] = int(time.time())
    env["command"] = "belief_reference_compare " + " ".join(sys.argv[1:])

    rows = []
    for d in args.distances:
        circ = make_circuit(stim, d, args.noise, args.basis)
        sdem = circ.detector_error_model(decompose_errors=True)
        det, obs = circ.compile_detector_sampler(seed=args.seed).sample(
            shots=args.shots, separate_observables=True)
        det = det.astype(np.uint8)
        obs = obs.astype(np.uint8)

        pm = pymatching.Matching.from_detector_error_model(sdem)
        qb = QBelief.from_detector_error_model(sdem, max_iter=args.max_iter)
        # Reference API uses keyword-only ``max_bp_iters`` (not ``max_iter``).
        rb = RefBelief.from_detector_error_model(sdem, max_bp_iters=args.max_iter)

        t0 = time.perf_counter()
        pm_err = ler_count(pm.decode_batch(det), det, obs)
        pm_t = time.perf_counter() - t0

        t0 = time.perf_counter()
        qb_err = ler_count(qb.decode_batch(det), det, obs)
        qb_t = time.perf_counter() - t0

        t0 = time.perf_counter()
        rb_err = ler_count(rb.decode_batch(det), det, obs)
        rb_t = time.perf_counter() - t0

        n = len(det)
        row = {
            "distance": d,
            "basis": args.basis,
            "noise": args.noise,
            "shots": n,
            "detectors": int(sdem.num_detectors),
            "pymatching": {"errors": pm_err, "ler": pm_err / n,
                           "ler_ci95": wilson(pm_err, n), "us_per_shot": 1e6 * pm_t / n},
            "qector_belief": {"errors": qb_err, "ler": qb_err / n,
                              "ler_ci95": wilson(qb_err, n), "us_per_shot": 1e6 * qb_t / n},
            "ref_belief": {"errors": rb_err, "ler": rb_err / n,
                           "ler_ci95": wilson(rb_err, n), "us_per_shot": 1e6 * rb_t / n},
            "belief_vs_pm_reduction_pct": (
                100.0 * (pm_err - qb_err) / pm_err if pm_err else 0.0),
            "qector_vs_ref_abs_ler_diff": abs(qb_err - rb_err) / n,
        }
        rows.append(row)
        print(f"d={d}: PM={pm_err}/{n}  QECTOR-belief={qb_err}/{n}  "
              f"ref-belief={rb_err}/{n}  (|q-ref|={row['qector_vs_ref_abs_ler_diff']:.4f})")

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    bm.write_json(args.out + ".json", {"environment": env, "results": rows})
    _write_md(args.out + ".md", env, rows)
    print(f"\nWrote {args.out}.json and {args.out}.md")
    return 0


def _write_md(path, env, rows):
    lines = ["# QECTOR belief-matching vs reference `beliefmatching`", "",
             f"git_commit: `{env.get('git_commit')}`  shots/d: {rows[0]['shots'] if rows else 0}", "",
             "| d | PyMatching LER | QECTOR-belief LER | ref-belief LER | belief vs PM | |q-ref| |",
             "|---|---|---|---|---|---|"]
    for r in rows:
        lines.append(
            f"| {r['distance']} | {r['pymatching']['ler']:.4f} | "
            f"{r['qector_belief']['ler']:.4f} | {r['ref_belief']['ler']:.4f} | "
            f"{r['belief_vs_pm_reduction_pct']:+.1f}% | {r['qector_vs_ref_abs_ler_diff']:.4f} |")
    with open(path, "w", encoding="utf-8") as fh:
        fh.write("\n".join(lines) + "\n")


if __name__ == "__main__":
    raise SystemExit(main())
