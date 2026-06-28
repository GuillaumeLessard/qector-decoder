#!/usr/bin/env python
"""Shot-level mismatch audit for the d=15 QECTOR-vs-PyMatching LER gap.

Decodes the SAME Stim shots with three decoders to isolate the cause of the gap:

  * QECTOR-Blossom on the COLLAPSED graph         (what the benchmark calls QECTOR)
  * PyMatching on the SAME collapsed graph        (isolates QECTOR's MWPM quality)
  * PyMatching on the FULL, uncollapsed sdem       (the benchmark reference)

QECTOR-collapsed vs PyMatching-collapsed share an identical weighted matching
graph, so any weight difference is a QECTOR MWPM-optimality problem; any
same-weight observable difference is logical-coset/tie selection. PyMatching-
collapsed vs PyMatching-full isolates the effect of the DEM collapse itself.

Writes a per-shot CSV of every mismatch and prints a cause breakdown.

    python scripts/d15_mismatch_audit.py --distance 15 --basis x --shots 40000 \
        --seed 20260622 --out benchmark_results/d15_mismatch_audit.csv
"""
from __future__ import annotations

import argparse
import csv
import os
import sys

_REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
import importlib.util
if importlib.util.find_spec("qector_decoder_v3.qector_decoder_v3") is None:
    sys.path.insert(0, os.path.join(_REPO, "python"))

import numpy as np  # noqa: E402

from qector_decoder_v3 import dem  # noqa: E402

EPS = 1e-6


def build(d, basis, noise):
    import stim
    circ = stim.Circuit.generated(
        f"surface_code:rotated_memory_{basis}", distance=d, rounds=d,
        after_clifford_depolarization=noise,
        before_measure_flip_probability=noise,
        after_reset_flip_probability=noise,
    )
    sdem = circ.detector_error_model(decompose_errors=True)
    raw = dem.from_stim(sdem)
    model = raw.collapse_to_graph() if raw.is_graphlike else raw
    return circ, sdem, raw, model


def _bits(a):
    return "".join(str(int(x)) for x in np.asarray(a).reshape(-1).tolist())


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--distance", type=int, default=15)
    ap.add_argument("--basis", default="x", choices=["x", "z"])
    ap.add_argument("--noise", type=float, default=0.005)
    ap.add_argument("--shots", type=int, default=40000)
    ap.add_argument("--seed", type=int, default=20260622)
    ap.add_argument("--out", default="benchmark_results/d15_mismatch_audit.csv")
    args = ap.parse_args()

    import pymatching

    circ, sdem, raw, model = build(args.distance, args.basis, args.noise)
    H = np.asarray(model.check_matrix())
    L = np.asarray(model.observables_matrix())
    w = np.asarray(model.weights(), dtype=float)
    edges = model.num_errors

    # QECTOR-Blossom exactly as the benchmark/report uses it: the weighted
    # pymatching_compat.Matching (NOT model.make_decoder("blossom"), which is the
    # unweighted cardinality path).
    from qector_decoder_v3 import pymatching_compat
    qm = pymatching_compat.Matching.from_detector_error_model(sdem)
    pm_coll = pymatching.Matching.from_check_matrix(H, weights=w, faults_matrix=L)
    pm_full = pymatching.Matching.from_detector_error_model(sdem)

    det, obs = circ.compile_detector_sampler(seed=args.seed).sample(
        shots=args.shots, separate_observables=True)
    det = det.astype(np.uint8)
    obs = obs.astype(np.uint8)

    n = args.shots
    q_err = pc_err = pf_err = 0
    q_heavier = q_lighter = 0
    gap_shots = 0                      # qector wrong AND pm_full right
    cause = {"QECTOR_suboptimal_weight": 0, "same_weight_diff_coset": 0,
             "collapse_effect": 0, "other": 0}
    rows = []

    for i in range(n):
        s = det[i]
        sw = int(s.sum())
        cq = np.asarray(qm.decode_to_edges_array(s)).astype(np.uint8)
        oq = ((L @ cq) & 1).astype(np.uint8)
        wq = float(w[cq.astype(bool)].sum())
        supq = int(cq.sum())

        pc, wc = pm_coll.decode(s, return_weight=True)
        pc = np.asarray(pc).astype(np.uint8).reshape(-1)
        pf, wf = pm_full.decode(s, return_weight=True)
        pf = np.asarray(pf).astype(np.uint8).reshape(-1)

        act = obs[i]
        qok = int(np.array_equal(oq, act))
        pcok = int(np.array_equal(pc, act))
        pfok = int(np.array_equal(pf, act))
        q_err += 1 - qok
        pc_err += 1 - pcok
        pf_err += 1 - pfok
        if wq > wc + EPS:
            q_heavier += 1
        elif wq < wc - EPS:
            q_lighter += 1
        if not qok and pfok:
            gap_shots += 1

        # record every shot where QECTOR's observable differs from the reference,
        # or QECTOR is wrong where PyMatching (full) is right.
        record = (not np.array_equal(oq, pf)) or (qok != pfok)
        if not record:
            continue
        if wq > wc + EPS:
            note = "QECTOR_suboptimal_weight"
        elif not np.array_equal(oq, pc) and abs(wq - wc) <= EPS:
            note = "same_weight_diff_coset"
        elif not np.array_equal(pc, pf):
            note = "collapse_effect"
        else:
            note = "other"
        cause[note] += 1
        rows.append({
            "shot_id": i, "syndrome_weight": sw,
            "qector_weight": round(wq, 4), "pymatching_weight": round(wc, 4),
            "weight_delta": round(wq - wc, 4),
            "qector_observable": _bits(oq),
            "pm_collapsed_observable": _bits(pc),
            "pm_full_observable": _bits(pf),
            "actual_observable": _bits(act),
            "qector_correct": qok, "pm_collapsed_correct": pcok,
            "pm_full_correct": pfok,
            "correction_support": supq, "collapse_edge_count": edges,
            "notes": note,
        })

    summary = {
        "distance": args.distance, "basis": args.basis, "shots": n,
        "seed": args.seed, "noise": args.noise, "collapse_edges": edges,
        "ler_qector_collapsed": q_err / n, "ler_pm_collapsed": pc_err / n,
        "ler_pm_full": pf_err / n,
        "errors_qector": q_err, "errors_pm_collapsed": pc_err, "errors_pm_full": pf_err,
        "shots_qector_heavier": q_heavier, "shots_qector_lighter": q_lighter,
        "gap_shots": gap_shots, "cause_breakdown": cause,
    }
    import json as _json
    sj = args.out.rsplit(".", 1)[0] + "_summary.json"
    with open(sj, "w", encoding="utf-8") as fh:
        _json.dump(summary, fh, indent=2)

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    cols = ["shot_id", "syndrome_weight", "qector_weight", "pymatching_weight",
            "weight_delta", "qector_observable", "pm_collapsed_observable",
            "pm_full_observable", "actual_observable", "qector_correct",
            "pm_collapsed_correct", "pm_full_correct", "correction_support",
            "collapse_edge_count", "notes"]
    with open(args.out, "w", newline="", encoding="utf-8") as fh:
        wtr = csv.DictWriter(fh, fieldnames=cols)
        wtr.writeheader()
        wtr.writerows(rows)

    print(f"=== d={args.distance} memory_{args.basis}  shots={n}  seed={args.seed} "
          f"edges={edges} ===")
    print(f"LER  QECTOR(collapsed) = {q_err/n:.5f} ({q_err} errs)")
    print(f"LER  PyMatching(collapsed) = {pc_err/n:.5f} ({pc_err} errs)")
    print(f"LER  PyMatching(full sdem) = {pf_err/n:.5f} ({pf_err} errs)")
    print(f"shots QECTOR heavier than PM(collapsed): {q_heavier}  "
          f"(QECTOR lighter: {q_lighter})")
    print(f"GAP shots (QECTOR wrong, PM-full right): {gap_shots}")
    print(f"recorded mismatch rows: {len(rows)}  cause breakdown: {cause}")
    print(f"wrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
