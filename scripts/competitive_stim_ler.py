#!/usr/bin/env python
"""
Real Stim circuit-level head-to-head: QECTOR vs PyMatching.

Generates rotated-surface-code memory circuits at circuit-level noise across a
distance sweep, builds the decoding problem from the Stim Detector Error Model,
and measures the **logical error rate** (with Wilson 95% intervals) and decode
latency for:

  * QECTOR `BlossomDecoder` — weighted, exact polynomial MWPM (uses the DEM's
    `log((1-p)/p)` edge weights, like PyMatching).
  * QECTOR `UnionFindDecoder` — the fast near-linear path (unweighted).
  * `pymatching.Matching` — the reference weighted MWPM decoder.

All three decode the *same* sampled shots from the *same* DEM, so the LER
comparison is apples-to-apples. Output: JSON + a Markdown table with real numbers.

Usage:
    python scripts/competitive_stim_ler.py --distances 3 5 7 9 11 \
        --noise 0.005 --rounds-equal-distance --shots 40000 \
        --out benchmark_results/competitive_stim_ler
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

import qector_decoder_v3 as qd  # noqa: E402
from qector_decoder_v3 import dem  # noqa: E402
from qector_decoder_v3 import benchmarking as bm  # noqa: E402
from qector_decoder_v3 import pymatching_compat  # noqa: E402


def wilson(k: int, n: int, z: float = 1.959963985) -> tuple[float, float]:
    if n == 0:
        return 0.0, 1.0
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    w = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return max(0.0, c - w), min(1.0, c + w)


def ler_for(predict_obs, dets, obs) -> tuple[int, float, float]:
    """Return (errors, latency_seconds_total, ler) for a predictor over shots."""
    actual = obs.astype(np.uint8)
    errors = 0
    t0 = time.perf_counter()
    preds = predict_obs(dets)
    dt = time.perf_counter() - t0
    preds = np.asarray(preds, dtype=np.uint8).reshape(len(dets), -1)
    for i in range(len(dets)):
        if not np.array_equal(preds[i], actual[i]):
            errors += 1
    return errors, dt, errors / len(dets)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--distances", type=int, nargs="+", default=[3, 5, 7, 9])
    ap.add_argument("--noise", type=float, default=0.005)
    ap.add_argument("--shots", type=int, default=30000)
    ap.add_argument("--rounds", type=int, default=0, help="0 => rounds = distance")
    ap.add_argument("--task", default="surface_code:rotated_memory_x")
    ap.add_argument("--out", default="benchmark_results/competitive_stim_ler")
    args = ap.parse_args()

    import stim
    import pymatching

    env = bm.capture_environment()
    env["timestamp_unix"] = int(time.time())
    env["command"] = " ".join(sys.argv)
    env["stim_version"] = stim.__version__
    env["pymatching_version"] = pymatching.__version__

    rows = []
    for d in args.distances:
        rounds = args.rounds or d
        circ = stim.Circuit.generated(
            args.task,
            distance=d,
            rounds=rounds,
            after_clifford_depolarization=args.noise,
            before_measure_flip_probability=args.noise,
            after_reset_flip_probability=args.noise,
        )
        sdem = circ.detector_error_model(decompose_errors=True)
        raw = dem.from_stim(sdem)
        model = raw.collapse_to_graph() if raw.is_graphlike else raw
        c2q, nq = model.check_to_qubits(), model.num_errors
        L = model.observables_matrix()

        sampler = circ.compile_detector_sampler()
        dets, obs = sampler.sample(shots=args.shots, separate_observables=True)
        dets = dets.astype(np.uint8)

        # --- QECTOR drop-in Matching: collapsed graph, weighted MWPM, batched ---
        qm = pymatching_compat.Matching.from_detector_error_model(sdem)

        def qector_blossom_predict(dd, _m=qm):
            return _m.decode_batch(dd)

        # --- QECTOR UnionFind (fast, unweighted) on the collapsed graph, batched ---
        uf = qd.UnionFindDecoder(c2q, nq)

        def qector_uf_predict(dd, _dec=uf, _L=L):
            corr = np.asarray(_dec.batch_decode(dd), dtype=np.uint8)
            return ((_L @ corr.T) & 1).T.astype(np.uint8)

        # --- PyMatching reference (weighted, batched) ---
        pm = pymatching.Matching.from_detector_error_model(sdem)

        def pymatching_predict(dd, _m=pm):
            return _m.decode_batch(dd)

        result = {"distance": d, "rounds": rounds, "noise": args.noise,
                  "shots": args.shots, "detectors": model.num_detectors,
                  "raw_mechanisms": raw.num_errors, "collapsed_edges": model.num_errors,
                  "graphlike": model.is_graphlike}

        for name, fn in [
            ("qector_blossom_weighted", qector_blossom_predict),
            ("qector_unionfind", qector_uf_predict),
            ("pymatching", pymatching_predict),
        ]:
            k, dt, ler = ler_for(fn, dets, obs)
            lo, hi = wilson(k, args.shots)
            result[name] = {
                "logical_errors": k,
                "ler": ler,
                "ler_ci95": [lo, hi],
                "decode_us_per_shot": dt / args.shots * 1e6,
                "shots_per_s": args.shots / dt if dt > 0 else None,
            }
        rows.append(result)
        print(
            f"d={d:2d} rounds={rounds:2d} raw={raw.num_errors:5d}->edges={model.num_errors:4d} | "
            f"QECTOR-Blossom LER={result['qector_blossom_weighted']['ler']:.4f} "
            f"({result['qector_blossom_weighted']['decode_us_per_shot']:.1f}us) | "
            f"PyMatching LER={result['pymatching']['ler']:.4f} "
            f"({result['pymatching']['decode_us_per_shot']:.1f}us) | "
            f"QECTOR-UF LER={result['qector_unionfind']['ler']:.4f} "
            f"({result['qector_unionfind']['decode_us_per_shot']:.1f}us)",
            flush=True,
        )

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    with open(args.out + ".json", "w", encoding="utf-8") as fh:
        json.dump({"environment": env, "results": rows}, fh, indent=2)
    _write_markdown(args.out + ".md", env, rows)
    print(f"\nwrote {args.out}.json and {args.out}.md")
    return 0


def _write_markdown(path, env, rows):
    lines = [
        "# Stim circuit-level head-to-head — QECTOR vs PyMatching",
        "",
        f"- Task: `surface_code:rotated_memory_x`, rounds = distance, "
        f"circuit-level depolarizing+measurement+reset noise p = {rows[0]['noise'] if rows else 'n/a'}",
        f"- Shots per point: {rows[0]['shots'] if rows else 'n/a'}",
        f"- CPU: {env.get('processor') or env.get('platform')}; "
        f"Python {env.get('python_version')}; NumPy {env.get('numpy_version')}; "
        f"Stim {env.get('stim_version')}; PyMatching {env.get('pymatching_version')}",
        "",
        "LER with Wilson 95% interval; latency is per-shot decode time (hot path).",
        "",
        "| d | rounds | raw mech | collapsed edges | QECTOR-Blossom LER | PyMatching LER | QECTOR-UF LER | QB µs/shot | PM µs/shot | UF µs/shot |",
        "|---|--------|----------|-----------------|--------------------|----------------|---------------|-----------|-----------|-----------|",
    ]
    for r in rows:
        qb, pm, uf = r["qector_blossom_weighted"], r["pymatching"], r["qector_unionfind"]
        lines.append(
            f"| {r['distance']} | {r['rounds']} | {r['raw_mechanisms']} | {r['collapsed_edges']} | "
            f"{qb['ler']:.4f} [{qb['ler_ci95'][0]:.4f},{qb['ler_ci95'][1]:.4f}] | "
            f"{pm['ler']:.4f} [{pm['ler_ci95'][0]:.4f},{pm['ler_ci95'][1]:.4f}] | "
            f"{uf['ler']:.4f} | "
            f"{qb['decode_us_per_shot']:.1f} | {pm['decode_us_per_shot']:.1f} | "
            f"{uf['decode_us_per_shot']:.1f} |"
        )
    with open(path, "w", encoding="utf-8") as fh:
        fh.write("\n".join(lines) + "\n")


if __name__ == "__main__":
    raise SystemExit(main())
