#!/usr/bin/env python
"""
Per-shot decode latency benchmark — p50/p99 per round.

Measures the **distribution** of individual decode call latencies across
circuit-level surface-code memory experiments. Reports p50 (median) and p99
(99th-percentile) decode time per shot, normalised by rounds to give
µs/(shot·round).

Decoders
--------
* QECTOR-Blossom (``qector_blossom``) — weighted exact MWPM.
* QECTOR-UnionFind (``qector_unionfind``) — fast unweighted UF.
* PyMatching 2 (``pymatching``) — reference MWPM.

Usage::

    python scripts/throughput_stim_ler.py --distances 3 5 7 11 15 21 --shots 2000

Output: JSON + Markdown table with p50/p99 decode latency.
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
from qector_decoder_v3 import pymatching_compat, dem  # noqa: E402


def percentile(data: np.ndarray, p: float) -> float:
    """Simple percentile without numpy dependency on nanpercentile quirks."""
    if len(data) == 0:
        return 0.0
    s = np.sort(data)
    k = int(math.ceil(p / 100.0 * len(s))) - 1
    return float(s[max(0, min(k, len(s) - 1))])


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--distances", type=int, nargs="+",
        default=[3, 5, 7],
        help="Distance values (default: 3 5 7)",
    )
    ap.add_argument("--noise", type=float, default=0.005)
    ap.add_argument(
        "--shots", type=int, default=2000,
        help="Number of decode calls per distance (default: 2000)",
    )
    ap.add_argument(
        "--out",
        default="benchmark_results/throughput_stim_ler",
        help="Output prefix (default: benchmark_results/throughput_stim_ler)",
    )
    args = ap.parse_args()

    import stim
    import pymatching

    env = {}
    env["platform"] = os.uname()._asdict() if hasattr(os, "uname") else {}
    env["python_version"] = sys.version
    env["qector_version"] = qd.__version__
    env["stim_version"] = stim.__version__
    env["pymatching_version"] = pymatching.__version__
    env["command"] = " ".join(sys.argv)

    rows = []
    for d in args.distances:
        n_shots = args.shots
        rounds = d
        circ = stim.Circuit.generated(
            "surface_code:rotated_memory_x",
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

        sampler = circ.compile_detector_sampler(seed=42)
        dets, _obs = sampler.sample(shots=n_shots, separate_observables=True)
        dets = dets.astype(np.uint8)

        # Build decoders
        qm = pymatching_compat.Matching.from_detector_error_model(sdem)
        uf = qd.UnionFindDecoder(c2q, nq)
        pm = pymatching.Matching.from_detector_error_model(sdem)

        print(f"d={d}: benchmarking {n_shots} shots... ", end="", flush=True)

        result = {
            "distance": d,
            "rounds": rounds,
            "noise": args.noise,
            "shots": n_shots,
            "detectors": model.num_detectors,
        }

        for label, dec_fn in [
            ("qector_blossom", lambda s: np.asarray(qm.decode(s), dtype=np.uint8)),
            ("qector_unionfind",
             lambda s: np.asarray(uf.decode(s), dtype=np.uint8)),
            ("pymatching", lambda s: pm.decode(s)),
        ]:
            latencies = np.empty(n_shots, dtype=np.float64)
            for i in range(n_shots):
                t0 = time.perf_counter_ns()
                dec_fn(dets[i])
                latencies[i] = (time.perf_counter_ns() - t0) / 1000.0  # ns → µs
            p50 = percentile(latencies, 50)
            p99 = percentile(latencies, 99)
            mean = float(np.mean(latencies))
            result[label] = {
                "p50_us": p50,
                "p99_us": p99,
                "mean_us": mean,
                "p50_us_per_round": p50 / rounds,
                "p99_us_per_round": p99 / rounds,
                "mean_shots_per_s": 1e6 / mean if mean > 0 else 0,
            }
            print(f"{label}: p50={p50:.1f}µs p99={p99:.1f}µs ", end="", flush=True)
        print()

        rows.append(result)

    # Write output
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    with open(args.out + ".json", "w", encoding="utf-8") as fh:
        json.dump({"environment": env, "results": rows}, fh, indent=2)

    lines = [
        "# Per-shot decode latency — QECTOR vs PyMatching",
        "",
        f"- Circuit: `surface_code:rotated_memory_x`, noise p = {args.noise},",
        "  rounds = distance. Decoder built from collapsed DEM.",
        f"- {n_shots} individual decode calls per point. All latencies in µs.",
        f"- Environment: Python {sys.version.split()[0]}, stim {stim.__version__},",
        f"  pymatching {pymatching.__version__}.",
        "",
        "| d | rounds | QB p50 | QB p99 | QB mean | UF p50 | UF p99 | UF mean |",
        "|   |        | µs/rd | µs/rd | µs/rd | µs/rd | µs/rd | µs/rd |",
        "|---|--------|-------|-------|--------|-------|-------|--------|"
        "-------|-------|---------|",
    ]
    for r in rows:
        rd = r["rounds"]
        qb = r["qector_blossom"]
        uf = r["qector_unionfind"]
        pm = r["pymatching"]
        lines.append(
            f"| {r['distance']} | {rd} |"
            f" {qb['p50_us']/rd:.1f} | {qb['p99_us']/rd:.1f} | {qb['mean_us']/rd:.1f} |"
            f" {uf['p50_us']/rd:.1f} | {uf['p99_us']/rd:.1f} | {uf['mean_us']/rd:.1f} |"
            f" {pm['p50_us']/rd:.1f} | {pm['p99_us']/rd:.1f} | {pm['mean_us']/rd:.1f} |"
        )

    with open(args.out + ".md", "w", encoding="utf-8") as fh:
        fh.write("\n".join(lines) + "\n")

    print(f"wrote {args.out}.json and {args.out}.md")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
