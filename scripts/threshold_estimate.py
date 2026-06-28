#!/usr/bin/env python
"""
Estimate the surface-code threshold with QECTOR via Sinter.

Sweeps physical error rate p across distances and collects the logical error rate
with a QECTOR decoder through the standard Sinter harness. Below threshold the LER
curves fan out (higher d => lower LER); above threshold they cross. The crossing
region is the threshold. For the rotated surface code under this circuit-level
noise model it should land near ~0.5–1% (matching the literature) — a sanity proof
that QECTOR's decoding is physically correct, not just syndrome-consistent.

Usage:
    python scripts/threshold_estimate.py --decoder qector_belief \
        --distances 3 5 7 --probs 0.002 0.004 0.006 0.008 0.01 \
        --max-shots 40000 --max-errors 800
"""
from __future__ import annotations

import argparse
import os
import sys

_REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
import importlib.util
if importlib.util.find_spec("qector_decoder_v3.qector_decoder_v3") is None:
    sys.path.insert(0, os.path.join(_REPO, "python"))


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--decoder", default="qector_belief",
                    choices=["qector_belief", "qector_blossom", "qector_unionfind", "pymatching"])
    ap.add_argument("--distances", type=int, nargs="+", default=[3, 5, 7])
    ap.add_argument("--probs", type=float, nargs="+",
                    default=[0.002, 0.004, 0.006, 0.008, 0.01, 0.012])
    ap.add_argument("--rounds", type=int, default=0, help="0 => rounds = distance")
    ap.add_argument("--max-shots", type=int, default=40000)
    ap.add_argument("--max-errors", type=int, default=800)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--out", default="benchmark_results/threshold")
    args = ap.parse_args()

    import stim
    import sinter
    from qector_decoder_v3.sinter_compat import qector_sinter_decoders

    tasks = []
    for d in args.distances:
        rounds = args.rounds or d
        for p in args.probs:
            circ = stim.Circuit.generated(
                "surface_code:rotated_memory_x", distance=d, rounds=rounds,
                after_clifford_depolarization=p,
                before_measure_flip_probability=p,
                after_reset_flip_probability=p,
            )
            tasks.append(sinter.Task(circuit=circ, json_metadata={"d": d, "p": p}))

    custom = qector_sinter_decoders() if args.decoder.startswith("qector") else {}
    results = sinter.collect(
        num_workers=args.workers, tasks=tasks, decoders=[args.decoder],
        custom_decoders=custom, max_shots=args.max_shots, max_errors=args.max_errors,
        print_progress=True,
    )

    grid = {}
    for r in results:
        d, p = r.json_metadata["d"], r.json_metadata["p"]
        grid[(d, p)] = (r.errors / r.shots if r.shots else float("nan"), r.shots, r.errors)

    print(f"\nLogical error rate — decoder={args.decoder}")
    header = "p \\ d   " + "".join(f"  d={d:<10d}" for d in args.distances)
    print(header)
    for p in args.probs:
        row = f"{p:<8.4f}"
        for d in args.distances:
            ler = grid.get((d, p), (float('nan'),))[0]
            row += f"  {ler:<10.4g}"
        print(row)

    # crossing estimate: smallest p where increasing d no longer lowers LER
    crossing = None
    for p in args.probs:
        lers = [grid.get((d, p), (float("nan"),))[0] for d in args.distances]
        if all(x == x for x in lers) and not _strictly_decreasing(lers):
            crossing = p
            break
    print(f"\nApprox. threshold (first p where larger d stops helping): "
          f"{crossing if crossing is not None else '> max p tested'}")

    import json
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    with open(args.out + ".json", "w", encoding="utf-8") as fh:
        json.dump({"decoder": args.decoder,
                   "grid": {f"d{d}_p{p}": grid[(d, p)] for (d, p) in grid}}, fh, indent=2)
    print(f"wrote {args.out}.json")
    return 0


def _strictly_decreasing(xs) -> bool:
    return all(xs[i] > xs[i + 1] for i in range(len(xs) - 1))


if __name__ == "__main__":
    raise SystemExit(main())
