"""qector_decoder_v3.bench — one-command competitive benchmark (C5).

python -m qector_decoder_v3.bench --all
python -m qector_decoder_v3.bench --code surface --d 5 --decoders blossom pymatching --shots 20000 --p 0.005
"""

from __future__ import annotations

import argparse
import json
import time
from typing import Any

import numpy as np

from . import codes
from .ler import estimate_ler, run_threshold_sweep, wilson_ci

__all__ = ["main", "run"]

SURFACE_FACTORY = codes.rotated_surface_code
BB72 = lambda: codes.bivariate_bicycle_code(6, 6, [("x", 3), ("y", 1), ("y", 2)], [("y", 3), ("x", 1), ("x", 2)])[0]


def run(
    code_name: str = "surface",
    distance: int = 5,
    p: float = 0.005,
    shots: int = 2000,
    seed: int = 0,
    decoders: list[str] | None = None,
) -> dict[str, Any]:
    """Run a competitive benchmark and return a structured result dict."""
    if decoders is None:
        decoders = ["blossom", "pymatching"]
    if code_name == "surface":
        code = codes.rotated_surface_code(distance)
    elif code_name == "bb72":
        code = BB72()
    elif code_name == "repetition":
        code = codes.repetition_code(distance)
    else:
        raise ValueError(f"unknown code: {code_name}")

    rows: list[dict[str, Any]] = []
    for dec_name in decoders:
        if dec_name == "pymatching":
            rows.append(_pymatching_row(code, p, shots, seed))
        elif dec_name == "threshold":
            rows.append(_threshold_row(code_name, distance, p, shots, seed))
        else:
            r = estimate_ler(code, dec_name, p=p, shots=shots, seed=seed)
            rows.append(r.to_dict())

    return {
        "code": getattr(code, "name", code_name),
        "distance": distance,
        "p": p,
        "shots": shots,
        "seed": seed,
        "results": rows,
    }


def _pymatching_row(code, p, shots, seed):
    import pymatching
    import stim

    try:
        d = code.distance or 5
        circ = stim.Circuit.generated(
            "surface_code:rotated_memory_x",
            distance=d,
            rounds=d,
            after_clifford_depolarization=p,
            before_measure_flip_probability=p,
            after_reset_flip_probability=p,
        )
        dem = circ.detector_error_model(decompose_errors=True)
        det, obs = circ.compile_detector_sampler(seed=seed).sample(shots=shots, separate_observables=True)
        pm = pymatching.Matching.from_detector_error_model(dem)
        t0 = time.perf_counter()
        pred = np.asarray(pm.decode_batch(det.astype(np.uint8)), np.uint8)
        dt = time.perf_counter() - t0
        err = int(np.any(pred.reshape(len(det), -1) != obs.reshape(len(det), -1), axis=1).sum())
        lo, hi = wilson_ci(err, shots)
        return {
            "decoder": "pymatching",
            "ler": err / shots,
            "ci95": (round(lo, 6), round(hi, 6)),
            "decodes_per_s": round(shots / dt, 1),
            "shots": shots,
            "errors": err,
        }
    except (ImportError, RuntimeError, ValueError) as e:
        return {"decoder": "pymatching", "error": str(e)}


def _threshold_row(code_name, distance, p, shots, seed):
    res = run_threshold_sweep(
        lambda d: codes.rotated_surface_code(d) if code_name == "surface" else codes.repetition_code(d),
        distances=[distance, distance + 2],
        p_values=[p * s for s in (0.5, 1.0, 1.5, 2.0)],
        decoder="blossom",
        shots=shots,
        seed=seed,
    )
    return {"decoder": "threshold_sweep", "crossing": res.crossing, "points": [r.to_dict() for r in res.results]}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--all", action="store_true")
    ap.add_argument("--code", default="surface")
    ap.add_argument("--d", type=int, default=5)
    ap.add_argument("--p", type=float, default=0.005)
    ap.add_argument("--shots", type=int, default=2000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--decoders", nargs="+", default=["blossom", "pymatching"])
    ap.add_argument("--json", action="store_true")
    args = ap.parse_args()

    if args.all:
        args.decoders = ["blossom", "sparse_blossom", "union_find", "pymatching"]
        # Use the competitive suite which covers surface + qLDPC
        from .ler import run_competitive_suite

        results_list = run_competitive_suite(p=args.p, shots=args.shots, seed=args.seed)
        if args.json:
            print(json.dumps(results_list, indent=2))
        else:
            print(f"\nCompetitive suite: p={args.p} {args.shots} shots\n")
            for r in results_list:
                if "error" in r:
                    print(f"  {r['decoder']:<20} ERROR: {r['error']}")
                    continue
                lo, hi = r.get("ci95_lo", 0), r.get("ci95_hi", 0)
                print(
                    f"  {r.get('code', '?'):<22} {r['decoder']:<18} LER={r['ler']:.5f} ({lo:.5f},{hi:.5f}) {r.get('decodes_per_s', 0):>10,.0f}/s"
                )
        return 0

    result = run(args.code, args.d, args.p, args.shots, args.seed, args.decoders)

    if args.json:
        print(json.dumps(result, indent=2))
    else:
        print(f"\n{result['code']} d={result['distance']} p={result['p']} {result['shots']} shots")
        for r in result["results"]:
            if "error" in r:
                print(f"  {r['decoder']:<20} ERROR: {r['error']}")
                continue
            lo, hi = r.get("ci95", (0, 0))
            print(f"  {r['decoder']:<20} LER={r['ler']:.5f} ({lo:.5f},{hi:.5f}) {r.get('decodes_per_s', 0):>12,.0f}/s")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
