#!/usr/bin/env python
"""Circuit-level head-to-head: QECTOR vs PyMatching on identical Stim shots.

Rotated surface-code memory-Z circuits (``surface_code:rotated_memory_z``,
rounds = d) with uniform depolarizing noise p (after_clifford, before_measure,
after_reset, before_round). Every decoder is compiled from the same detector
error model through the sinter ``Decoder`` interface and decodes the same
bit-packed shots, so logical error rates are directly comparable.

Reported per (d, p, decoder): logical errors, LER with a Wilson 95% interval,
and mean decode time per shot (whole-batch wall clock / shots, single thread).

    python benchmarks/stim_head_to_head.py --distances 3 5 7 9 11 \\
        --noise 0.001 0.003 0.005 0.007 --shots 20000 --out results.json
"""
from __future__ import annotations

import argparse
import json
import math
import os
import platform
import time

import numpy as np


def wilson(k: int, n: int, z: float = 1.959964) -> tuple[float, float]:
    if n == 0:
        return (0.0, 1.0)
    p = k / n
    den = 1 + z * z / n
    centre = (p + z * z / (2 * n)) / den
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / den
    return (max(0.0, centre - half), min(1.0, centre + half))


def decoders():
    import pymatching
    import sinter
    from qector_decoder_v3.sinter_compat import qector_sinter_decoders

    q = qector_sinter_decoders()
    out = {
        "pymatching": sinter.BUILT_IN_DECODERS["pymatching"] if hasattr(sinter, "BUILT_IN_DECODERS") else None,
        "qector_blossom": q["qector_blossom"],
        "qector_unionfind": q["qector_unionfind"],
    }
    if out["pymatching"] is None:  # pragma: no cover - very old sinter
        from sinter._decoding._decoding_pymatching import PyMatchingDecoder

        out["pymatching"] = PyMatchingDecoder()
    return out, pymatching.__version__


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--distances", type=int, nargs="+", default=[3, 5, 7, 9, 11])
    ap.add_argument("--noise", type=float, nargs="+", default=[0.001, 0.003, 0.005, 0.007])
    ap.add_argument("--shots", type=int, default=20000)
    ap.add_argument("--seed", type=int, default=20261105)
    ap.add_argument("--out", default="stim_head_to_head.json")
    args = ap.parse_args()

    import stim
    import qector_decoder_v3 as qd

    decs, pm_version = decoders()
    rows = []
    for d in args.distances:
        for p in args.noise:
            circuit = stim.Circuit.generated(
                "surface_code:rotated_memory_z", distance=d, rounds=d,
                after_clifford_depolarization=p, before_measure_flip_probability=p,
                after_reset_flip_probability=p, before_round_data_depolarization=p,
            )
            dem = circuit.detector_error_model(decompose_errors=True)
            sampler = circuit.compile_detector_sampler(seed=args.seed)
            dets, obs = sampler.sample(args.shots, separate_observables=True, bit_packed=True)
            for name, dec in decs.items():
                compiled = dec.compile_decoder_for_dem(dem=dem)
                compiled.decode_shots_bit_packed(bit_packed_detection_event_data=dets[:16])  # warm-up
                t0 = time.perf_counter()
                pred = compiled.decode_shots_bit_packed(bit_packed_detection_event_data=dets)
                dt = time.perf_counter() - t0
                k = int(np.any(pred != obs, axis=1).sum())
                lo, hi = wilson(k, args.shots)
                row = dict(distance=d, p=p, decoder=name, shots=args.shots, errors=k,
                           ler=k / args.shots, ler_lo=lo, ler_hi=hi, us_per_shot=1e6 * dt / args.shots)
                rows.append(row)
                print(f"d={d:2d} p={p:.3f} {name:17s} errors={k:6d} LER={k/args.shots:.5f} "
                      f"[{lo:.5f},{hi:.5f}]  {row['us_per_shot']:9.1f} us/shot", flush=True)
    meta = dict(qector=qd.__version__, pymatching=pm_version, stim=stim.__version__,
                python=platform.python_version(), cpu=platform.processor() or platform.machine(),
                cores=os.cpu_count(), shots=args.shots, seed=args.seed,
                circuit="surface_code:rotated_memory_z, rounds=d, uniform depolarizing p")
    with open(args.out, "w") as f:
        json.dump(dict(meta=meta, rows=rows), f, indent=1)


if __name__ == "__main__":
    main()
