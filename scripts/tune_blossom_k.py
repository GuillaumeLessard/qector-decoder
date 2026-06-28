#!/usr/bin/env python
"""Tune the adaptive-k multiplier for QECTOR-Blossom: find the SMALLEST
QECTOR_BLOSSOM_K_MULT that preserves exact-MWPM logical parity at large d, so the
latency cost of the d>=15 optimality fix is minimised.

For each multiplier it rebuilds the decoder (env read at construction), decodes
the same seeded shots, and reports the logical gap vs PyMatching and the decode
latency. Pick the smallest multiplier with gap == 0.

    python scripts/tune_blossom_k.py --distances 13 15 --shots 3000
"""
from __future__ import annotations

import argparse
import os
import sys
import time

_REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
import importlib.util
if importlib.util.find_spec("qector_decoder_v3.qector_decoder_v3") is None:
    sys.path.insert(0, os.path.join(_REPO, "python"))

import numpy as np  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--distances", type=int, nargs="+", default=[13, 15])
    ap.add_argument("--mults", type=float, nargs="+",
                    default=[1.0, 1.5, 2.0, 2.5, 3.0, 4.0])
    ap.add_argument("--shots", type=int, default=3000)
    ap.add_argument("--noise", type=float, default=0.005)
    ap.add_argument("--seed", type=int, default=20260622)
    args = ap.parse_args()

    import stim
    import pymatching
    from qector_decoder_v3 import dem

    # Pre-build problems + reference (independent of the multiplier).
    probs = {}
    for d in args.distances:
        circ = stim.Circuit.generated(
            "surface_code:rotated_memory_x", distance=d, rounds=d,
            after_clifford_depolarization=args.noise,
            before_measure_flip_probability=args.noise,
            after_reset_flip_probability=args.noise)
        sdem = circ.detector_error_model(decompose_errors=True)
        dem.from_stim(sdem).collapse_to_graph()
        det, obs = circ.compile_detector_sampler(seed=args.seed).sample(
            shots=args.shots, separate_observables=True)
        pmf = pymatching.Matching.from_detector_error_model(sdem)
        pf = np.asarray(pmf.decode_batch(det.astype(np.uint8)), np.uint8).reshape(args.shots, -1)
        pf_err = int(np.any(pf != obs.astype(np.uint8), axis=1).sum())
        probs[d] = (sdem, det.astype(np.uint8), obs.astype(np.uint8), pf_err)

    print(f"{'mult':>5} " + " ".join(f"d={d}:gap/lat" for d in args.distances))
    for mult in args.mults:
        os.environ["QECTOR_BLOSSOM_K_MULT"] = str(mult)
        # fresh import each time so BlossomDecoder.new() re-reads the env
        from qector_decoder_v3 import pymatching_compat
        cells = []
        for d in args.distances:
            sdem, det, obs, pf_err = probs[d]
            qm = pymatching_compat.Matching.from_detector_error_model(sdem)
            t0 = time.perf_counter()
            qp = np.asarray(qm.decode_batch(det), np.uint8).reshape(args.shots, -1)
            lat = (time.perf_counter() - t0) / args.shots * 1e6
            q_err = int(np.any(qp != obs, axis=1).sum())
            gap = q_err - pf_err  # logical gap vs exact MWPM
            cells.append(f"{gap:+d}/{lat:.0f}us")
        print(f"{mult:>5.1f} " + "  ".join(f"d={d} {c}" for d, c in zip(args.distances, cells)),
              flush=True)
    print("\nPick the smallest mult whose gap is ~0 at every d; that minimises latency.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
