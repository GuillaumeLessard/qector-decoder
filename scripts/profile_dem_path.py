#!/usr/bin/env python
"""S1 (K3dev.md): stage-by-stage profile of the Stim-DEM decode path.

Splits the wall time of QECTOR's ``Matching.from_detector_error_model`` path
into: DEM flatten+parse, graph collapse, decoder construction, and batched
decode — and compares the decode stage against PyMatching on identical shots.
The goal is to attribute the gap to Python-bridge overhead (fixable here) vs
the algorithmic core (dense exact Edmonds vs sparse blossom).

    python scripts/profile_dem_path.py --distances 3 5 7 --shots 20000
"""
from __future__ import annotations

import argparse
import os
import sys
import time

_REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(_REPO, "python"))

import numpy as np


def profile_distance(d: int, shots: int, p: float, seed: int) -> dict:
    import pymatching
    import stim
    from qector_decoder_v3.dem import from_stim
    from qector_decoder_v3.pymatching_compat import Matching as QMatching

    circ = stim.Circuit.generated(
        "surface_code:rotated_memory_x", distance=d, rounds=d,
        after_clifford_depolarization=p, before_measure_flip_probability=p,
        after_reset_flip_probability=p,
    )
    dem = circ.detector_error_model(decompose_errors=True)
    det, obs = circ.compile_detector_sampler(seed=seed).sample(shots=shots, separate_observables=True)
    det = np.ascontiguousarray(det.astype(np.uint8))
    obs = np.asarray(obs, np.uint8)

    # ---- stage 1: flatten + regex parse (qector dem.from_stim)
    t0 = time.perf_counter()
    model = from_stim(dem)
    t_parse = time.perf_counter() - t0

    # ---- stage 2: collapse to graph
    t0 = time.perf_counter()
    if model.is_graphlike:
        model = model.collapse_to_graph()
    t_collapse = time.perf_counter() - t0

    # ---- stage 3: build QECTOR Matching (incl. dense Blossom construction)
    H = model.check_matrix()
    W = model.weights()
    F = model.observables_matrix()
    t0 = time.perf_counter()
    qm = QMatching.from_check_matrix(H, weights=W, faults_matrix=F)
    qm._ensure_decoder()
    t_build_q = time.perf_counter() - t0

    # ---- stage 4a: QECTOR full wrapper decode_batch
    qm.decode_batch(det[:64])  # warmup
    t0 = time.perf_counter()
    qpred = np.asarray(qm.decode_batch(det), dtype=np.uint8)
    t_qwrap = time.perf_counter() - t0

    # ---- stage 4b: raw Rust batch decode (no wrapper, no faults matmul)
    dec = qm._ensure_decoder()
    t0 = time.perf_counter()
    corr = np.asarray(dec.batch_decode(det), dtype=np.uint8)
    t_qraw = time.perf_counter() - t0

    # ---- stage 4c: faults-matrix observable mapping alone
    t0 = time.perf_counter()
    _ = ((qm._faults_matrix @ corr.T) & 1).T
    t_faults = time.perf_counter() - t0

    # ---- stage 5: PyMatching build + decode_batch on the same DEM/shots
    t0 = time.perf_counter()
    pm = pymatching.Matching.from_detector_error_model(dem)
    t_build_pm = time.perf_counter() - t0
    pm.decode_batch(det[:64])
    t0 = time.perf_counter()
    ppred = np.asarray(pm.decode_batch(det), dtype=np.uint8)
    t_pm = time.perf_counter() - t0

    q_err = int(np.any(qpred.reshape(len(det), -1) != obs.reshape(len(det), -1), axis=1).sum())
    pm_err = int(np.any(ppred.reshape(len(det), -1) != obs.reshape(len(det), -1), axis=1).sum())

    return {
        "d": d, "shots": shots,
        "detectors": int(dem.num_detectors), "mechanisms_raw": len(model.errors),
        "edges_collapsed": int(H.shape[1]),
        "t_parse_s": t_parse, "t_collapse_s": t_collapse,
        "t_build_qector_s": t_build_q, "t_build_pymatching_s": t_build_pm,
        "qector_wrap_dec_s": shots / t_qwrap, "qector_raw_dec_s": shots / t_qraw,
        "qector_faults_s": shots / t_faults if t_faults > 0 else float("inf"),
        "pymatching_dec_s": shots / t_pm,
        "qector_errors": q_err, "pymatching_errors": pm_err,
        "wrapper_overhead_pct": 100.0 * (t_qwrap - t_qraw) / t_qwrap if t_qwrap > 0 else 0.0,
    }


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--distances", type=int, nargs="+", default=[3, 5, 7])
    ap.add_argument("--shots", type=int, default=20000)
    ap.add_argument("--p", type=float, default=0.005)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    print(f"S1 DEM-path profile — {args.shots} shots, p={args.p}\n")
    for d in args.distances:
        r = profile_distance(d, args.shots, args.p, args.seed)
        ratio = r["qector_wrap_dec_s"] / r["pymatching_dec_s"]
        print(
            f"d={d}: det={r['detectors']} edges={r['edges_collapsed']} | "
            f"parse={r['t_parse_s']*1e3:.1f}ms collapse={r['t_collapse_s']*1e3:.1f}ms "
            f"buildQ={r['t_build_qector_s']*1e3:.1f}ms buildPM={r['t_build_pymatching_s']*1e3:.1f}ms\n"
            f"   decode: QECTOR-wrap={r['qector_wrap_dec_s']:,.0f}/s "
            f"QECTOR-raw={r['qector_raw_dec_s']:,.0f}/s PM={r['pymatching_dec_s']:,.0f}/s "
            f"| wrap-overhead={r['wrapper_overhead_pct']:.1f}% | Q/PM={ratio:.2f}x "
            f"| LER errors Q={r['qector_errors']} PM={r['pymatching_errors']}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
