#!/usr/bin/env python
"""E4 + C1 (K3dev.md): QECTOR BP-OSD vs the ldpc reference family on qLDPC.

Runs the full BP-family comparison on the BB[[72,12]] bivariate-bicycle code
with *real* logical operators (E1): every decoder gets identical sampled
errors; we count syndrome-faithfulness, logical failures (via the biorthogonal
duals), Wilson 95% intervals, and decode throughput.

Decoders:
  * QECTOR native Rust BP-OSD, osd_order in {0, 2, 5}   (E4: orders >2 work)
  * QECTOR pure-Python BP-OSD (OSD-0)
  * ldpc ``BpOsdDecoder`` OSD_CS order 0 and 5          (reference ground truth)
  * ldpc ``BpLsdDecoder`` lsd_order 5                   (BP-LSD)
  * ldpc ``BeliefFindDecoder``                          (BeliefFind)

    python scripts/competitive_extended.py --shots 3000 --p 0.03 --seed 1
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time

_REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(_REPO, "python"))

import numpy as np
from qector_decoder_v3 import codes
from qector_decoder_v3.ler import wilson_ci

BB72_A = [("x", 3), ("y", 1), ("y", 2)]
BB72_B = [("y", 3), ("x", 1), ("x", 2)]


def _eval(name: str, C: np.ndarray, E: np.ndarray, S: np.ndarray, H: np.ndarray, Lx: np.ndarray, dt: float) -> dict:
    C = np.asarray(C, dtype=np.uint8)
    if C.shape != E.shape:
        return {"decoder": name, "error": f"shape {C.shape} != {E.shape}"}
    faithful = int(np.all(((C @ H.T) & 1) == S, axis=1).sum())
    flips = ((C ^ E).astype(np.uint8) @ Lx.T) & 1
    fails = int(np.any(flips, axis=1).sum())
    n = len(E)
    lo, hi = wilson_ci(fails, n)
    return {
        "decoder": name,
        "shots": n,
        "faithful_pct": 100.0 * faithful / n,
        "logical_failures": fails,
        "ler": fails / n,
        "ci95": (round(lo, 6), round(hi, 6)),
        "decodes_per_s": round(n / dt, 1) if dt > 0 else None,
        "seconds": round(dt, 4),
    }


def run(shots: int, p: float, seed: int) -> list[dict]:
    from qector_decoder_v3 import BPOSDDecoder
    from qector_decoder_v3.bposd import BpOsdDecoder as PyBpOsd

    cx, _cz = codes.bivariate_bicycle_code(6, 6, BB72_A, BB72_B)
    H = cx.parity_check_matrix().astype(np.uint8)
    Lx = cx.logicals_matrix()
    rng = np.random.default_rng(seed)
    E = (rng.random((shots, cx.n_qubits)) < p).astype(np.uint8)
    S = ((E @ H.T) & 1).astype(np.uint8)

    rows: list[dict] = []

    # ---- QECTOR native Rust BP-OSD at several OSD orders (E4)
    for order in (0, 2, 5):
        dec = BPOSDDecoder(cx.check_to_qubits, cx.n_qubits, p, osd_order=order)
        t0 = time.perf_counter()
        C = np.stack([np.asarray(dec.decode(s)) for s in S])
        rows.append(_eval(f"qector_rust_bposd_osd{order}", C, E, S, H, Lx, time.perf_counter() - t0))

    # ---- QECTOR pure-Python BP-OSD (OSD-0, batched)
    dec = PyBpOsd(H, error_rate=p, max_iter=30, osd_order=0)
    t0 = time.perf_counter()
    C = np.asarray(dec.batch_decode(S))
    rows.append(_eval("qector_py_bposd_osd0", C, E, S, H, Lx, time.perf_counter() - t0))

    # ---- ldpc reference family (C1) — each decoder fault-isolated so an
    # applicability failure (e.g. BeliefFind's peeling needs graphlike PCM
    # columns) is reported honestly instead of aborting the comparison.
    try:
        from ldpc import BeliefFindDecoder, BpLsdDecoder
        from ldpc import BpOsdDecoder as LdpcOsd
    except ImportError as exc:  # pragma: no cover - ldpc optional
        rows.append({"decoder": "ldpc_family", "error": f"ldpc not installed: {exc}"})
        return rows

    def _run_ldpc(name: str, build) -> None:
        try:
            dec = build()
            t0 = time.perf_counter()
            C = np.stack([np.asarray(dec.decode(s)) for s in S])
            rows.append(_eval(name, C, E, S, H, Lx, time.perf_counter() - t0))
        except (ValueError, RuntimeError, TypeError) as exc:
            # Applicability/availability failures (e.g. BeliefFind's peeling
            # needs graphlike PCM columns) are reported, not crashed on.
            rows.append({"decoder": name, "error": str(exc).splitlines()[0]})

    for order in (0, 5):
        _run_ldpc(
            f"ldpc_bposd_cs{order}",
            lambda order=order: LdpcOsd(H, error_rate=p, max_iter=30, bp_method="product_sum",
                                        osd_method="OSD_CS", osd_order=order),
        )
    _run_ldpc("ldpc_bplsd_5", lambda: BpLsdDecoder(H, error_rate=p, max_iter=30,
                                                   bp_method="product_sum", lsd_order=5))

    def _belieffind():
        try:
            return BeliefFindDecoder(H, error_rate=p, max_iter=30, bp_method="product_sum")
        except ValueError:
            # peeling UF needs graphlike (degree<=2) PCM columns; retry inversion
            return BeliefFindDecoder(H, error_rate=p, max_iter=30, bp_method="product_sum",
                                     uf_method="inversion")

    _run_ldpc("ldpc_belieffind", _belieffind)

    try:
        import beliefmatching
        import stim
        def _beliefmatching():
            dem_lines = []
            for q in range(H.shape[1]):
                checks = np.where(H[:, q])[0]
                if len(checks) == 1:
                    dem_lines.append(f"error({p}) D{checks[0]}")
                elif len(checks) == 2:
                    dem_lines.append(f"error({p}) D{checks[0]} D{checks[1]}")
            if not dem_lines:
                raise ValueError("BeliefMatching requires graphlike edges (check degree <= 2)")
            dem = stim.DetectorErrorModel("\n".join(dem_lines))
            return beliefmatching.BeliefMatching.from_detector_error_model(dem)
        _run_ldpc("beliefmatching", _beliefmatching)
    except Exception as exc:
        rows.append({"decoder": "beliefmatching", "error": str(exc).splitlines()[0]})

    return rows


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--shots", type=int, default=3000)
    ap.add_argument("--p", type=float, default=0.03)
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--out", default="")
    args = ap.parse_args()

    rows = run(args.shots, args.p, args.seed)
    print(f"\nBB[[72,12]] X-sector, p={args.p}, {args.shots} shots, seed={args.seed}")
    print(f"{'decoder':<26} {'LER':>8} {'CI95':>18} {'faithful%':>10} {'dec/s':>12}")
    for r in rows:
        if "error" in r:
            print(f"{r['decoder']:<26} ERROR: {r['error']}")
            continue
        lo, hi = r["ci95"]
        print(f"{r['decoder']:<26} {r['ler']:>8.5f} ({lo:.5f},{hi:.5f}) {r['faithful_pct']:>9.1f}% {r['decodes_per_s']:>12,.0f}")

    if args.out:
        os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
        with open(args.out, "w", encoding="utf-8") as fh:
            json.dump({"p": args.p, "shots": args.shots, "seed": args.seed, "results": rows}, fh, indent=2)
        print(f"wrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
