# -*- coding: utf-8 -*-
"""
QECTOR v0.7.0 — PARITY SWEEP: unweighted UF vs weighted UF vs BPOSD vs PyMatching

Compares accuracy (LER) and throughput on rotated surface codes and a
bivariate-bicycle qLDPC code.  The weighted UF (UF-01) uses per-mechanism
edge weights derived from DEM error probabilities ``log((1-p)/p)``.

Usage::

    python scripts/parity_sweep_weighted.py [--shots 4000]
"""

from __future__ import annotations

import argparse
import math
import os
import time
import json
import logging
from dataclasses import dataclass, asdict

logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s] %(levelname)s — %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("PARITY_SWEEP")

import numpy as np
import stim
import pymatching

import qector_decoder_v3 as qd
from qector_decoder_v3.qector_memory_align import aligned_stride
from qector_decoder_v3 import codes


P_NOISE = 0.005


@dataclass
class SweepRow:
    code: str
    decoder: str
    distance: int
    shots: int = 0
    det: int = 0
    obs: int = 0
    ler: float = 0.0
    ler_ci95: tuple = (0.0, 0.0)
    logical_errors: int = 0
    duration_s: float = 0.0
    throughput_sps: float = 0.0
    weighted: bool = False
    osd_order: int | None = None


def ler_ci(ler: float, n: int) -> tuple[float, float]:
    z = 1.96
    h = z * math.sqrt(ler * (1.0 - ler) / n) if n > 0 and ler > 0 else 0.0
    return (max(0.0, ler - h), min(1.0, ler + h))


def run_surface(
    d: int,
    shots: int,
    noise: float = P_NOISE,
) -> list[SweepRow]:
    """Run all decoders on rotated surface code at distance d."""
    circ = stim.Circuit.generated(
        "surface_code:rotated_memory_x",
        distance=d,
        rounds=d,
        after_clifford_depolarization=noise,
        before_measure_flip_probability=noise,
        after_reset_flip_probability=noise,
    )
    sdem = circ.detector_error_model(decompose_errors=True)
    sampler = circ.compile_detector_sampler(seed=42)
    dets, obs = sampler.sample(shots, separate_observables=True)
    dets = np.ascontiguousarray(dets, dtype=np.uint8)
    obs_b = np.ascontiguousarray(obs, dtype=np.uint8)

    # Build DEM model for QECTOR decoders
    model = qd.dem.from_stim(sdem)
    c2q, nq = model.check_to_qubits(), model.num_errors
    weights = model.weights().tolist()  # log((1-p)/p)

    n_checks = dets.shape[1]
    stride = aligned_stride(n_checks)
    buf = np.zeros(shots * stride, dtype=np.uint8)
    for i in range(shots):
        buf[i * stride : i * stride + n_checks] = dets[i]

    rows: list[SweepRow] = []
    code_name = f"rotated_surface_d{d}"

    def eval_decoder(label: str, pred: np.ndarray, dt: float, **kw):
        if pred.ndim == 1:
            pred_2d = pred.reshape(shots, -1)
        else:
            pred_2d = pred
        common = min(pred_2d.shape[1], obs_b.shape[1])
        errs = np.any(pred_2d[:, :common] != obs_b[:, :common], axis=1)
        n_err = int(errs.sum())
        ler_v = n_err / shots
        rows.append(SweepRow(
            code=code_name,
            decoder=label,
            distance=d,
            shots=shots,
            det=n_checks,
            obs=obs_b.shape[1],
            ler=ler_v,
            ler_ci95=ler_ci(ler_v, shots),
            logical_errors=n_err,
            duration_s=dt,
            throughput_sps=shots / dt if dt > 0 else 0.0,
            **kw,
        ))
        ci = rows[-1].ler_ci95
        logger.info(
            f"  {label:25s}  LER={ler_v:.5f} [{ci[0]:.5f}, {ci[1]:.5f}]"
            f"  ({n_err}/{shots})  {rows[-1].throughput_sps:>8,.0f} shots/s"
            f"  {'W' if kw.get('weighted') else ' '}"
            f"  OSD={kw.get('osd_order','-')}"
        )

    # ── QECTOR: UF unweighted ──
    uf = qd.UnionFindDecoder(c2q, nq)
    t0 = time.perf_counter()
    p = uf.decode_batch_flat(buf, shots, stride)
    dt = time.perf_counter() - t0
    eval_decoder("uf_unweighted", np.asarray(p), dt, weighted=False)

    # ── QECTOR: UF weighted ──
    uf_w = qd.UnionFindDecoder(c2q, nq, edge_weights=weights)
    t0 = time.perf_counter()
    p = uf_w.decode_batch_flat(buf, shots, stride)
    dt = time.perf_counter() - t0
    eval_decoder("uf_weighted", np.asarray(p), dt, weighted=True)

    # ── QECTOR: BPOSD ──
    try:
        bposd = qd.BPOSDDecoder(c2q, nq, error_rate=noise, osd_order=0)
        t0 = time.perf_counter()
        p = bposd.decode_batch_flat(buf, shots, stride)
        dt = time.perf_counter() - t0
        eval_decoder("bposd_osd0", np.asarray(p), dt, weighted=False, osd_order=0)
    except Exception as e:
        logger.warning(f"  bposd_osd0               SKIP — {e}")

    try:
        bposd1 = qd.BPOSDDecoder(c2q, nq, error_rate=noise, osd_order=1)
        t0 = time.perf_counter()
        p = bposd1.decode_batch_flat(buf, shots, stride)
        dt = time.perf_counter() - t0
        eval_decoder("bposd_osd1", np.asarray(p), dt, weighted=False, osd_order=1)
    except Exception as e:
        logger.warning(f"  bposd_osd1               SKIP — {e}")

    # ── PyMatching 2 (reference) ──
    matcher = pymatching.Matching.from_detector_error_model(sdem)
    t0 = time.perf_counter()
    p = matcher.decode_batch(dets)
    dt = time.perf_counter() - t0
    eval_decoder("pymatching", np.asarray(p, dtype=np.uint8), dt, weighted=True)

    return rows


def run_bicycle(shots: int, noise: float = P_NOISE) -> list[SweepRow]:
    """Bivariate bicycle code ([[144, 12, 12]] or similar) via the code module."""
    rows: list[SweepRow] = []
    code_name = "bicycle_144_12_12"

    # Build parity check matrix from the code module
    try:
        cx, _cz = codes.bivariate_bicycle_code(
            12, 6,
            [("x", 3), ("y", 1), ("y", 2)],
            [("y", 3), ("x", 1), ("x", 2)],
        )
    except Exception as e:
        logger.warning(f"  bicycle  SKIP all — cannot build code: {e}")
        return rows

    H = cx.parity_check_matrix()
    n_checks = H.shape[0]
    n_qubits = H.shape[1]

    c2q = []
    for r in range(n_checks):
        cols = np.flatnonzero(H[r])
        if len(cols) > 0:
            c2q.append(cols.tolist())

    if not c2q:
        logger.warning("  bicycle  SKIP — empty check matrix")
        return rows

    logger.info(f"  bicycle: {n_checks} checks × {n_qubits} qubits")

    # Generate random test syndromes
    rng = np.random.default_rng(42)
    errors = (rng.random((shots, n_qubits), dtype=np.float64) < noise).astype(np.uint8)
    syndromes = (errors @ H.T.astype(np.uint8)) & 1
    syndromes = np.ascontiguousarray(syndromes, dtype=np.uint8)

    # For observables we use the logical Z operators from the code
    # (simplified: use first `n_qubits - n_checks` logicals as observables)
    n_logical = n_qubits - n_checks
    obs_mat = H[:, :n_logical].T.copy() if n_logical > 0 else np.eye(n_checks, dtype=np.uint8)[:1]
    observables = (errors[:, : obs_mat.shape[0]] if obs_mat.shape[0] <= errors.shape[1] else np.zeros((shots, obs_mat.shape[0]), dtype=np.uint8))

    # Flat aligned buffer
    stride = aligned_stride(n_checks)
    buf = np.zeros(shots * stride, dtype=np.uint8)
    for i in range(shots):
        buf[i * stride : i * stride + n_checks] = syndromes[i]

    def eval_decoder(label: str, pred: np.ndarray, dt: float, **kw):
        if pred.ndim == 1:
            pred_2d = pred.reshape(shots, -1)
        else:
            pred_2d = pred
        common = min(pred_2d.shape[1], observables.shape[1])
        errs = np.any(pred_2d[:, :common] != observables[:, :common], axis=1)
        n_err = int(errs.sum())
        ler_v = n_err / shots
        rows.append(SweepRow(
            code=code_name,
            decoder=label,
            distance=0,
            shots=shots,
            det=n_checks,
            obs=observables.shape[1],
            ler=ler_v,
            ler_ci95=ler_ci(ler_v, shots),
            logical_errors=n_err,
            duration_s=dt,
            throughput_sps=shots / dt if dt > 0 else 0.0,
            **kw,
        ))
        ci = rows[-1].ler_ci95
        logger.info(
            f"  {label:25s}  LER={ler_v:.5f} [{ci[0]:.5f}, {ci[1]:.5f}]"
            f"  ({n_err}/{shots})  {rows[-1].throughput_sps:>8,.0f} shots/s"
        )

    # ── UF unweighted ──
    try:
        uf = qd.UnionFindDecoder(c2q, n_qubits)
        t0 = time.perf_counter()
        p = uf.decode_batch_flat(buf, shots, stride)
        dt = time.perf_counter() - t0
        eval_decoder("uf_unweighted", np.asarray(p), dt, weighted=False)
    except Exception as e:
        logger.warning(f"  uf_unweighted  SKIP — {type(e).__name__}: hyperedge code")

    # ── UF weighted (uniform weights — no DEM for custom code) ──
    try:
        uniform_w = [noise] * n_qubits
        uf_w = qd.UnionFindDecoder(c2q, n_qubits, edge_weights=uniform_w)
        t0 = time.perf_counter()
        p = uf_w.decode_batch_flat(buf, shots, stride)
        dt = time.perf_counter() - t0
        eval_decoder("uf_weighted", np.asarray(p), dt, weighted=True)
    except Exception as e:
        logger.warning(f"  uf_weighted  SKIP — {type(e).__name__}: hyperedge code")

    # ── BPOSD ──
    for order in (0, 1):
        try:
            bp = qd.BPOSDDecoder(c2q, n_qubits, error_rate=noise, osd_order=order)
            t0 = time.perf_counter()
            p = bp.decode_batch_flat(buf, shots, stride)
            dt = time.perf_counter() - t0
            eval_decoder(f"bposd_osd{order}", np.asarray(p), dt, weighted=False, osd_order=order)
        except Exception as e:
            logger.warning(f"  bposd_osd{order}  SKIP — {e}")

    return rows


def main():
    ap = argparse.ArgumentParser(description="Parity sweep — weighted UF vs BPOSD vs PyMatching")
    ap.add_argument("--shots", type=int, default=2000)
    args = ap.parse_args()

    logger.info("=" * 72)
    logger.info("QECTOR v0.7.0 — PARITY SWEEP (weighted UF vs BPOSD vs PyMatching)")
    logger.info(f"Shots per cell: {args.shots}")
    logger.info("=" * 72)

    all_rows: list[SweepRow] = []

    for d in (3, 5):
        logger.info(f"── rotated_surface d={d} ───────────────────────")
        all_rows.extend(run_surface(d, args.shots))

    logger.info("── bivariate_bicycle ───────────────────────")
    all_rows.extend(run_bicycle(args.shots // 2))

    # Summary
    logger.info("")
    logger.info("=" * 72)
    logger.info("SUMMARY — LER TABLE")
    logger.info("=" * 72)
    hdr = f"{'CODE':30s} {'DECODER':22s} {'LER':>10s} {'CI95':>24s} {'shots/s':>10s} {'W':>3s}"
    logger.info(hdr)
    logger.info("-" * len(hdr) * 2)
    for r in all_rows:
        ci = r.ler_ci95
        w = "W" if r.weighted else ("*" if r.osd_order is not None else " ")
        logger.info(
            f"{r.code:30s} {r.decoder:22s} {r.ler:>10.5f}"
            f" [{ci[0]:>8.5f}, {ci[1]:>8.5f}]"
            f" {r.throughput_sps:>8,.0f}  {w:>3s}"
        )

    os.makedirs("benchmark_results", exist_ok=True)
    path = "benchmark_results/parity_sweep_weighted.json"
    with open(path, "w") as f:
        json.dump([asdict(r) for r in all_rows], f, indent=2, default=str)
    logger.info(f"Wrote {path} ({len(all_rows)} rows)")


if __name__ == "__main__":
    main()
