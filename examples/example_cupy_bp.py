#!/usr/bin/env python3
"""
example_cupy_bp.py — QECTOR Decoder v3 (v0.5.7) batched GPU BP-OSD

Demonstrates the CuPy-accelerated, GPU-resident batched belief propagation that
feeds BP-OSD (``qector_decoder_v3.bp_cupy`` + ``qector_decoder_v3.bposd``) on a
*non-graphlike* bivariate-bicycle qLDPC code — exactly the regime matching
decoders cannot handle.

Honest behaviour:
  * When CuPy + a usable CUDA device are present, the BP stage for the whole batch
    runs once on the GPU; only the residual non-converged shots take the exact
    GF(2) OSD post-process.
  * When CuPy/CUDA are absent, every call degrades transparently to the NumPy
    path with bit-identical correctness (H @ c == s on converged shots).

Run:
    PYTHONPATH=python python examples/example_cupy_bp.py
"""

import sys

# Force UTF-8 stdout/stderr so emoji output doesn't crash on legacy consoles
# (e.g. Windows cp1252). No-op where reconfigure is unavailable.
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

import numpy as np

from qector_decoder_v3 import codes, gpu_backend as gb
from qector_decoder_v3.bp_cupy import batched_bp_decode
from qector_decoder_v3.bposd import BpOsdDecoder


def syndrome_validity(H: np.ndarray, corr: np.ndarray, syn: np.ndarray) -> np.ndarray:
    """Return a per-shot bool mask of rows satisfying H @ c == s (mod 2)."""
    return np.all(((corr @ H.T) & 1) == syn, axis=1)


def main():
    print("=" * 64)
    print("QECTOR v3 (0.5.7) — Batched GPU BP-OSD on a qLDPC code")
    print("=" * 64)

    # ---- Capability snapshot (honest hardware report) --------------------
    backend = gb.get_backend()
    print("\nBackend capability:")
    for k, v in backend.summary().items():
        print(f"   {k:18s}: {v}")
    using_gpu = gb.gpu_available() and gb.get_prefer_gpu()
    print(f"\n   -> active math path: {'CuPy GPU' if using_gpu else 'NumPy CPU fallback'}")

    # ---- Build a bivariate-bicycle qLDPC code ----------------------------
    # ell=6, m=6 -> n = 2*ell*m = 72 physical qubits; X sector, weight-6 checks.
    code_x, _code_z = codes.bivariate_bicycle_code(
        6, 6, [("x", 3), ("y", 1), ("y", 2)], [("y", 3), ("x", 1), ("x", 2)]
    )
    H = code_x.parity_check_matrix()
    n_checks, n_qubits = H.shape
    print(f"\nCode: {code_x.name}  ({n_qubits} qubits, {n_checks} checks)")
    print(f"   graphlike (matching)? {code_x.is_matching_graph()}  "
          f"max qubit degree = {code_x.max_qubit_degree()}")
    print("   -> non-graphlike: BP-OSD is the only valid decoder family here.")

    # ---- Synthesize a batch of error -> syndrome shots -------------------
    p = 0.03
    batch = 256
    rng = np.random.default_rng(7)
    errors = (rng.random((batch, n_qubits)) < p).astype(np.uint8)
    syndromes = (errors @ H.T) & 1
    print(f"\nBatch: {batch} shots @ physical error rate p = {p}")

    # ---- 1. Raw batched BP (the GPU-resident stage) ----------------------
    gb.reset_telemetry()
    corr_bp, converged = batched_bp_decode(
        H, syndromes, error_rate=p, max_iter=40, bp_method="min_sum"
    )
    valid_bp = syndrome_validity(H, corr_bp, syndromes)
    n_conv = int(converged.sum())
    print("\n1. batched_bp_decode (BP only, GPU-resident when available):")
    print(f"   converged (BP alone explains syndrome): {n_conv}/{batch}")
    print(f"   H·c == s on converged shots: "
          f"{bool(valid_bp[converged].all()) if n_conv else 'n/a'}")
    print(f"   GPU telemetry (this call): {gb.TELEMETRY}")

    # ---- 2. Full BP-OSD batch (BP fast-path + exact GF(2) OSD residual) ---
    gb.reset_telemetry()
    dec = BpOsdDecoder(H, error_rate=p, max_iter=40, osd_order=0, bp_method="min_sum")
    corr_osd = dec.batch_decode(syndromes)
    valid_osd = syndrome_validity(H, corr_osd, syndromes)
    print("\n2. BpOsdDecoder.batch_decode (BP-OSD; OSD-0 fast path):")
    print(f"   H·c == s on ALL {batch} shots: {bool(valid_osd.all())}")
    print(f"   shots needing exact OSD post-process: {batch - n_conv}")
    print(f"   mean correction weight: {corr_osd.sum(axis=1).mean():.2f}")
    print(f"   GPU telemetry (this call): {gb.TELEMETRY}")

    # ---- 3. Prove the NumPy fallback is correctness-equivalent -----------
    # Force the CPU/NumPy path and confirm every correction is still valid.
    prev = gb.get_prefer_gpu()
    try:
        gb.set_prefer_gpu(False)
        gb.reset_telemetry()
        dec_cpu = BpOsdDecoder(H, error_rate=p, max_iter=40, osd_order=0, bp_method="min_sum")
        corr_cpu = dec_cpu.batch_decode(syndromes)
        valid_cpu = syndrome_validity(H, corr_cpu, syndromes)
        print("\n3. Forced NumPy fallback (set_prefer_gpu(False)):")
        print(f"   H·c == s on ALL {batch} shots: {bool(valid_cpu.all())}")
        print(f"   GPU telemetry (should show no GPU calls): {gb.TELEMETRY}")
    finally:
        gb.set_prefer_gpu(prev)

    ok = bool(valid_osd.all()) and (n_conv == 0 or bool(valid_bp[converged].all())) \
        and bool(valid_cpu.all())
    print("\n" + "=" * 64)
    print(f"Result: {'ALL corrections valid (H·c == s). OK.' if ok else 'VALIDITY FAILURE'}")
    print("=" * 64)
    if not ok:
        sys.exit(1)


if __name__ == "__main__":
    main()
