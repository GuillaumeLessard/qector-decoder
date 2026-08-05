"""BP-OSD iteration throughput benchmark (K3dev A3).

Measures end-to-end ``batch_decode`` dec/s of the native Rust ``BPOSDDecoder``
on (a) the BB[[72,12]] bivariate-bicycle X-sector and (b) a d=9 rotated-surface
circuit-level DEM. Run before/after a change to ``bp_osd.rs``; the numbers are
recorded in the K3dev Evidence Log.

Usage:
    python scripts/bench_bp_iteration.py [--shots 20000] [--seed 7]
"""

from __future__ import annotations

import argparse
import time

import numpy as np
from qector_decoder_v3 import BPOSDDecoder, codes
from qector_decoder_v3.dem import from_stim


def _bench(name: str, check_to_qubits: list[list[int]], n_qubits: int, p: float,
           shots: int, seed: int) -> float:
    rng = np.random.default_rng(seed)
    n_checks = len(check_to_qubits)
    # Reachable syndromes: sample physical errors at rate p, derive H @ e.
    errors = (rng.random((shots, n_qubits)) < p).astype(np.uint8)
    syndromes = np.zeros((shots, n_checks), dtype=np.uint8)
    for c, qs in enumerate(check_to_qubits):
        if qs:
            syndromes[:, c] = errors[:, qs].sum(axis=1) % 2
    dec = BPOSDDecoder(check_to_qubits, n_qubits, p)
    # Warm-up (decoder construction already done; first-call lazy state).
    dec.batch_decode(syndromes[:64])
    t0 = time.perf_counter()
    dec.batch_decode(syndromes)
    dt = time.perf_counter() - t0
    rate = shots / dt
    nnz = sum(len(q) for q in check_to_qubits)
    print(f"{name}: n_checks={n_checks} n_qubits={n_qubits} nnz={nnz} "
          f"shots={shots} p={p} -> {rate:,.0f} dec/s ({dt:.2f}s)")
    return rate


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--shots", type=int, default=5_000,
                    help="shots for the BB72 bench")
    ap.add_argument("--dem-shots", type=int, default=200,
                    help="shots for the d=9 DEM bench (26,823 mechanisms — "
                         "the GF(2) solve dominates; ~0.7 dec/s pre-A2)")
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--p", type=float, default=0.03)
    args = ap.parse_args()

    # BB[[72,12]] X-sector (same construction as competitive_extended.py).
    bb72_a = [("x", 3), ("y", 1), ("y", 2)]
    bb72_b = [("y", 3), ("x", 1), ("x", 2)]
    cx, _cz = codes.bivariate_bicycle_code(6, 6, bb72_a, bb72_b)
    _bench("BB[[72,12]] X-sector", [list(map(int, qs)) for qs in cx.check_to_qubits],
           int(cx.n_qubits), args.p, args.shots, args.seed)

    # d=9 rotated-surface memory-X circuit-level DEM (9 rounds).
    import stim

    circuit = stim.Circuit.generated(
        "surface_code:rotated_memory_x", distance=9, rounds=9,
        after_clifford_depolarization=0.001,
        before_measure_flip_probability=0.001,
        after_reset_flip_probability=0.001,
    )
    model = from_stim(circuit.detector_error_model(decompose_errors=True))
    c2q = model.check_to_qubits()
    n_q = max((q for qs in c2q for q in qs), default=-1) + 1
    _bench("d=9 surface DEM (9 rounds)", c2q, n_q, args.p, args.dem_shots, args.seed)


if __name__ == "__main__":
    main()
