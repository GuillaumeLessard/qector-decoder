#!/usr/bin/env python3
"""
example_advanced_decoders.py — belief-matching, BP-OSD, and Sinter.

Demonstrates the decoders that go beyond plain MWPM:
  * BeliefMatching  — BP-reweighted exact MWPM (lower LER than PyMatching),
  * BpOsdDecoder    — BP + ordered statistics for LDPC / qLDPC codes,
  * Sinter plug-in  — QECTOR as a sinter.Decoder.

Heavy/optional steps degrade gracefully when stim/pymatching are absent, so the
example always runs.
"""
import sys

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

import numpy as np
from qector_decoder_v3 import codes
from qector_decoder_v3.bposd import BpOsdDecoder


def demo_bposd():
    print("=" * 60)
    print("BP-OSD on a bivariate-bicycle [[72,12]] LDPC code")
    print("=" * 60)
    cx, _ = codes.bivariate_bicycle_code(
        6, 6, [("x", 3), ("y", 1), ("y", 2)], [("y", 3), ("x", 1), ("x", 2)]
    )
    H = cx.parity_check_matrix()
    dec = BpOsdDecoder(H, error_rate=0.03, max_iter=30, osd_order=5)
    rng = np.random.default_rng(0)
    faithful = 0
    for _ in range(200):
        e = (rng.random(H.shape[1]) < 0.03).astype(np.uint8)
        s = (H @ e) & 1
        c = np.asarray(dec.decode(s)).astype(np.uint8)
        faithful += int(np.array_equal((H @ c) & 1, s))
    print(f"qubits={H.shape[1]} checks={H.shape[0]}  faithful decodes: {faithful}/200")


def demo_belief_matching():
    try:
        import stim  # noqa: F401
        from qector_decoder_v3.belief_matching import BeliefMatching
        import pymatching
    except Exception:
        print("\n(stim/pymatching not installed — skipping belief-matching demo)")
        return
    print("\n" + "=" * 60)
    print("Belief-matching vs PyMatching (rotated surface d=3, circuit-level)")
    print("=" * 60)
    circ = stim.Circuit.generated(
        "surface_code:rotated_memory_x", distance=3, rounds=3,
        after_clifford_depolarization=0.01,
        before_measure_flip_probability=0.01,
        after_reset_flip_probability=0.01,
    )
    sdem = circ.detector_error_model(decompose_errors=True)
    det, obs = circ.compile_detector_sampler(seed=1).sample(
        shots=2000, separate_observables=True)
    det = det.astype(np.uint8)
    obs = obs.astype(np.uint8)
    pm = pymatching.Matching.from_detector_error_model(sdem)
    bm = BeliefMatching.from_detector_error_model(sdem)
    pm_err = int(np.any(np.asarray(pm.decode_batch(det)) != obs, axis=1).sum())
    bm_err = int(np.any(np.asarray(bm.decode_batch(det)) != obs, axis=1).sum())
    print(f"PyMatching errors={pm_err}  belief-matching errors={bm_err}  (2000 shots)")


def main():
    demo_bposd()
    demo_belief_matching()
    print("\nDone.")


if __name__ == "__main__":
    main()
