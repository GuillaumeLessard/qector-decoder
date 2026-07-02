#!/usr/bin/env python3
"""
example_auto_routing.py — QECTOR Decoder v3 (v0.5.8) decoder auto-routing

Demonstrates ``qector_decoder_v3.routing``:

  * ``recommend_decoder`` — pure, side-effect-free policy that maps a decoding
    problem (code family / distance / qubit count / batch size) and a goal
    (accuracy / speed / balanced) onto a concrete decoder *name*.
  * ``AutoRouter`` — constructs, caches, and dispatches the chosen decoder, and
    inspects the ACTUAL check structure so a non-graphlike (qLDPC / hypergraph)
    problem is always routed to BP-OSD — never to a matching-only decoder that
    could not satisfy H·c == s on hyperedges.

Run:
    PYTHONPATH=python python examples/example_auto_routing.py
"""

import sys

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

import numpy as np

from qector_decoder_v3 import codes, routing


def main():
    print("=" * 70)
    print("QECTOR v3 (0.5.8) — Decoder Auto-Routing")
    print("=" * 70)
    hw = routing.detect_hardware()
    print(f"\nDetected hardware: cuda_rust={hw.cuda_rust}  gpu(cupy)={hw.gpu}")

    # ---- 1. Pure-metadata recommendations --------------------------------
    print("\n1. recommend_decoder(...) — pure policy by family / priority / batch:")
    scenarios = [
        dict(code_family="surface", distance=5, priority="accuracy"),
        dict(code_family="surface", distance=21, priority="accuracy"),
        dict(code_family="repetition", distance=9, priority="speed"),
        dict(code_family="surface", distance=11, batch_size=100_000, priority="speed"),
        dict(code_family="qldpc", batch_size=4096, priority="balanced"),
        dict(code_family="bivariate_bicycle", batch_size=1, priority="accuracy"),
        dict(code_family="hypergraph_product", priority="accuracy"),
    ]
    for sc in scenarios:
        name = routing.recommend_decoder(**sc)
        label = ", ".join(f"{k}={v}" for k, v in sc.items())
        print(f"   [{label}]")
        print(f"       -> {name}")

    # ---- 2. Rich recommendation with reasoning ---------------------------
    print("\n2. recommend(...) — full reasoning for a qLDPC batch:")
    rec = routing.recommend(code_family="qldpc", batch_size=8192, priority="balanced")
    print(f"   decoder        : {rec.decoder}")
    print(f"   family         : {rec.family}")
    print(f"   gpu_batched_bp : {rec.gpu_batched_bp}")
    print(f"   reason         : {rec.reason}")

    # ---- 3. AutoRouter across real codes — structural guard --------------
    print("\n3. AutoRouter.decode(...) across real codes (inspects actual H):")
    router = routing.AutoRouter(priority="accuracy")
    rng = np.random.default_rng(0)

    surface = codes.rotated_surface_code(5)
    repetition = codes.repetition_code(9)
    # X sector of the [[72,12,6]] bivariate-bicycle code — non-graphlike qLDPC.
    bb_x, _ = codes.bivariate_bicycle_code(
        6, 6, [("x", 3), ("y", 1), ("y", 2)], [("y", 3), ("x", 1), ("x", 2)]
    )
    # Deliberately MISLABEL the qLDPC code as "surface" to prove the guard.
    examples = [
        ("rotated_surface d=5", surface, {"code_family": "surface", "distance": 5}),
        ("repetition d=9", repetition, {"code_family": "repetition", "distance": 9}),
        ("bivariate-bicycle [[72,12,6]] (mislabelled 'surface')", bb_x,
         {"code_family": "surface"}),
    ]

    all_valid = True
    for label, code, ctx in examples:
        H = code.parity_check_matrix()
        err = code.random_error(0.05, rng)
        syn = (H @ err) & 1
        corr = router.decode(code, syn, **ctx)
        rec = router.last_recommendation
        valid = bool(np.array_equal((H @ corr) & 1, syn))
        all_valid = all_valid and valid
        print(f"\n   {label}")
        print(f"       graphlike(matching)? {code.is_matching_graph()}")
        print(f"       routed -> {rec.decoder}  (forced={rec.forced})")
        print(f"       H·c == s ? {valid}")
        if rec.forced:
            print(f"       guard reason: {rec.reason}")

    # ---- 4. Same router, batch of qLDPC syndromes ------------------------
    print("\n4. AutoRouter on a BATCH of qLDPC syndromes (routes to BP-OSD):")
    Hbb = bb_x.parity_check_matrix()
    errs = (rng.random((64, bb_x.n_qubits)) < 0.04).astype(np.uint8)
    syns = (errs @ Hbb.T) & 1
    corrs = router.decode(bb_x, syns, code_family="qldpc")
    batch_valid = bool(np.all(((corrs @ Hbb.T) & 1) == syns))
    print(f"   batch shape: {syns.shape} -> corrections {corrs.shape}")
    print(f"   routed -> {router.last_recommendation.decoder} "
          f"(gpu_batched_bp={router.last_recommendation.gpu_batched_bp})")
    print(f"   H·c == s on all 64 shots ? {batch_valid}")
    all_valid = all_valid and batch_valid

    print("\n" + "=" * 70)
    print(f"Result: {'every routed correction valid (H·c == s). OK.' if all_valid else 'VALIDITY FAILURE'}")
    print("=" * 70)
    if not all_valid:
        sys.exit(1)


if __name__ == "__main__":
    main()
