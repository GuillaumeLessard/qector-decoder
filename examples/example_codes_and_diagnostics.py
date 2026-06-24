#!/usr/bin/env python3
"""
example_codes_and_diagnostics.py — code-family helpers + rich decode results.

Shows the new `codes` module (build any standard family in one call) and the
`DecodeResult` diagnostics (sparse / bit-packed / logical flips / timing / JSON).
"""
import sys

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

import numpy as np
from qector_decoder_v3 import codes
from qector_decoder_v3.result import decode_with_diagnostics


def main():
    print("=" * 60)
    print("QECTOR — code families")
    print("=" * 60)
    for code in [
        codes.repetition_code(11),
        codes.rotated_surface_code(7),
        codes.unrotated_surface_code(5),
        codes.toric_code(5),
        codes.heavy_hex_code(5),
    ]:
        print(f"{code.name:22s} qubits={code.n_qubits:3d} checks={code.n_checks:3d} "
              f"matching_graph={code.is_matching_graph()}")

    print("\n" + "=" * 60)
    print("Decode diagnostics (rotated surface d=7)")
    print("=" * 60)
    code = codes.rotated_surface_code(7)
    rng = np.random.default_rng(0)
    e = code.random_error(0.08, rng)
    s = code.syndrome(e)
    res = decode_with_diagnostics(code, s, kind="blossom")
    print(res.explain())
    print("sparse indices :", res.sparse_indices.tolist())
    print("bit-packed     :", res.bit_packed.tolist())
    print("JSON           :", res.to_json()[:120], "...")


if __name__ == "__main__":
    main()
