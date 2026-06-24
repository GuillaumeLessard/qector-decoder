#!/usr/bin/env python3
"""
example_pymatching_and_backend.py — drop-in Matching + automatic backend.

Part 1: use QECTOR through a PyMatching-compatible ``Matching`` API.
Part 2: let ``AutoDecoder`` pick CPU / Rayon / CUDA / OpenCL by batch size.
"""
import sys

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

import numpy as np
from qector_decoder_v3 import codes
from qector_decoder_v3.pymatching_compat import Matching
from qector_decoder_v3.backend import AutoDecoder, BackendConfig


def main():
    code = codes.rotated_surface_code(5)
    H = code.parity_check_matrix()
    L = code.logicals_matrix()

    print("=" * 60)
    print("PyMatching-compatible Matching")
    print("=" * 60)
    m = Matching.from_check_matrix(H, faults_matrix=L)
    print(repr(m))
    rng = np.random.default_rng(0)
    err = (rng.random(code.n_qubits) < 0.08).astype(np.uint8)
    s = (H @ err) & 1
    print("predicted observables:", m.decode(s).tolist())
    print("edge correction      :", m.decode_to_edges_array(s).tolist())

    print("\n" + "=" * 60)
    print("AutoDecoder backend selection")
    print("=" * 60)
    dec = AutoDecoder(code.check_to_qubits, code.n_qubits)
    print("available backends:", dec.available_backends())
    for n in (1, 16, 4096, 100000):
        print(f"  batch={n:>6d} -> would run on {dec.select(n)}")
    syns = (rng.random((2000, code.n_checks)) < 0.08).astype(np.uint8)
    out = dec.batch_decode(syns)
    ok = all(np.array_equal((H @ out[i]) & 1, syns[i]) for i in range(len(syns)))
    print(f"decoded {len(syns)} shots on {dec.last_backend}; all faithful = {ok}")


if __name__ == "__main__":
    main()
