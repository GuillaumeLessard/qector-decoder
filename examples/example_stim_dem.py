#!/usr/bin/env python3
"""
example_stim_dem.py — Stim DEM -> QECTOR -> logical observables.

Loads a detector error model (here from inline `.dem` text so the example runs
without Stim installed; the same call accepts a live ``stim.DetectorErrorModel``),
builds a QECTOR decoder over the detector graph, decodes, and reads off the
predicted logical-observable flips.
"""
import sys

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

import numpy as np
from qector_decoder_v3 import dem


DEM_TEXT = """
error(0.05) D0 L0
error(0.05) D0 D1
error(0.05) D1 D2
error(0.05) D2 D3
error(0.05) D3 D4
error(0.05) D4 L0
"""


def main():
    model = dem.parse_dem(DEM_TEXT)            # or dem.from_stim(stim_dem_object)
    print("=" * 60)
    print("Stim DEM -> QECTOR")
    print("=" * 60)
    print(model)
    print("check matrix (detectors x mechanisms):")
    print(model.check_matrix())
    print("observables matrix:", model.observables_matrix().tolist())

    decoder = model.make_decoder("sparse_blossom")
    H = model.check_matrix()
    rng = np.random.default_rng(1)
    shots = 5000
    logical_errors = 0
    for _ in range(shots):
        err = (rng.random(model.num_errors) < 0.05).astype(np.uint8)
        syndrome = (H @ err) & 1
        correction = np.asarray(decoder.decode(syndrome.astype(np.uint8))).astype(np.uint8)
        assert np.array_equal((H @ correction) & 1, syndrome)  # faithful
        actual = model.predicted_observables(err)
        predicted = model.predicted_observables(correction)
        if not np.array_equal(actual, predicted):
            logical_errors += 1
    print(f"\nshots={shots}  logical error rate = {logical_errors / shots:.4f}")


if __name__ == "__main__":
    main()
