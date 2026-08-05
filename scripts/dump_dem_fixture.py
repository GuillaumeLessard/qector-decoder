"""Dump a real circuit-level DEM (collapsed) + sampled syndromes as a compact
text fixture the Rust benches can read, so core optimisation is measured against
the graph the flagship actually decodes rather than a ring toy.

Read by ``src/dem_fixture.rs``. The benches that use it
(``fast_uf::bench::bench_circuit_level_dem``) skip with a printed notice when no
fixtures are present, so this is a developer tool, not a build dependency.

Usage
-----
    python scripts/dump_dem_fixture.py <distance> <p> <shots> <out.txt>

    # generate the standard set the benches look for, then point Rust at it
    for d in 5 7 11; do
        python scripts/dump_dem_fixture.py $d 0.003 2000 /tmp/fx/dem_d$d.txt
    done
    QECTOR_DEM_FIXTURE_DIR=/tmp/fx cargo test --release --no-default-features \\
        -- --nocapture bench_circuit_level_dem

Format (whitespace separated, one section after another):
  line 1: n_checks n_qubits n_shots
  next n_checks lines: <k> q0 q1 ... q(k-1)          (check -> mechanism ids)
  next 1 line: n_qubits floats                        (per-mechanism weight log((1-p)/p))
  next n_shots lines: <k> d0 d1 ...                   (defect detector ids per shot)
"""
import os
import sys

os.environ.setdefault("QECTOR_SILENT", "1")
import numpy as np
import stim
from qector_decoder_v3.dem import from_stim


def main():
    d = int(sys.argv[1]) if len(sys.argv) > 1 else 7
    p = float(sys.argv[2]) if len(sys.argv) > 2 else 0.003
    shots = int(sys.argv[3]) if len(sys.argv) > 3 else 2000
    out = sys.argv[4]

    circ = stim.Circuit.generated(
        "surface_code:rotated_memory_z", distance=d, rounds=d,
        after_clifford_depolarization=p, before_measure_flip_probability=p,
        after_reset_flip_probability=p, before_round_data_depolarization=p,
    )
    dem = circ.detector_error_model(decompose_errors=True)
    model = from_stim(dem)
    if model.is_graphlike:
        model = model.collapse_to_graph()
    c2q = model.check_to_qubits()
    w = model.weights()
    det = np.ascontiguousarray(
        circ.compile_detector_sampler(seed=1).sample(shots).astype(np.uint8)
    )

    with open(out, "w") as fh:
        fh.write(f"{len(c2q)} {model.num_errors} {shots}\n")
        fh.writelines(f"{len(row)} " + " ".join(map(str, row)) + "\n" for row in c2q)
        fh.write(" ".join(f"{x:.9g}" for x in w) + "\n")
        for s in det:
            idx = np.flatnonzero(s)
            fh.write(f"{len(idx)} " + " ".join(map(str, idx)) + "\n")
    print(
        f"wrote {out}: d={d} p={p} checks={len(c2q)} mechanisms={model.num_errors} "
        f"shots={shots} mean_defects={det.sum()/shots:.2f}"
    )


if __name__ == "__main__":
    main()
