#!/usr/bin/env python
"""Calibrate the AutoDecoder CPU/GPU crossover and report the routing decisions.

For each distance it runs :meth:`AutoDecoder.calibrate` to find the smallest
batch size at which a GPU path beats the Rayon CPU path on *this* machine, then
records, for each requested batch size, which backend ``AutoDecoder.select``
would route to.  It also verifies that the AutoDecoder output is bit-identical to
the single-thread CPU decoder on a fixed batch (so routing never changes the
answer).

    python scripts/auto_backend_calibrate.py --distances 3 5 7 9 11 13 \
        --batches 64 256 1024 4096 16384 65536
"""
from __future__ import annotations

import argparse
import os
import sys
import time

_REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(_REPO, "python"))

import numpy as np  # noqa: E402

import qector_decoder_v3 as qd  # noqa: E402
from qector_decoder_v3 import benchmarking as bm  # noqa: E402
from qector_decoder_v3 import codes  # noqa: E402
from qector_decoder_v3.backend import AutoDecoder  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--distances", type=int, nargs="+", default=[3, 5, 7, 9, 11, 13])
    ap.add_argument("--batches", type=int, nargs="+",
                    default=[64, 256, 1024, 4096, 16384, 65536])
    ap.add_argument("--error-rate", type=float, default=0.05)
    ap.add_argument("--repeats", type=int, default=3)
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--out", default="benchmark_results/auto_backend_calibrate")
    args = ap.parse_args()

    env = bm.capture_environment()
    env["timestamp_unix"] = int(time.time())
    env["command"] = "auto_backend_calibrate " + " ".join(sys.argv[1:])

    rng = np.random.default_rng(args.seed)
    rows = []
    for d in args.distances:
        code = codes.rotated_surface_code(d)
        c2q, nq = code.check_to_qubits, code.n_qubits
        H = code.parity_check_matrix().astype(np.uint8)
        dec = AutoDecoder(c2q, nq)
        cal = dec.calibrate(sizes=tuple(b for b in args.batches if b >= 64) or (64,),
                            repeats=args.repeats, seed=args.seed)

        routing = {str(b): dec.select(b) for b in args.batches}

        # Correctness: AutoDecoder == single-thread CPU on a fixed mid-size batch.
        nbatch = min(2048, max(args.batches))
        err = (rng.random((nbatch, nq)) < args.error_rate).astype(np.uint8)
        syn = ((err @ H.T) & 1).astype(np.uint8)
        auto_out = np.asarray(dec.batch_decode(syn), dtype=np.uint8)
        single = qd.FastUnionFindDecoder(c2q, nq)
        ref = np.stack([np.asarray(single.decode(syn[i]), np.uint8) for i in range(nbatch)])
        bit_identical = bool(np.array_equal(auto_out, ref))
        faithful = bool(np.array_equal((auto_out @ H.T) & 1, syn))

        row = {
            "distance": d, "n_qubits": nq, "n_checks": code.n_checks,
            "available_backends": dec.available_backends(),
            "gpu_backend": cal.get("gpu_backend"),
            "crossover": cal.get("crossover"),
            "gpu_threshold": cal.get("gpu_threshold"),
            "routing": routing,
            "auto_bit_identical_to_cpu": bit_identical,
            "auto_syndrome_faithful": faithful,
        }
        rows.append(row)
        print(f"d={d:2d} gpu={cal.get('gpu_backend')} crossover={cal.get('crossover')} "
              f"routing={routing} identical={bit_identical}", flush=True)

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    bm.write_json(args.out + ".json", {"environment": env, "results": rows})
    print(f"wrote {args.out}.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
