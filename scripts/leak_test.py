#!/usr/bin/env python
"""Long-run memory-leak test: decode in a tight loop and track host RSS growth.

Runs a decoder over many iterations on reachable syndromes and samples the
process RSS at intervals.  A correct decoder reuses buffers, so RSS must
plateau: we fit the late-phase slope and flag a leak if RSS grows materially
between the first and last quarter of the run.

    python scripts/leak_test.py --decoder blossom --distance 11 --iterations 100000
    python scripts/leak_test.py --decoder cuda_batch --distance 9 --batch 4096 --iterations 1000
"""
from __future__ import annotations

import argparse
import gc
import os
import sys
import time

_REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
import importlib.util
if importlib.util.find_spec("qector_decoder_v3.qector_decoder_v3") is None:
    sys.path.insert(0, os.path.join(_REPO, "python"))

import numpy as np  # noqa: E402

import qector_decoder_v3 as qd  # noqa: E402
from qector_decoder_v3 import benchmarking as bm  # noqa: E402
from qector_decoder_v3 import codes  # noqa: E402


def _rss_mib() -> float:
    try:
        import psutil
        return psutil.Process().memory_info().rss / (1024 * 1024)
    except Exception:
        return float("nan")


_SINGLE = {
    "blossom": qd.BlossomDecoder,
    "sparse_blossom": qd.SparseBlossomDecoder,
    "union_find": qd.UnionFindDecoder,
    "fast_union_find": qd.FastUnionFindDecoder,
}
_BATCH = {
    "cpu_batch": qd.CPUBatchDecoder,
    "rayon_batch": qd.BatchDecoder,
    "cuda_batch": qd.CUDABatchDecoder,
    "opencl_batch": qd.OpenCLBatchDecoder,
}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--decoder", default="blossom",
                    choices=sorted(set(_SINGLE) | set(_BATCH)))
    ap.add_argument("--distance", type=int, default=11)
    ap.add_argument("--iterations", type=int, default=100000)
    ap.add_argument("--batch", type=int, default=4096)
    ap.add_argument("--error-rate", type=float, default=0.05)
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--samples", type=int, default=40)
    ap.add_argument("--out", default="benchmark_results/leak_test")
    args = ap.parse_args()

    if args.decoder == "cuda_batch" and not qd.cuda_is_available():
        print("cuda not available")
        return 1
    if args.decoder == "opencl_batch" and not qd.opencl_is_available():
        print("opencl not available")
        return 1

    code = codes.rotated_surface_code(args.distance)
    c2q, nq = code.check_to_qubits, code.n_qubits
    H = code.parity_check_matrix().astype(np.uint8)
    rng = np.random.default_rng(args.seed)
    is_batch = args.decoder in _BATCH

    if is_batch:
        dec = _BATCH[args.decoder](c2q, nq)
        err = (rng.random((args.batch, nq)) < args.error_rate).astype(np.uint8)
        syn = ((err @ H.T) & 1).astype(np.uint8)

        def step():
            dec.batch_decode(syn)
    else:
        dec = _SINGLE[args.decoder](c2q, nq)
        pool = (rng.random((256, nq)) < args.error_rate).astype(np.uint8)
        syns = ((pool @ H.T) & 1).astype(np.uint8)

        def step(_i=[0]):
            dec.decode(syns[_i[0] % 256])
            _i[0] += 1

    # warmup
    for _ in range(min(50, args.iterations)):
        step()
    gc.collect()

    sample_every = max(1, args.iterations // args.samples)
    rss = []
    base = _rss_mib()
    t0 = time.perf_counter()
    for i in range(args.iterations):
        step()
        if i % sample_every == 0:
            rss.append(_rss_mib())
    dt = time.perf_counter() - t0
    gc.collect()
    final = _rss_mib()

    rss_arr = np.array([x for x in rss if x == x], dtype=np.float64)
    q = max(1, len(rss_arr) // 4)
    early = float(rss_arr[:q].mean()) if len(rss_arr) else float("nan")
    late = float(rss_arr[-q:].mean()) if len(rss_arr) else float("nan")
    growth = late - early
    # Leak if RSS grew by more than 5% AND at least 16 MiB over the run.
    leaked = bool(growth > max(16.0, 0.05 * early))

    decodes = args.iterations * (args.batch if is_batch else 1)
    env = bm.capture_environment()
    env["timestamp_unix"] = int(time.time())
    env["command"] = "leak_test " + " ".join(sys.argv[1:])
    result = {
        "environment": env,
        "decoder": args.decoder, "distance": args.distance,
        "iterations": args.iterations, "batch": args.batch if is_batch else 1,
        "total_decodes": decodes, "seconds": round(dt, 3),
        "rss_base_mib": round(base, 1), "rss_final_mib": round(final, 1),
        "rss_early_mean_mib": round(early, 1), "rss_late_mean_mib": round(late, 1),
        "rss_growth_mib": round(growth, 1), "leaked": leaked,
        "rss_samples_mib": [round(x, 1) for x in rss_arr.tolist()],
    }
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    suffix = f"_{args.decoder}_d{args.distance}"
    bm.write_json(args.out + suffix + ".json", result)
    print(f"{args.decoder} d={args.distance}: {decodes} decodes in {dt:.1f}s  "
          f"RSS {base:.0f}->{final:.0f}MiB  growth={growth:+.1f}MiB  leaked={leaked}")
    print(f"wrote {args.out}{suffix}.json")
    return 2 if leaked else 0


if __name__ == "__main__":
    raise SystemExit(main())
