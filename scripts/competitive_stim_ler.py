#!/usr/bin/env python
"""
Real Stim circuit-level head-to-head: QECTOR vs PyMatching.
Distance sweep with shot ladder + out-of-core decode_mmap support.

Generates rotated-surface-code memory circuits at circuit-level noise across a
distance sweep (default distances 3..200), builds the decoding problem from the
Stim Detector Error Model, and measures the **logical error rate** (with Wilson
95% intervals) and decode latency for:

  * QECTOR `BlossomDecoder` — weighted, exact polynomial MWPM (uses the DEM's
    ``log((1-p)/p)`` edge weights, like PyMatching).
  * QECTOR `UnionFindDecoder` — the fast near-linear path (unweighted).
  * `pymatching.Matching` — the reference weighted MWPM decoder.

All three decode the *same* sampled shots from the *same* DEM, so the LER
comparison is apples-to-apples.  For large syndrome arrays (>2 GB) the script
transparently uses memory-mapped out-of-core decoding so the full sweep fits in
any workstation's RAM.

Shot ladder
-----------
The ``--shots`` argument accepts:

* A single integer (e.g. ``--shots 50000``) — every distance gets the same count.
* A range ``MIN..MAX`` (e.g. ``--shots 5000..1000000``) — smaller distances get
  more shots for statistical power, larger distances get fewer.
* The word ``auto`` (default) — uses a built-in distance-aware table that targets
  k >= 100 logical errors per point at circuit-level noise 0.005.

Usage::

    python scripts/competitive_stim_ler.py \\
        --distances 3 5 7 9 11 13 15 17 19 21 25 31 41 51 71 101 151 200 \\
        --noise 0.005 --shots auto \\
        --out benchmark_results/competitive_stim_ler

    # Quick check: 3 distances, 20k shots each
    python scripts/competitive_stim_ler.py --distances 3 5 7 --shots 20000
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
import tempfile
import time

_REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(_REPO, "python"))

import numpy as np  # noqa: E402

import qector_decoder_v3 as qd  # noqa: E402
from qector_decoder_v3 import dem  # noqa: E402
from qector_decoder_v3 import benchmarking as bm  # noqa: E402
from qector_decoder_v3 import pymatching_compat  # noqa: E402

# Threshold for switching to out-of-core decode_mmap (2 GB of syndrome data).
_OOC_THRESHOLD_BYTES = 2 << 30  # 2 GB

# Default distance sweep for v0.7.0 full methodology.
_DEFAULT_DISTANCES = [
    3,
    5,
    7,
    9,
    11,
    13,
    15,
    17,
    19,
    21,
    25,
    31,
    41,
    51,
    71,
    101,
    151,
    200,
]

# Distance-to-shots mapping for ``--shots auto``.
# Targets k >= 100 logical errors at circuit-level noise 0.005.
_SHOT_LADDER = [
    (5, 20000),  # d <= 5:  high LER, 20k shots enough for good CI
    (9, 50000),  # d <= 9:  medium LER, 50k for tight CI
    (15, 100000),  # d <= 15: 100k shots
    (25, 200000),  # d <= 25: 200k shots
    (51, 500000),  # d <= 51: 500k shots
    (200, 1000000),  # d > 51:  1M shots for low-LER regime
]


def _auto_shots(distance: int) -> int:
    for max_d, shots in _SHOT_LADDER:
        if distance <= max_d:
            return shots
    return _SHOT_LADDER[-1][1]


def _parse_shots_arg(raw: str | int | None) -> str | int | tuple[int, int]:
    """Parse the ``--shots`` argument.

    Returns one of:
        * ``"auto"`` — use distance-dependent ladder
        * ``int`` — fixed shot count for all distances
        * ``(min_shots, max_shots)`` — range: smaller distances get max,
          larger get min (linearly interpolated in log-space)
    """
    if raw is None or raw == "auto":
        return "auto"
    if isinstance(raw, int):
        return raw
    s = str(raw)
    if ".." in s:
        parts = s.split("..", 1)
        return int(parts[0]), int(parts[1])
    return int(s)


def _shots_for_distance(distance: int, spec: str | int | tuple[int, int]) -> int:
    if spec == "auto":
        return _auto_shots(distance)
    if isinstance(spec, int):
        return spec
    lo, hi = spec
    if distance <= 5:
        return hi
    if distance >= 200:
        return lo
    # Log-linear interpolation between (d=5, hi) and (d=200, lo).
    log_hi = math.log(hi)
    log_lo = math.log(lo)
    t = (distance - 5) / 195.0
    return int(round(math.exp(log_hi + t * (log_lo - log_hi))))


def wilson(k: int, n: int, z: float = 1.959963985) -> tuple[float, float]:
    """Wilson 95% confidence interval for a binomial proportion."""
    if n == 0:
        return 0.0, 1.0
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    w = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return max(0.0, c - w), min(1.0, c + w)


def ler_for(predict_obs, dets, obs) -> tuple[int, float, float]:
    """Return (errors, latency_seconds_total, ler) for a predictor over shots."""
    actual = obs.astype(np.uint8)
    errors = 0
    t0 = time.perf_counter()
    preds = predict_obs(dets)
    dt = time.perf_counter() - t0
    preds = np.asarray(preds, dtype=np.uint8).reshape(len(dets), -1)
    for i in range(len(dets)):
        if not np.array_equal(preds[i], actual[i]):
            errors += 1
    return errors, dt, errors / len(dets)


def _use_ooc(n_shots: int, n_checks: int) -> bool:
    """True if the syndrome array exceeds the out-of-core threshold."""
    return n_shots * n_checks * np.dtype(np.uint8).itemsize > _OOC_THRESHOLD_BYTES


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--distances",
        type=int,
        nargs="+",
        default=_DEFAULT_DISTANCES,
        help="Distance values to sweep (default: 3..200 comprehensive)",
    )
    ap.add_argument("--noise", type=float, default=0.005)
    ap.add_argument(
        "--shots",
        default="auto",
        help="Shot count: int, MIN..MAX range, or 'auto' for distance-dependent ladder (default: auto)",
    )
    ap.add_argument("--rounds", type=int, default=0, help="0 => rounds = distance")
    ap.add_argument("--task", default="surface_code:rotated_memory_x")
    ap.add_argument(
        "--out",
        default="benchmark_results/competitive_stim_ler",
        help="Output prefix (default: benchmark_results/competitive_stim_ler)",
    )
    ap.add_argument(
        "--force-ooc",
        action="store_true",
        help="Force out-of-core decode_mmap even for small datasets",
    )
    args = ap.parse_args()

    import stim
    import pymatching

    env = bm.capture_environment()
    env["timestamp_unix"] = int(time.time())
    env["command"] = " ".join(sys.argv)
    env["stim_version"] = stim.__version__
    env["pymatching_version"] = pymatching.__version__

    shots_spec = _parse_shots_arg(args.shots)
    env["shots_spec"] = str(shots_spec)

    rows = []
    for d in args.distances:
        n_shots = _shots_for_distance(d, shots_spec)
        rounds = args.rounds or d
        circ = stim.Circuit.generated(
            args.task,
            distance=d,
            rounds=rounds,
            after_clifford_depolarization=args.noise,
            before_measure_flip_probability=args.noise,
            after_reset_flip_probability=args.noise,
        )
        sdem = circ.detector_error_model(decompose_errors=True)
        raw = dem.from_stim(sdem)
        model = raw.collapse_to_graph() if raw.is_graphlike else raw
        c2q, nq = model.check_to_qubits(), model.num_errors
        L = model.observables_matrix()

        sampler = circ.compile_detector_sampler()
        dets, obs = sampler.sample(shots=n_shots, separate_observables=True)
        dets = dets.astype(np.uint8)

        qm = pymatching_compat.Matching.from_detector_error_model(sdem)

        def qector_blossom_predict(dd, _m=qm):
            return _m.decode_batch(dd)

        uf = qd.UnionFindDecoder(c2q, nq)

        def qector_uf_predict(dd, _dec=uf, _L=L):
            corr = np.asarray(_dec.batch_decode(dd), dtype=np.uint8)
            return ((_L @ corr.T) & 1).T.astype(np.uint8)

        pm = pymatching.Matching.from_detector_error_model(sdem)

        def pymatching_predict(dd, _m=pm):
            return _m.decode_batch(dd)

        # C1-05: external reference decoders, optional — skipped with a note if
        # the package is absent, so the QECTOR/PyMatching table still generates.
        ldpc_predict = None
        try:
            from ldpc import BpOsdDecoder as _LdpcOsd

            _H = model.check_matrix()
            _priors = model.priors()
            _ldpc_dec = _LdpcOsd(
                _H,
                error_channel=list(_priors),
                max_iter=30,
                bp_method="product_sum",
                osd_method="OSD_CS",
                osd_order=0,
            )

            def ldpc_bposd_predict(dd, _dec=_ldpc_dec, _L=L):
                dd = np.asarray(dd, dtype=np.uint8)
                corr = np.stack([np.asarray(_dec.decode(s), dtype=np.uint8) for s in dd])
                return ((_L @ corr.T) & 1).T.astype(np.uint8)

            ldpc_predict = ldpc_bposd_predict
        except ImportError:
            print(f"d={d}: ldpc not installed — skipping ldpc_bposd column")

        belief_predict = None
        try:
            import beliefmatching as _bm

            _belief_dec = _bm.BeliefMatching(sdem)

            def beliefmatching_predict(dd, _dec=_belief_dec):
                dd = np.asarray(dd, dtype=np.uint8)
                return np.stack([np.asarray(_dec.decode(s), dtype=np.uint8) for s in dd])

            belief_predict = beliefmatching_predict
        except ImportError:
            print(f"d={d}: beliefmatching not installed — skipping beliefmatching column")

        n_checks = model.num_detectors
        ooc = args.force_ooc or _use_ooc(n_shots, n_checks)

        result = {
            "distance": d,
            "rounds": rounds,
            "noise": args.noise,
            "shots": n_shots,
            "detectors": n_checks,
            "raw_mechanisms": raw.num_errors,
            "collapsed_edges": model.num_errors,
            "graphlike": model.is_graphlike,
            "out_of_core": ooc,
        }

        if ooc:
            # Out-of-core path: write syndromes to a temp file,
            # use decode_mmap, then read corrections.
            for name, fn in [
                ("qector_blossom_weighted", qector_blossom_predict),
                ("qector_uf_predict", qector_uf_predict),
                ("pymatching", pymatching_predict),
            ]:
                tmpdir = tempfile.mkdtemp(prefix=f"comp_stim_d{d}_")
                syn_path = os.path.join(tmpdir, "syndromes.bin")
                corr_path = os.path.join(tmpdir, "corrections.bin")
                syn_mmap = np.memmap(syn_path, dtype=np.uint8, mode="w+", shape=(n_shots, n_checks))
                syn_mmap[:] = dets
                syn_mmap.flush()
                del syn_mmap

                qd.decode_mmap(
                    syn_path,
                    corr_path,
                    c2q,
                    nq,
                    decoder_type="cpu_batch",
                    batch_size=65536,
                    n_shots=n_shots,
                    verbose=False,
                )

                corr = np.memmap(corr_path, dtype=np.uint8, mode="r", shape=(n_shots, nq))
                k, dt = 0, 0.0
                t0 = time.perf_counter()
                for i in range(n_shots):
                    pred = (L @ corr[i].astype(np.uint8)) & 1
                    if pred != obs[i]:
                        k += 1
                dt = time.perf_counter() - t0

                lo, hi = wilson(k, n_shots)
                result[name] = {
                    "logical_errors": k,
                    "ler": k / n_shots,
                    "ler_ci95": [lo, hi],
                    "decode_mmap_us_per_shot": dt / n_shots * 1e6,
                }
                del corr
                for f in (syn_path, corr_path):
                    try:
                        os.remove(f)
                    except OSError:
                        pass
                try:
                    os.rmdir(tmpdir)
                except OSError:
                    pass
        else:
            # In-memory path.
            decoders_in_mem = [
                ("qector_blossom_weighted", qector_blossom_predict),
                ("qector_unionfind", qector_uf_predict),
                ("pymatching", pymatching_predict),
            ]
            if ldpc_predict is not None:
                decoders_in_mem.append(("ldpc_bposd", ldpc_predict))
            if belief_predict is not None:
                decoders_in_mem.append(("beliefmatching", belief_predict))
            for name, fn in decoders_in_mem:
                k, dt, ler = ler_for(fn, dets, obs)
                lo, hi = wilson(k, n_shots)
                result[name] = {
                    "logical_errors": k,
                    "ler": ler,
                    "ler_ci95": [lo, hi],
                    "decode_us_per_shot": dt / n_shots * 1e6,
                    "shots_per_s": n_shots / dt if dt > 0 else None,
                }

        rows.append(result)
        _print_result_row(result)
        # Save incrementally so partial output survives a timeout.
        _save_results(args.out, env, rows)
        print(flush=True)

    return 0


def _save_results(prefix: str, env: dict, rows: list[dict]) -> None:
    os.makedirs(os.path.dirname(prefix) or ".", exist_ok=True)
    with open(prefix + ".json", "w", encoding="utf-8") as fh:
        json.dump({"environment": env, "results": rows}, fh, indent=2)
    _write_markdown(prefix + ".md", env, rows)


def _print_result_row(result: dict) -> None:
    """Print a single result row to stdout."""
    parts = [
        f"d={result['distance']:3d}",
        f"rounds={result['rounds']:2d}",
        f"det={result['detectors']:5d}",
        f"shots={result['shots']:>7d}",
        f"ooc={'Y' if result.get('out_of_core') else 'N'}",
    ]
    for key, label in [
        ("qector_blossom_weighted", "QB"),
        ("qector_unionfind", "UF"),
        ("pymatching", "PM"),
        ("ldpc_bposd", "LDPC"),
        ("beliefmatching", "BM"),
    ]:
        r = result.get(key)
        if r is None:
            continue
        ler = r["ler"]
        lo, hi = r["ler_ci95"]
        us = r.get("decode_us_per_shot", r.get("decode_mmap_us_per_shot", 0))
        parts.append(f"{label} LER={ler:.5f} [{lo:.5f},{hi:.5f}] {us:.1f}us")
    print(" | ".join(parts))


def _write_markdown(path: str, env: dict, rows: list[dict]) -> None:
    if not rows:
        return
    stim_ver = env.get("stim_version", "?")
    pym_ver = env.get("pymatching_version", "?")
    lines = [
        "# Stim circuit-level head-to-head — QECTOR vs PyMatching",
        "",
        f"- Task: `surface_code:rotated_memory_x`, rounds = distance, "
        f"circuit-level depolarizing+measurement+reset noise p = {rows[0]['noise']}",
        "- Shots: distance-dependent ladder (auto) or fixed per `--shots`",
        "- Out-of-core decode_mmap used when syndrome data > 2 GB",
        f"- CPU: {env.get('processor') or env.get('platform')}; "
        f"Python {env.get('python_version')}; NumPy {env.get('numpy_version')}; "
        f"Stim {stim_ver}; PyMatching {pym_ver}",
        "",
        "LER with Wilson 95% interval; latency is per-shot decode time (hot path).",
        "For out-of-core runs, latency includes mmap overhead.",
        "",
        "| d | rounds | shots | det | QECTOR-Blossom LER | PyMatching LER | "
        "QECTOR-UF LER | ldpc BP-OSD LER | BeliefMatching LER | QB µs | PM µs | "
        "LDPC µs | BM µs |",
        "|---|--------|-------|-----|--------------------|----------------|"
        "---------------|------------------|---------------------|-------|"
        "-------|----------|-------|",
    ]

    def _ler_cell(r):
        if r is None:
            return "N/A"
        return f"{r['ler']:.5f} [{r['ler_ci95'][0]:.5f},{r['ler_ci95'][1]:.5f}]"

    def _us_cell(r):
        if r is None:
            return "N/A"
        return f"{r.get('decode_us_per_shot', r.get('decode_mmap_us_per_shot', 0)):.1f}"

    for r in rows:
        qb = r.get("qector_blossom_weighted")
        pm = r.get("pymatching")
        uf = r.get("qector_unionfind")
        lp = r.get("ldpc_bposd")
        bm_ = r.get("beliefmatching")
        uf_ler = f"{uf['ler']:.5f}" if uf else "N/A"
        lines.append(
            f"| {r['distance']} | {r['rounds']} | {r['shots']} | {r['detectors']} | "
            f"{_ler_cell(qb)} | {_ler_cell(pm)} | {uf_ler} | "
            f"{_ler_cell(lp)} | {_ler_cell(bm_)} | "
            f"{_us_cell(qb)} | {_us_cell(pm)} | {_us_cell(lp)} | {_us_cell(bm_)} |"
        )
    with open(path, "w", encoding="utf-8") as fh:
        fh.write("\n".join(lines) + "\n")


if __name__ == "__main__":
    raise SystemExit(main())
