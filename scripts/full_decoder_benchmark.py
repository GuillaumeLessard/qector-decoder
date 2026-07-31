#!/usr/bin/env python
"""Full decoder-family benchmark on real circuit-level syndromes.

Every measurement here comes from syndromes **sampled from a Stim circuit**, so
each one is physically realizable and has a valid correction. That matters more
than it sounds: the older `benchmarks_session/harnesses/*.py` scripts fed
decoders uniform-random bit patterns on a boundaryless ring code, where ~49% of
"syndromes" have odd defect parity and therefore admit no correction at all. A
Union-Find decoder answers such an input by growing clusters until they span the
whole graph, so those harnesses were timing a pathological non-decode -- roughly
2 ms/shot for every backend -- and reporting it as decoder throughput.

Measured per (distance, decoder):
  * LER      -- fraction of shots whose predicted logical observable is wrong,
                with a Wilson 95% interval, against the circuit's own recorded
                observable flips.
  * latency  -- microseconds per shot.
  * faithful -- fraction of shots where H @ correction == syndrome (mod 2).

Usage::

    python scripts/full_decoder_benchmark.py --distances 3 5 7 --shots 20000
    python scripts/full_decoder_benchmark.py --out benchmark_results/full
"""

from __future__ import annotations

import argparse
import json
import math
import os
import platform
import sys
import time
from datetime import datetime, timezone

_REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(_REPO, "python"))

os.environ.setdefault("QECTOR_SILENT", "1")

import numpy as np  # noqa: E402

import qector_decoder_v3 as qd  # noqa: E402
from qector_decoder_v3 import dem as demmod  # noqa: E402

# Decoder families built from the DEM via `DemModel.make_decoder`.
_DEM_KINDS = [
    "union_find",
    "fast_union_find",
    "blossom",
    "sparse_blossom",
    "bp_osd",
    "lookup_table",
    "hybrid_cascade",
    "ambiguity_cluster",
]


def wilson(successes: int, trials: int, z: float = 1.96) -> tuple[float, float]:
    """Wilson score interval. Correct at p=0 and p=1, unlike the normal
    approximation, which is exactly where low-LER measurements live."""
    if trials <= 0:
        return (0.0, 0.0)
    p = successes / trials
    denom = 1.0 + z * z / trials
    centre = (p + z * z / (2 * trials)) / denom
    margin = z * math.sqrt(max(p * (1 - p) / trials + z * z / (4 * trials * trials), 0.0)) / denom
    return (max(0.0, centre - margin), min(1.0, centre + margin))


def build_problem(distance: int, rounds: int, noise: float, shots: int, seed: int):
    """Sample real detector data + observable flips from a Stim surface code."""
    import stim

    circuit = stim.Circuit.generated(
        "surface_code:rotated_memory_x",
        distance=distance,
        rounds=rounds,
        after_clifford_depolarization=noise,
        before_measure_flip_probability=noise,
        after_reset_flip_probability=noise,
        before_round_data_depolarization=noise,
    )
    sampler = circuit.compile_detector_sampler(seed=seed)
    detectors, observables = sampler.sample(shots, separate_observables=True)
    model = demmod.from_stim(circuit.detector_error_model(decompose_errors=True))
    graph = model.collapse_to_graph()
    return circuit, graph, detectors.astype(np.uint8), observables.astype(np.uint8)


def score(graph, corrections: np.ndarray, syndromes: np.ndarray, observables: np.ndarray):
    """Return (logical_errors, faithful_frac) over the shots actually decoded."""
    n = corrections.shape[0]
    obs_matrix = graph.observables_matrix().astype(np.uint8)
    check_matrix = graph.check_matrix().astype(np.uint8)

    predicted = (corrections @ obs_matrix.T) & 1
    truth = observables[:n] if observables.ndim == 2 else observables[:n].reshape(-1, 1)
    truth = truth[:, : predicted.shape[1]]
    wrong = int((predicted != truth).any(axis=1).sum())

    reproduced = (corrections @ check_matrix.T) & 1
    faithful = float((reproduced == syndromes[:n]).all(axis=1).mean())
    return wrong, faithful


#: Shots used for the cost probe that sizes each decoder's run.
_PROBE_SHOTS = 128


def time_decoder(decoder, syndromes: np.ndarray, n_mech: int, repeats: int = 1,
                 budget_s: float = 0.0):
    """Decode shots, preferring a batch entry point.

    Returns ``(corrections, seconds, shots_used)``. When ``budget_s`` is set, a
    short probe estimates the per-shot cost and the shot count is trimmed so one
    slow decoder cannot stall the sweep. The trimmed count is returned and
    recorded, so a row measured on fewer shots is never presented as if it had
    the full count -- the widened Wilson interval shows the cost.
    """
    shots = syndromes.shape[0]
    batch_fn = None
    for method in ("batch_decode_par", "batch_decode"):
        fn = getattr(decoder, method, None)
        if fn is not None:
            batch_fn = fn
            break

    def run(n: int):
        if batch_fn is not None:
            return np.asarray(batch_fn(syndromes[:n]), dtype=np.uint8).reshape(n, n_mech)
        return np.array(
            [np.asarray(decoder.decode(syndromes[i]), dtype=np.uint8) for i in range(n)],
            dtype=np.uint8,
        ).reshape(n, n_mech)

    probe_n = min(_PROBE_SHOTS, shots)
    t0 = time.perf_counter()
    run(probe_n)  # doubles as warm-up
    probe_s = time.perf_counter() - t0

    use = shots
    if budget_s > 0 and probe_s > 0:
        per_shot = probe_s / probe_n
        affordable = int(budget_s / max(per_shot * repeats, 1e-12))
        use = max(probe_n, min(shots, affordable))

    best = math.inf
    out = None
    for _ in range(repeats):
        t0 = time.perf_counter()
        out = run(use)
        best = min(best, time.perf_counter() - t0)
    return out, best, use


def record(rows, *, distance, rounds, noise, shots, decoder, detectors, mechanisms,
           wrong=None, faithful=None, seconds=None, status="ok", error=""):
    row = {
        "distance": distance,
        "rounds": rounds,
        "noise": noise,
        "shots": shots,
        "decoder": decoder,
        "detectors": detectors,
        "mechanisms": mechanisms,
        "status": status,
    }
    if status == "ok":
        lo, hi = wilson(wrong, shots)
        row.update(
            {
                "logical_errors": wrong,
                "ler": wrong / shots,
                "ler_ci95_lo": lo,
                "ler_ci95_hi": hi,
                "faithful_frac": faithful,
                "latency_us": seconds * 1e6 / shots,
                "throughput_shots_s": shots / seconds,
            }
        )
    else:
        row["error"] = error
    rows.append(row)
    return row


def run_qector(rows, graph, syndromes, observables, *, distance, rounds, noise, shots, repeats, budget):
    n_mech = graph.num_errors
    for kind in _DEM_KINDS:
        label = f"qector:{kind}"
        try:
            decoder = graph.make_decoder(kind)
            corrections, seconds, used = time_decoder(decoder, syndromes, n_mech, repeats, budget)
            wrong, faithful = score(graph, corrections, syndromes, observables)
            r = record(
                rows, distance=distance, rounds=rounds, noise=noise, shots=used,
                decoder=label, detectors=graph.num_detectors, mechanisms=n_mech,
                wrong=wrong, faithful=faithful, seconds=seconds,
            )
            print(f"    {label:28s} LER {r['ler']:.5f}  {r['latency_us']:9.2f} us/shot  "
                  f"faithful {r['faithful_frac']*100:6.2f}%  n={used:,}", flush=True)
        except Exception as exc:
            record(rows, distance=distance, rounds=rounds, noise=noise, shots=shots,
                   decoder=label, detectors=graph.num_detectors, mechanisms=n_mech,
                   status="failed", error=f"{type(exc).__name__}: {exc}")
            print(f"    {label:28s} FAILED  {type(exc).__name__}: {str(exc)[:70]}", flush=True)


def run_gpu(rows, graph, syndromes, observables, *, distance, rounds, noise, shots, repeats, budget):
    """GPU backends take the same check->mechanism adjacency as the CPU core."""
    c2q = graph.check_to_qubits()
    n_mech = graph.num_errors
    mean_p = max(float(graph.priors().mean()), 1e-3)
    # The GPU union-find backends now accept the same DEM matching weights the
    # CPU decoders get. Both spellings are measured: the unweighted one is the
    # historical behaviour and is kept so the accuracy cost of decoding
    # weight-blind stays visible rather than being quietly retired.
    weights = graph.weights().tolist()
    candidates = [
        ("gpu:cuda_union_find", lambda: qd.CUDABatchDecoder(c2q, n_mech)),
        ("gpu:cuda_union_find_weighted", lambda: qd.CUDABatchDecoder(c2q, n_mech, weights)),
        ("gpu:cuda_bp_osd", lambda: qd.CUDABpOsdDecoder(c2q, n_mech, mean_p)),
        ("gpu:opencl_union_find", lambda: qd.OpenCLBatchDecoder(c2q, n_mech)),
        ("gpu:opencl_union_find_weighted", lambda: qd.OpenCLBatchDecoder(c2q, n_mech, weights)),
    ]
    for label, ctor in candidates:
        try:
            decoder = ctor()
        except Exception as exc:
            record(rows, distance=distance, rounds=rounds, noise=noise, shots=shots,
                   decoder=label, detectors=graph.num_detectors, mechanisms=n_mech,
                   status="unavailable", error=f"{type(exc).__name__}: {exc}")
            print(f"    {label:28s} unavailable ({type(exc).__name__}: {str(exc)[:52]})", flush=True)
            continue
        try:
            corrections, seconds, used = time_decoder(decoder, syndromes, n_mech, repeats, budget)
            wrong, faithful = score(graph, corrections, syndromes, observables)
            r = record(
                rows, distance=distance, rounds=rounds, noise=noise, shots=used,
                decoder=label, detectors=graph.num_detectors, mechanisms=n_mech,
                wrong=wrong, faithful=faithful, seconds=seconds,
            )
            print(f"    {label:28s} LER {r['ler']:.5f}  {r['latency_us']:9.2f} us/shot  "
                  f"faithful {r['faithful_frac']*100:6.2f}%  n={used:,}", flush=True)
        except Exception as exc:
            record(rows, distance=distance, rounds=rounds, noise=noise, shots=shots,
                   decoder=label, detectors=graph.num_detectors, mechanisms=n_mech,
                   status="failed", error=f"{type(exc).__name__}: {exc}")
            print(f"    {label:28s} FAILED  {type(exc).__name__}: {str(exc)[:70]}", flush=True)


def run_competitors(rows, circuit, graph, syndromes, observables, *, distance, rounds,
                    noise, shots, repeats):
    n_mech = graph.num_errors

    # PyMatching, built from the same Stim DEM.
    label = "competitor:pymatching"
    try:
        import pymatching

        matching = pymatching.Matching.from_detector_error_model(
            circuit.detector_error_model(decompose_errors=True)
        )
        matching.decode_batch(syndromes[:8])
        best = math.inf
        predicted = None
        for _ in range(repeats):
            t0 = time.perf_counter()
            predicted = matching.decode_batch(syndromes)
            best = min(best, time.perf_counter() - t0)
        predicted = np.asarray(predicted, dtype=np.uint8)
        truth = observables[:, : predicted.shape[1]]
        wrong = int((predicted != truth).any(axis=1).sum())
        lo, hi = wilson(wrong, shots)
        rows.append({
            "distance": distance, "rounds": rounds, "noise": noise, "shots": shots,
            "decoder": label, "detectors": graph.num_detectors, "mechanisms": n_mech,
            "status": "ok", "logical_errors": wrong, "ler": wrong / shots,
            "ler_ci95_lo": lo, "ler_ci95_hi": hi,
            # PyMatching returns observable predictions directly, not a
            # mechanism-space correction, so faithfulness is not defined for it.
            "faithful_frac": None,
            "latency_us": best * 1e6 / shots, "throughput_shots_s": shots / best,
        })
        print(f"    {label:28s} LER {wrong/shots:.5f}  {best*1e6/shots:9.2f} us/shot  faithful      n/a", flush=True)
    except Exception as exc:
        record(rows, distance=distance, rounds=rounds, noise=noise, shots=shots,
               decoder=label, detectors=graph.num_detectors, mechanisms=n_mech,
               status="unavailable", error=f"{type(exc).__name__}: {exc}")
        print(f"    {label:28s} unavailable ({type(exc).__name__}: {str(exc)[:52]})")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--distances", type=int, nargs="+", default=[3, 5, 7, 9])
    ap.add_argument("--shots", type=int, default=20000)
    ap.add_argument("--noise", type=float, default=0.005)
    ap.add_argument("--repeats", type=int, default=3, help="timed repeats; the fastest is reported")
    ap.add_argument("--seed", type=int, default=20260730)
    ap.add_argument("--out", default=os.path.join(_REPO, "benchmark_results", "full_decoder_benchmark"))
    ap.add_argument("--budget", type=float, default=25.0,
                    help="seconds per decoder per distance; shots are trimmed to fit "
                         "(0 disables). Keeps one slow decoder from stalling the sweep.")
    ap.add_argument("--skip-gpu", action="store_true")
    ap.add_argument("--skip-competitors", action="store_true")
    args = ap.parse_args()

    info = qd.get_license_info()
    print("=" * 96)
    print(f"QECTOR {qd.__version__} full decoder benchmark -- real Stim circuit-level syndromes")
    print(f"license: tier={info.get('tier')} max_distance={info.get('max_distance')} "
          f"gpu={info.get('gpu_enabled')} key_status={info.get('key_status')}")
    print(f"noise={args.noise}  shots={args.shots}  repeats={args.repeats}  seed={args.seed}")
    print("=" * 96)

    rows: list[dict] = []
    for d in args.distances:
        rounds = d
        circuit, graph, syndromes, observables = build_problem(
            d, rounds, args.noise, args.shots, args.seed + d
        )
        print(f"\n--- d={d}, rounds={rounds}: {graph.num_detectors} detectors, "
              f"{graph.num_errors} mechanisms, {args.shots} shots ---", flush=True)
        run_qector(rows, graph, syndromes, observables, distance=d, rounds=rounds,
                   noise=args.noise, shots=args.shots, repeats=args.repeats,
                   budget=args.budget)
        if not args.skip_gpu:
            run_gpu(rows, graph, syndromes, observables, distance=d, rounds=rounds,
                    noise=args.noise, shots=args.shots, repeats=args.repeats,
                    budget=args.budget)
        if not args.skip_competitors:
            run_competitors(rows, circuit, graph, syndromes, observables, distance=d,
                            rounds=rounds, noise=args.noise, shots=args.shots, repeats=args.repeats)

    payload = {
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "qector_version": qd.__version__,
        "license": info,
        "environment": {
            "python": sys.version.split()[0],
            "platform": platform.platform(),
            "processor": platform.processor(),
            "numpy": np.__version__,
        },
        "parameters": {
            "circuit": "surface_code:rotated_memory_x",
            "noise": args.noise,
            "shots": args.shots,
            "repeats": args.repeats,
            "seed": args.seed,
            "distances": args.distances,
        },
        "results": rows,
    }

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    json_path = f"{args.out}.json"
    with open(json_path, "w", encoding="utf-8") as fh:
        json.dump(payload, fh, indent=2)

    csv_path = f"{args.out}.csv"
    columns = ["distance", "rounds", "noise", "shots", "decoder", "detectors", "mechanisms",
               "status", "logical_errors", "ler", "ler_ci95_lo", "ler_ci95_hi",
               "faithful_frac", "latency_us", "throughput_shots_s", "error"]
    with open(csv_path, "w", encoding="utf-8", newline="") as fh:
        fh.write(",".join(columns) + "\n")
        for r in rows:
            fh.write(",".join(str(r.get(c, "")).replace(",", ";") for c in columns) + "\n")

    ok = sum(1 for r in rows if r["status"] == "ok")
    print(f"\n{ok}/{len(rows)} measurements succeeded")
    print(f"wrote {json_path}")
    print(f"wrote {csv_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
