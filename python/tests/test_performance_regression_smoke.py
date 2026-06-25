"""Section 9 (latency): a small end-to-end performance smoke mirroring a
competitive-benchmark sweep.

Benchmark blossom + union_find + sparse_blossom at d in (3, 5), n_trials=400,
and assert the structural health of every result: syndrome-faithful, monotonic
percentiles, positive throughput, and a recorded peak Python allocation.
NO wall-clock thresholds -- only invariants that hold by construction.
"""


from qector_decoder_v3 import benchmarking as bm, codes

KINDS = ["blossom", "union_find", "sparse_blossom"]
DISTANCES = [3, 5]


def test_performance_regression_smoke():
    results = []
    for d in DISTANCES:
        code = codes.rotated_surface_code(d)
        for kind in KINDS:
            r = bm.benchmark_decoder(kind, code, n_trials=400, warmup=60)
            results.append((kind, d, r))

    assert len(results) == len(KINDS) * len(DISTANCES)

    for kind, d, r in results:
        ctx = f"{kind} d={d}"

        # Gate: correctness of every benchmarked decode.
        assert r["syndrome_faithful"] is True, f"{ctx} not faithful"

        lat = r["latency_us"]
        # Monotone order statistics (by construction).
        assert lat["p50"] <= lat["p90"] <= lat["p95"] <= lat["p99"], ctx
        assert lat["min"] <= lat["p50"] <= lat["max"], ctx
        assert lat["ci95_low"] <= lat["mean"] <= lat["ci95_high"], ctx

        # Throughput must be strictly positive (it is 1/mean-ish, never <= 0).
        assert r["throughput_decodes_per_s"] > 0, ctx

        # Peak Python allocation is measured and reported.
        assert r["peak_python_alloc_kib"] is not None, ctx
        assert r["peak_python_alloc_kib"] >= 0, ctx
