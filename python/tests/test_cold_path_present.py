"""Section 9 (latency): benchmark_decoder reports the COLD path (decoder
construction time) separately from the HOT path (decode latency).

We assert ``cold_path_us`` is present, is a numeric summary dict with the same
summary keys as ``latency_us``, and is a distinct measurement (different sample
count and own values) -- proving construction cost is reported apart from the
steady-state decode cost. No wall-clock thresholds are asserted.
"""

import numbers

import pytest

from qector_decoder_v3 import benchmarking as bm, codes

SUMMARY_KEYS = ("mean", "median", "p50", "p99")


@pytest.mark.parametrize("kind", ["blossom", "sparse_blossom", "union_find"])
def test_cold_path_present(kind):
    code = codes.rotated_surface_code(5)
    result = bm.benchmark_decoder(kind, code, n_trials=500, warmup=60)

    assert "cold_path_us" in result
    cold = result["cold_path_us"]
    hot = result["latency_us"]

    assert isinstance(cold, dict)
    assert isinstance(hot, dict)

    # Same summary schema as the hot path.
    for key in SUMMARY_KEYS:
        assert key in cold, f"cold path missing summary key {key!r}"
        assert key in hot
        assert isinstance(cold[key], numbers.Real)
        assert cold[key] >= 0.0

    # Cold-path order statistics are internally consistent (by construction).
    assert cold["p50"] <= cold["p99"]
    assert cold["median"] == cold["p50"]

    # Cold (construction) is a SEPARATE measurement from hot (decode):
    # it has its own (smaller) sample count, reported independently.
    assert cold["n"] >= 1
    assert hot["n"] >= 1
    assert cold["n"] != hot["n"]
