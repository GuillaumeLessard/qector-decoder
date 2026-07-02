"""Section 9 (latency): benchmark_decoder latency percentiles are order-statistic
monotonic for blossom / sparse_blossom / union_find on the rotated surface code.

We ONLY assert facts that hold by construction (p50<=p90<=p95<=p99, min<=p50<=max,
ci95_low<=mean<=ci95_high) plus syndrome-faithfulness. No wall-clock thresholds:
those would flake under machine load.

NOTE: ``rotated_surface_code`` takes a single ``distance`` argument in this build,
so the spec's "(5,7)" is realized as two distances, d=5 and d=7.
"""

import pytest

import qector_decoder_v3 as qd
from qector_decoder_v3 import benchmarking as bm, codes

KINDS = ["blossom", "sparse_blossom", "union_find"]
DISTANCES = [5, 7]


@pytest.mark.parametrize("kind", KINDS)
@pytest.mark.parametrize("distance", DISTANCES)
def test_latency_percentiles_monotonic(kind, distance):
    code = codes.rotated_surface_code(distance)
    result = bm.benchmark_decoder(kind, code, n_trials=600, warmup=80)

    # Every benchmarked decode must be correct (the gate behind the numbers).
    assert result["syndrome_faithful"] is True

    lat = result["latency_us"]
    assert isinstance(lat, dict)
    assert lat["n"] > 0

    # Order statistics hold by construction.
    assert lat["p50"] <= lat["p90"] <= lat["p95"] <= lat["p99"]
    assert lat["min"] <= lat["p50"] <= lat["max"]

    # The mean lies inside its own 95% confidence interval by construction.
    assert lat["ci95_low"] <= lat["mean"] <= lat["ci95_high"]

    # median == p50 in this summary; both must be finite and non-negative.
    assert lat["median"] == lat["p50"]
    assert lat["min"] >= 0.0
