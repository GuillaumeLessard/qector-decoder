"""Scaling/benchmark example from docs/SCALING.md, executed for real.

docs/SCALING.md drives ``benchmark_decoder`` across a distance sweep and reads
``latency_us``, ``cold_path_us``, ``n_qubits`` etc. from the report. This test
runs that benchmark for rotated surface codes at d in (3, 5, 7) and asserts the
report's structural and order-statistic invariants:

* latency percentiles are monotone (p50 <= p90 <= p95 <= p99) by construction,
* ``cold_path_us`` (construction cost) is present,
* the run is syndrome-faithful, and
* ``n_qubits`` grows with distance (d^2 for the rotated surface code).

Only order-statistic and faithfulness facts are asserted — never wall-clock
thresholds — so the test does not flake under machine load.
"""

import numpy as np
import pytest

from qector_decoder_v3 import benchmarking as bm
from qector_decoder_v3 import codes


@pytest.fixture(scope="module")
def reports():
    out = {}
    for d in (3, 5, 7):
        code = codes.rotated_surface_code(d)
        out[d] = bm.benchmark_decoder(
            "blossom", code, n_trials=300, warmup=50, seed=1234
        )
    return out


@pytest.mark.parametrize("d", [3, 5, 7])
def test_report_structure_and_monotone_percentiles(reports, d):
    rep = reports[d]

    assert "latency_us" in rep
    lat = rep["latency_us"]
    for key in ("p50", "p90", "p95", "p99"):
        assert key in lat, f"latency_us missing {key}"
        assert np.isfinite(lat[key])

    # Order statistics are monotone by construction.
    assert lat["p50"] <= lat["p90"] <= lat["p95"] <= lat["p99"], lat

    # Construction cost report present and well-formed.
    assert "cold_path_us" in rep
    cold = rep["cold_path_us"]
    assert "median" in cold and np.isfinite(cold["median"])
    assert cold["median"] >= 0.0

    # The benchmark verified syndrome faithfulness internally.
    assert rep["syndrome_faithful"] is True

    # n_qubits is the d^2 of the rotated surface code.
    assert rep["n_qubits"] == d * d, (rep["n_qubits"], d)


def test_n_qubits_grows_with_distance(reports):
    nq = [reports[d]["n_qubits"] for d in (3, 5, 7)]
    assert nq[0] < nq[1] < nq[2], f"n_qubits not increasing with d: {nq}"
    # n_checks should also be non-decreasing with distance.
    nc = [reports[d]["n_checks"] for d in (3, 5, 7)]
    assert nc[0] <= nc[1] <= nc[2], f"n_checks not monotone with d: {nc}"
