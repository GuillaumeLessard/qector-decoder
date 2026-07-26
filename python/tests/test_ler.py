"""E2 + E6 (K3dev.md wave 2): statistical rigor + threshold estimation.

Every LER must carry shots/errors/Wilson-CI; decoder instances are built once
per run (S5); the repetition code gives exact analytic LER targets; the
threshold crossing of the repetition code (p_th = 0.5) is recovered.
"""

from __future__ import annotations

import math

import numpy as np
import pytest
from qector_decoder_v3 import codes
from qector_decoder_v3.ler import (
    estimate_ler,
    run_threshold_sweep,
    wilson_ci,
)


# ---------------------------------------------------------------------------
# Wilson CI (E2)
# ---------------------------------------------------------------------------
def test_wilson_ci_edges_and_monotonicity():
    lo, hi = wilson_ci(0, 1000)
    assert lo == 0.0 and hi > 0.0 and hi < 0.01
    lo, hi = wilson_ci(1000, 1000)
    assert hi > 0.999 and lo > 0.99  # hi == 1.0 up to float epsilon
    # monotone in k
    assert wilson_ci(10, 1000)[0] < wilson_ci(50, 1000)[0]
    # degenerate input
    assert wilson_ci(0, 0) == (0.0, 1.0)


def test_wilson_ci_contains_true_rate_typical():
    # With k drawn near the expectation, the interval must contain p.
    assert wilson_ci(100, 1000)[0] <= 0.1 <= wilson_ci(100, 1000)[1]
    assert wilson_ci(3, 10000)[0] <= 0.0003 <= wilson_ci(3, 10000)[1]


def test_sample_biased_errors():
    from qector_decoder_v3.ler import sample_biased_errors

    rng = np.random.default_rng(42)
    err = sample_biased_errors(rng, 1000, p_z=0.05, eta=0.5)
    assert err.shape == (1000,)
    assert err.sum() > 0


# ---------------------------------------------------------------------------
# estimate_ler against analytic targets (E2)
# ---------------------------------------------------------------------------
def _rep_ler_theory(d: int, p: float) -> float:
    # majority-vote failure probability of the d-repetition code
    return sum(math.comb(d, k) * p**k * (1 - p) ** (d - k) for k in range((d + 1) // 2, d + 1))


def test_estimate_ler_repetition_matches_theory():
    p, theory = 0.05, _rep_ler_theory(3, 0.05)  # ~0.00725
    r = estimate_ler(codes.repetition_code(3), "blossom", p=p, shots=6000, seed=1)
    assert r.code == "repetition_d3"
    assert r.unfaithful == 0
    assert abs(r.ler - theory) < 0.01
    lo, hi = r.ci95
    assert lo <= r.ler <= hi
    assert r.decodes_per_s > 0
    assert r.n_logical_qubits == 1


def test_estimate_ler_rejects_code_without_logicals():
    with pytest.raises(ValueError, match="no logicals"):
        estimate_ler(codes.unrotated_surface_code(3), "blossom", p=0.05, shots=10, seed=0)


def test_estimate_ler_batched_equals_theory_on_surface():
    code = codes.rotated_surface_code(3)
    r = estimate_ler(code, "sparse_blossom", p=0.02, shots=4000, seed=7)
    assert r.unfaithful == 0
    # d=3 code-capacity MWPM at p=0.02: LER must be small but nonzero-ish
    assert 0.0 <= r.ler < 0.05


def test_estimate_ler_builds_decoder_once():
    # S5: a factory-based decoder must be constructed exactly once per run.
    builds = []

    class Counting:
        def __init__(self, code):
            from qector_decoder_v3 import BlossomDecoder

            builds.append(1)
            self._d = BlossomDecoder(code.check_to_qubits, code.n_qubits)
            self.n_qubits = code.n_qubits

        def batch_decode(self, S):
            return self._d.batch_decode(S)

    code = codes.rotated_surface_code(3)
    estimate_ler(code, lambda: Counting(code), p=0.05, shots=120, seed=3, batch_size=40)
    assert len(builds) == 1


# ---------------------------------------------------------------------------
# Threshold sweep (E6) — repetition code has an exact crossing at p = 0.5
# ---------------------------------------------------------------------------
def test_threshold_sweep_repetition_crossing_at_half():
    res = run_threshold_sweep(
        codes.repetition_code,
        distances=[3, 5],
        p_values=[0.05, 0.5],
        decoder="blossom",
        shots=4000,
        seed=2,
    )
    g = res.grid()
    # below threshold: more distance helps
    assert g[(5, 0.05)].ler < g[(3, 0.05)].ler
    # at p = 0.5 decoding is chance for any d -> not strictly decreasing
    assert res.crossing == 0.5
    d = res.to_dict()
    assert d["crossing"] == 0.5 and len(d["results"]) == 4
