"""A1-04 / A2-02: noise-model integrity of the LER harness.

The defect these lock down: `run_competitive_suite` used to measure QECTOR with
`estimate_ler` (code-capacity — i.i.d. data-qubit flips, one syndrome, no
measurement error) while measuring PyMatching on a circuit-level Stim circuit
with `rounds=d`, then print both in one table. Those are different experiments,
and the comparison flattered QECTOR by orders of magnitude.
"""

from __future__ import annotations

import numpy as np
import pytest
from qector_decoder_v3 import codes, ler

stim = pytest.importorskip("stim", reason="circuit-level tests need stim")


# ---------------------------------------------------------------------------
# Noise-model tagging and the mixing guard (A1-01)
# ---------------------------------------------------------------------------
def test_code_capacity_results_are_tagged():
    r = ler.estimate_ler(codes.repetition_code(5), "union_find", p=0.05, shots=200, seed=1)
    assert r.noise_model == ler.CODE_CAPACITY
    assert r.to_dict()["noise_model"] == ler.CODE_CAPACITY


def test_circuit_level_results_are_tagged():
    r = ler.estimate_ler_circuit_level(distance=3, decoder="qector_blossom", p=0.003, shots=200, seed=1)
    assert r.noise_model == ler.CIRCUIT_LEVEL
    assert r.rounds == 3


def test_mixing_noise_models_is_refused():
    cc = ler.estimate_ler(codes.repetition_code(5), "union_find", p=0.05, shots=200, seed=1)
    cl = ler.estimate_ler_circuit_level(distance=3, decoder="qector_blossom", p=0.003, shots=200, seed=1)
    with pytest.raises(ler.NoiseModelMismatch, match="different noise models"):
        ler.assert_comparable([cc, cl])


def test_same_model_comparison_is_allowed():
    a = ler.estimate_ler_circuit_level(distance=3, decoder="qector_blossom", p=0.003, shots=200, seed=1)
    b = ler.estimate_ler_circuit_level(distance=3, decoder="qector_unionfind", p=0.003, shots=200, seed=1)
    assert ler.assert_comparable([a, b]) == ler.CIRCUIT_LEVEL


def test_untagged_result_is_refused():
    with pytest.raises(ler.NoiseModelMismatch, match="no noise_model"):
        ler.assert_comparable([{"decoder": "mystery"}])


# ---------------------------------------------------------------------------
# The two models must be measurably different (A1: this is *why* they can't mix)
# ---------------------------------------------------------------------------
def test_circuit_level_is_materially_harder_than_code_capacity():
    """At the same nominal p, circuit-level LER must exceed code-capacity LER.

    If this ever fails, the two estimators have converged and the whole premise
    of the separation needs re-examining.
    """
    p = 0.005
    cc = ler.estimate_ler(codes.rotated_surface_code(5), "blossom", p=p, shots=4000, seed=7)
    cl = ler.estimate_ler_circuit_level(distance=5, decoder="qector_blossom", p=p, shots=4000, seed=7)
    assert cl.ler > cc.ler, (
        f"circuit-level ({cl.ler:.5f}) must be worse than code-capacity ({cc.ler:.5f}) "
        f"at p={p} — these are not interchangeable measurements"
    )


# ---------------------------------------------------------------------------
# The suite itself (A1-02)
# ---------------------------------------------------------------------------
def test_competitive_suite_is_internally_comparable():
    rows = ler.run_competitive_suite(
        p=0.003, shots=1000, distances=(3,), decoders=("qector_blossom", "qector_unionfind")
    )
    assert rows, "suite produced no rows"
    assert {r["noise_model"] for r in rows} == {ler.CIRCUIT_LEVEL}
    ler.assert_comparable(rows)  # must not raise


def test_competitive_suite_reports_skipped_baselines():
    """A silently missing reference is how misleading tables get published.

    Every row carries the list, so a table rendered from the output can always
    say which baselines were absent rather than quietly omitting them.
    """
    rows = ler.run_competitive_suite(
        p=0.003, shots=500, distances=(3,), decoders=("qector_blossom", "qector_unionfind")
    )
    assert rows
    assert "skipped_baselines" in rows[0]
    assert isinstance(rows[0]["skipped_baselines"], list)


def test_competitive_suite_fails_loudly_on_an_unknown_decoder():
    """A typo'd decoder name must not be silently skipped.

    Only a *missing optional dependency* (ImportError) is tolerated; an
    unrecognised name is a mistake in the benchmark definition and would
    otherwise produce a table quietly missing a row.
    """
    with pytest.raises(ValueError, match="definitely_not_a_decoder"):
        ler.run_competitive_suite(p=0.003, shots=200, distances=(3,), decoders=("definitely_not_a_decoder",))


def test_qector_blossom_matches_pymatching_at_circuit_level():
    """PyMatching parity, measured honestly: same circuit, same samples, same scoring."""
    pytest.importorskip("pymatching")
    shots = 4000
    q = ler.estimate_ler_circuit_level(distance=5, decoder="qector_blossom", p=0.003, shots=shots, seed=11)
    pm = ler.estimate_ler_circuit_level(distance=5, decoder="pymatching", p=0.003, shots=shots, seed=11)
    q_lo, q_hi = q.ci95
    p_lo, p_hi = pm.ci95
    assert max(q_lo, p_lo) <= min(q_hi, p_hi), (
        f"QECTOR blossom LER {q.ler:.5f} {q.ci95} and PyMatching {pm.ler:.5f} {pm.ci95} "
        "have disjoint 95% CIs at identical circuit-level noise"
    )


# ---------------------------------------------------------------------------
# Biased noise (A2-02)
# ---------------------------------------------------------------------------
def test_biased_noise_eta_reaches_the_sampler():
    """Regression: `eta` used to be computed into a discarded expression."""
    rng = np.random.default_rng(0)
    n = 200_000

    z, x = ler.sample_biased_errors(rng, n, 0.1, eta=1.0, return_both_sectors=True)
    assert x.mean() == 0.0, "eta=1.0 is pure Z — no X errors may be sampled"
    assert 0.095 < z.mean() < 0.105

    z, x = ler.sample_biased_errors(rng, n, 0.1, eta=0.5, return_both_sectors=True)
    assert 0.095 < x.mean() < 0.105, "eta=0.5 is unbiased — p_x must equal p_z"

    z, x = ler.sample_biased_errors(rng, n, 0.05, eta=0.8, return_both_sectors=True)
    expected_px = 0.05 * (1 - 0.8) / 0.8  # 0.0125
    assert abs(x.mean() - expected_px) < 0.002, f"p_x={x.mean():.5f}, expected ~{expected_px}"


def test_biased_noise_rejects_out_of_range_eta():
    rng = np.random.default_rng(0)
    for bad in (0.0, -0.1, 1.5):
        with pytest.raises(ValueError, match="eta"):
            ler.sample_biased_errors(rng, 10, 0.1, eta=bad)


def test_biased_noise_default_returns_single_vector():
    """Back-compat: the default return shape is unchanged."""
    rng = np.random.default_rng(0)
    out = ler.sample_biased_errors(rng, 64, 0.1)
    assert isinstance(out, np.ndarray)
    assert out.shape == (64,)
    assert out.dtype == np.uint8
