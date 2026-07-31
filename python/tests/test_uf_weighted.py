"""UF-01 regression: Union-Find must consume the DEM's per-mechanism weights.

Before UF-01, ``dem.make_decoder`` handed ``log((1-p)/p)`` weights to Blossom and
SparseBlossom but constructed ``UnionFindDecoder(c2q, nq)`` with none — so the
flagship decoders were topology-only. Measured against PyMatching 2 on identical
circuit-level DEMs that cost 4.5-9.1x in logical error rate, because an
unweighted decoder cannot distinguish a ``p = 1e-4`` mechanism from a
``p = 1e-2`` one.

These tests assert the properties that fix has to keep:
  * weights actually reach the core (``is_weighted``, and bad input is rejected);
  * a weighted decode is still syndrome-faithful;
  * weighted strictly beats unweighted on a real circuit-level DEM;
  * ``make_decoder`` and the Sinter resolver both pass the weights through.
"""

from __future__ import annotations

import numpy as np
import pytest
from qector_decoder_v3 import FastUnionFindDecoder, UnionFindDecoder

stim = pytest.importorskip("stim")


# ---------------------------------------------------------------------------
# Plumbing
# ---------------------------------------------------------------------------
def _ring(n):
    return [[i, (i + 1) % n] for i in range(n)]


@pytest.mark.parametrize("cls", [UnionFindDecoder, FastUnionFindDecoder])
def test_weights_reach_the_core(cls):
    c2q = _ring(8)
    assert cls(c2q, 8).is_weighted is False
    assert cls(c2q, 8, edge_weights=[1.0] * 8).is_weighted is True


@pytest.mark.parametrize("cls", [UnionFindDecoder, FastUnionFindDecoder])
def test_bad_weights_are_rejected_not_ignored(cls):
    c2q = _ring(8)
    with pytest.raises(ValueError, match="length"):
        cls(c2q, 8, edge_weights=[1.0] * 5)
    with pytest.raises(ValueError, match="finite"):
        cls(c2q, 8, edge_weights=[1.0, float("nan")] + [1.0] * 6)
    with pytest.raises(ValueError, match="finite"):
        cls(c2q, 8, edge_weights=[float("inf")] + [1.0] * 7)


@pytest.mark.parametrize("cls", [UnionFindDecoder, FastUnionFindDecoder])
def test_set_edge_weights_in_place(cls):
    dec = cls(_ring(8), 8)
    assert dec.is_weighted is False
    dec.set_edge_weights([2.0] * 8)
    assert dec.is_weighted is True


def test_weighted_decode_is_syndrome_faithful():
    n = 24
    c2q = _ring(n)
    H = np.zeros((n, n), dtype=np.uint8)
    for ci, qs in enumerate(c2q):
        for q in qs:
            H[ci, q] ^= 1
    rng = np.random.default_rng(0)
    w = 0.5 + 8.0 * rng.random(n)
    dec = UnionFindDecoder(c2q, n, edge_weights=w)
    err = (rng.random((200, n)) < 0.1).astype(np.uint8)
    syn = ((err @ H.T) & 1).astype(np.uint8)
    corr = np.asarray(dec.batch_decode(syn), dtype=np.uint8)
    assert np.array_equal((corr @ H.T) & 1, syn), "weighted UF returned an unfaithful correction"


def test_weighted_prefers_the_cheap_path():
    """Two equal-hop arcs; only the weights can break the tie."""
    n = 8
    c2q = _ring(n)
    w = np.ones(n)
    w[[1, 2, 3, 4]] = 40.0  # arc 0->4 going up is 40x less likely
    dec = UnionFindDecoder(c2q, n, edge_weights=w)
    syn = np.zeros(n, dtype=np.uint8)
    syn[0] = 1
    syn[4] = 1
    corr = np.asarray(dec.decode(syn), dtype=np.uint8)
    assert corr[[1, 2, 3, 4]].sum() == 0, f"took the expensive arc: {corr}"
    assert corr[[5, 6, 7, 0]].sum() == 4, f"did not take the cheap arc: {corr}"


# ---------------------------------------------------------------------------
# Accuracy on a real circuit-level DEM
# ---------------------------------------------------------------------------
def _circuit(d, p):
    return stim.Circuit.generated(
        "surface_code:rotated_memory_z",
        distance=d,
        rounds=d,
        after_clifford_depolarization=p,
        before_measure_flip_probability=p,
        after_reset_flip_probability=p,
        before_round_data_depolarization=p,
    )


def _ler(dec, det, obs, L):
    corr = np.asarray(dec.batch_decode(det), dtype=np.uint8)
    pred = ((L @ corr.T) & 1).T.astype(np.uint8)
    return float(np.any(pred != obs, axis=1).mean())


@pytest.mark.parametrize("d", [5, 7])
def test_weighted_uf_beats_unweighted_on_circuit_level_dem(d):
    """The UF-01 acceptance bar: strictly better LER on an identical DEM.

    Measured at 20k shots and p=0.003, the weighted decoder is ~2.4-2.7x more
    accurate at these distances, so a 20% margin is a wide safety band around a
    real effect rather than a threshold tuned to one run.
    """
    from qector_decoder_v3.dem import from_stim

    p, shots = 0.003, 20000
    circ = _circuit(d, p)
    dem = circ.detector_error_model(decompose_errors=True)
    model = from_stim(dem)
    if model.is_graphlike:
        model = model.collapse_to_graph()
    c2q, nq = model.check_to_qubits(), model.num_errors
    L = model.observables_matrix()
    w = model.weights().tolist()

    det, obs = circ.compile_detector_sampler(seed=7).sample(shots, separate_observables=True)
    det = np.ascontiguousarray(det.astype(np.uint8))
    obs = np.ascontiguousarray(obs.astype(np.uint8))

    ler_unweighted = _ler(UnionFindDecoder(c2q, nq), det, obs, L)
    ler_weighted = _ler(UnionFindDecoder(c2q, nq, edge_weights=w), det, obs, L)

    assert ler_weighted < ler_unweighted * 0.8, (
        f"d={d}: weighted UF LER {ler_weighted:.5f} is not clearly better than unweighted {ler_unweighted:.5f}"
    )


@pytest.mark.parametrize("d", [5, 7])
def test_weighted_uf_is_within_3x_of_pymatching(d):
    """UF-01 targeted "within ~2x of PyMatching"; this asserts a 3x guard rail.

    Union-Find is expected to stay somewhat behind exact MWPM — that is the
    algorithm's trade, and it buys throughput. What must not come back is the
    4.5-9.1x gap that came from ignoring the weights entirely.
    """
    pymatching = pytest.importorskip("pymatching")
    from qector_decoder_v3.dem import from_stim

    p, shots = 0.003, 20000
    circ = _circuit(d, p)
    dem = circ.detector_error_model(decompose_errors=True)
    model = from_stim(dem)
    if model.is_graphlike:
        model = model.collapse_to_graph()
    L = model.observables_matrix()

    det, obs = circ.compile_detector_sampler(seed=7).sample(shots, separate_observables=True)
    det = np.ascontiguousarray(det.astype(np.uint8))
    obs = np.ascontiguousarray(obs.astype(np.uint8))

    dec = UnionFindDecoder(model.check_to_qubits(), model.num_errors, edge_weights=model.weights().tolist())
    ler_uf = _ler(dec, det, obs, L)

    pm = pymatching.Matching.from_detector_error_model(dem)
    pm_pred = np.asarray(pm.decode_batch(det), dtype=np.uint8)
    ler_pm = float(np.any(pm_pred != obs, axis=1).mean())

    assert ler_pm > 0, "degenerate comparison: PyMatching made no logical errors"
    assert ler_uf <= ler_pm * 3.0, f"d={d}: weighted UF LER {ler_uf:.5f} is more than 3x PyMatching's {ler_pm:.5f}"


# ---------------------------------------------------------------------------
# Resolvers must pass the weights through
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("kind", ["union_find", "fast_union_find"])
def test_make_decoder_threads_weights_into_uf(kind):
    from qector_decoder_v3.dem import from_stim

    dem = _circuit(5, 0.003).detector_error_model(decompose_errors=True)
    model = from_stim(dem).collapse_to_graph()
    assert model.make_decoder(kind).is_weighted is True
    assert model.make_decoder(kind, weighted=False).is_weighted is False


def test_sinter_resolver_uses_weighted_uf_by_default():
    from qector_decoder_v3.sinter_compat import _build_matcher

    dem = _circuit(5, 0.003).detector_error_model(decompose_errors=True)
    assert _build_matcher("unionfind", dem)._dec.is_weighted is True
    assert _build_matcher("unionfind_unweighted", dem)._dec.is_weighted is False
