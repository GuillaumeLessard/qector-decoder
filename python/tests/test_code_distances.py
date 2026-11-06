"""1.0.1 regression tests: every built-in code family has the distance its
name claims, and the Union-Find decoders correct every error of weight up to
floor((d-1)/2).

1.0.0 shipped rotated/unrotated/heavy-hex/colour generators that did not
match their labels (every rotated "d" code had distance 2, the unrotated and
colour generators encoded no logical qubit, heavy-hex "d" had distance 2d-1), and a
Union-Find peel that could miscorrect weight-1 errors next to the boundary.
These tests pin both fixes.
"""
from __future__ import annotations

import itertools

import numpy as np
import pytest

import qector_decoder_v3 as q
from qector_decoder_v3 import codes


FAMILIES = [
    ("rotated", codes.rotated_surface_code, range(2, 12)),
    ("unrotated", codes.unrotated_surface_code, range(2, 9)),
    ("heavy_hex_z", lambda d: codes.heavy_hex_code(d, "z"), range(3, 12, 2)),
    ("heavy_hex_x", lambda d: codes.heavy_hex_code(d, "x"), range(3, 12, 2)),
    ("toric", codes.toric_code, range(2, 8)),
    ("repetition", codes.repetition_code, range(2, 12)),
    ("ring", codes.ring_code, range(3, 12)),
]


@pytest.mark.parametrize(
    "family,make,d",
    [(f, m, d) for f, m, ds in FAMILIES for d in ds],
    ids=lambda v: str(v) if not callable(v) else "",
)
def test_exact_distance_matches_label(family, make, d):
    code = make(d)
    res = codes.code_distance(code)
    assert res["exact"], res
    assert res["distance"] == d, f"{family} d={d}: true distance {res['distance']}"
    # The witness is a logical operator of exactly that weight.
    w = np.zeros(code.n_qubits, np.uint8)
    w[list(res["witness"])] = 1
    H = code.parity_check_matrix()
    L = code.logicals_matrix()
    assert not ((H @ w) & 1).any()
    assert ((L @ w) & 1).any()
    assert int(w.sum()) == d


@pytest.mark.parametrize("d", [3, 5, 7])
def test_colour_code_parameters(d):
    code = codes.color_code(d)
    assert code.n_qubits == (3 * d * d + 1) // 4
    assert code.num_logical_qubits == 1
    res = codes.code_distance(code)
    assert res["exact"] and res["distance"] == d


@pytest.mark.parametrize("d", [3, 5, 7])
def test_qubit_counts(d):
    assert codes.rotated_surface_code(d).n_qubits == d * d
    assert codes.unrotated_surface_code(d).n_qubits == d * d + (d - 1) ** 2
    for c in (codes.rotated_surface_code(d), codes.unrotated_surface_code(d)):
        assert c.num_logical_qubits == 1


def test_code_distance_lower_bound_when_capped():
    res = codes.code_distance(codes.color_code(7), max_weight=3)
    assert not res["exact"]
    assert res["distance"] is None
    assert res["lower_bound"] == 4


UF_CASES = [
    (name, make, d, cls)
    for name, make in [
        ("rotated", codes.rotated_surface_code),
        ("unrotated", codes.unrotated_surface_code),
        ("toric", codes.toric_code),
        ("heavy_hex_x", lambda d: codes.heavy_hex_code(d, "x")),
    ]
    for d in (3, 5)
    for cls in ("UnionFindDecoder", "FastUnionFindDecoder")
]


@pytest.mark.parametrize("name,make,d,cls", UF_CASES, ids=lambda v: str(v) if not callable(v) else "")
def test_union_find_corrects_all_errors_up_to_half_distance(name, make, d, cls):
    code = make(d)
    H = code.parity_check_matrix()
    L = code.logicals_matrix()
    dec = getattr(q, cls)(code.check_to_qubits, n_qubits=code.n_qubits)
    t = (d - 1) // 2
    for w in range(1, t + 1):
        for support in itertools.combinations(range(code.n_qubits), w):
            e = np.zeros(code.n_qubits, np.uint8)
            e[list(support)] = 1
            s = (H @ e) & 1
            corr = np.asarray(dec.decode(s), np.uint8)
            assert not (((H @ corr) & 1) ^ s).any(), f"unfaithful on {support}"
            assert not ((L @ (corr ^ e)) & 1).any(), f"{cls} logical failure on {support}"


@pytest.mark.parametrize("decoder", ["UnionFindDecoder", "BlossomDecoder"])
def test_logical_error_rate_falls_with_distance(decoder):
    rng = np.random.default_rng(11)
    rates = []
    for d in (3, 5, 7, 9):
        code = codes.rotated_surface_code(d)
        H = code.parity_check_matrix()
        L = code.logicals_matrix()
        dec = getattr(q, decoder)(code.check_to_qubits, n_qubits=code.n_qubits)
        E = (rng.random((3000, code.n_qubits)) < 0.04).astype(np.uint8)
        S = (E @ H.T) & 1
        fails = 0
        for e, s in zip(E, S):
            fails += int(((L @ (np.asarray(dec.decode(s), np.uint8) ^ e)) & 1).any())
        rates.append(fails / len(E))
    assert all(a > b for a, b in zip(rates, rates[1:])), rates


def test_blossom_matches_pymatching_exactly():
    pymatching = pytest.importorskip("pymatching")
    code = codes.rotated_surface_code(7)
    H = code.parity_check_matrix()
    L = code.logicals_matrix()
    rng = np.random.default_rng(5)
    E = (rng.random((2000, code.n_qubits)) < 0.05).astype(np.uint8)
    S = (E @ H.T) & 1
    ours = q.BlossomDecoder(code.check_to_qubits, n_qubits=code.n_qubits)
    ref = pymatching.Matching(H)
    for s in S:
        a = np.asarray(ours.decode(s), np.uint8)
        b = np.asarray(ref.decode(s), np.uint8)
        # Minimum-weight corrections may differ by a stabiliser; logical class and weight may not.
        assert int(a.sum()) == int(b.sum())
        assert not ((L @ (a ^ b)) & 1).any()
