"""E1 (K3dev.md wave 1): logical operators for qLDPC / CSS codes.

Verifies that the GF(2) kernel/quotient machinery in ``codes.py`` produces
correct, biorthogonal logical-operator bases and that every CSS generator
exposes them uniformly (C6): bivariate-bicycle, bicycle, hypergraph-product
and toric codes. Known values locked: BB[[72,12]], BB[[144,12]], toric k=2.
"""

from __future__ import annotations

import numpy as np
import pytest
from qector_decoder_v3 import codes
from qector_decoder_v3.codes import css_logicals, gf2_kernel, gf2_rank

BB72_A = [("x", 3), ("y", 1), ("y", 2)]
BB72_B = [("y", 3), ("x", 1), ("x", 2)]


def _dense(logicals, n):
    L = np.zeros((len(logicals), n), dtype=np.uint8)
    for i, qs in enumerate(logicals):
        for q in qs:
            L[i, q] ^= np.uint8(1)
    return L


# ---------------------------------------------------------------------------
# GF(2) primitives
# ---------------------------------------------------------------------------
def test_gf2_kernel_and_rank_basic():
    H = np.array([[1, 1, 0], [0, 1, 1]], dtype=np.uint8)
    K = gf2_kernel(H)
    assert gf2_rank(H) == 2
    assert K.shape == (1, 3)
    assert not np.any((H @ K.T) % 2)
    assert K[0].tolist() == [1, 1, 1]


def test_css_logicals_rejects_non_css_pair():
    Hx = np.array([[1, 1, 0]], dtype=np.uint8)
    Hz = np.array([[1, 0, 0]], dtype=np.uint8)  # overlaps Hx on 1 qubit -> not CSS
    with pytest.raises(ValueError, match="not a CSS pair"):
        css_logicals(Hx, Hz)


def test_css_logicals_rejects_column_mismatch():
    with pytest.raises(ValueError, match="columns"):
        css_logicals(np.ones((1, 4), np.uint8), np.ones((1, 5), np.uint8))


# ---------------------------------------------------------------------------
# Bivariate bicycle (flagship qLDPC)
# ---------------------------------------------------------------------------
def test_bb72_has_12_biorthogonal_logicals():
    cx, cz = codes.bivariate_bicycle_code(6, 6, BB72_A, BB72_B)
    assert cx.num_logical_qubits == 12
    assert cz.num_logical_qubits == 12
    Hx, Hz = cx.parity_check_matrix(), cz.parity_check_matrix()
    Lx = _dense(cx.logicals, cx.n_qubits)
    Lz = _dense(cz.logicals, cz.n_qubits)
    assert Lx.shape == (12, 72) and Lz.shape == (12, 72)
    # Logicals commute with every stabiliser of the opposite type
    assert not np.any((Hx @ Lz.T) % 2)
    assert not np.any((Hz @ Lx.T) % 2)
    # Biorthogonal pairing: Lx_i detects exactly Lz_j == delta_ij
    assert np.array_equal((Lx @ Lz.T) % 2, np.eye(12, dtype=np.uint8))
    # Not products of stabilisers (independence from the check row spaces)
    assert gf2_rank(np.vstack([Hz, Lz])) == gf2_rank(Hz) + 12
    assert gf2_rank(np.vstack([Hx, Lx])) == gf2_rank(Hx) + 12


def test_bb144_has_12_biorthogonal_logicals():
    cx, cz = codes.bivariate_bicycle_code(12, 6, BB72_A, BB72_B)
    assert cx.num_logical_qubits == 12
    assert cz.num_logical_qubits == 12
    Lx = _dense(cx.logicals, cx.n_qubits)
    Lz = _dense(cz.logicals, cz.n_qubits)
    assert Lx.shape == (12, 144) and Lz.shape == (12, 144)
    assert np.array_equal((Lx @ Lz.T) % 2, np.eye(12, dtype=np.uint8))


# ---------------------------------------------------------------------------
# Hypergraph product / bicycle
# ---------------------------------------------------------------------------
def test_hypergraph_product_matches_css_dimension_formula():
    # Hamming [7,4,3] seed -> HGP CSS code; k must equal n - rank(Hx) - rank(Hz)
    H = np.array(
        [
            [1, 0, 1, 0, 1, 0, 1],
            [0, 1, 1, 0, 0, 1, 1],
            [0, 0, 0, 1, 1, 1, 1],
        ],
        dtype=np.uint8,
    )
    cx, cz = codes.hypergraph_product(H)
    Hx, Hz = cx.parity_check_matrix(), cz.parity_check_matrix()
    assert not np.any((Hx @ Hz.T) % 2)  # valid CSS pair
    k = cx.n_qubits - gf2_rank(Hx) - gf2_rank(Hz)
    assert k > 0
    assert cx.num_logical_qubits == k == cz.num_logical_qubits
    Lx = _dense(cx.logicals, cx.n_qubits)
    Lz = _dense(cz.logicals, cz.n_qubits)
    assert not np.any((Hx @ Lz.T) % 2)
    assert not np.any((Hz @ Lx.T) % 2)
    assert np.array_equal((Lx @ Lz.T) % 2, np.eye(k, dtype=np.uint8))


def test_bicycle_code_logicals_match_dimension_formula():
    cx, cz = codes.bicycle_code(12, weight=4, seed=3)
    Hx, Hz = cx.parity_check_matrix(), cz.parity_check_matrix()
    assert not np.any((Hx @ Hz.T) % 2)
    k = cx.n_qubits - gf2_rank(Hx) - gf2_rank(Hz)
    assert cx.num_logical_qubits == k == cz.num_logical_qubits
    Lx = _dense(cx.logicals, cx.n_qubits)
    Lz = _dense(cz.logicals, cz.n_qubits)
    assert np.array_equal((Lx @ Lz.T) % 2, np.eye(k, dtype=np.uint8))


# ---------------------------------------------------------------------------
# Toric code (k = 2 through the plaquette-derived quotient)
# ---------------------------------------------------------------------------
def test_toric_code_exposes_two_logicals():
    code = codes.toric_code(3)
    assert code.num_logical_qubits == 2
    H = code.parity_check_matrix()
    Lx = code.logicals_matrix()
    Lz = np.asarray(code._meta["logical_z_ops"], dtype=np.uint8)
    assert Lx.shape == (2, 18) and Lz.shape == (2, 18)
    # Z logicals are non-contractible loops: in the star kernel...
    assert not np.any((H @ Lz.T) % 2)
    # ...and independent of the plaquette boundary space, paired with X duals
    assert np.array_equal((Lx @ Lz.T) % 2, np.eye(2, dtype=np.uint8))


def test_toric_logical_representative_is_nonzero_weight():
    code = codes.toric_code(4)
    for logical in code._meta["logical_z_ops"]:
        assert int(np.sum(logical)) >= code.distance  # non-trivial loop


# ---------------------------------------------------------------------------
# Honest negative + no-regression on pre-existing generators
# ---------------------------------------------------------------------------
def test_unrotated_surface_single_sector_has_no_css_logical():
    code = codes.unrotated_surface_code(3)
    assert code.logicals is None
    assert code.num_logical_qubits == 0


def test_preexisting_logicals_unchanged():
    assert codes.rotated_surface_code(5).logicals == [[0, 1, 2, 3, 4]]
    assert codes.repetition_code(5).logicals == [[0]]
    assert codes.ring_code(6).logicals == [[0]]
    assert codes.heavy_hex_code(3).logicals == [[0, 1, 2]]


# ---------------------------------------------------------------------------
# C6: uniform observables API across every built-in family
# ---------------------------------------------------------------------------
def test_all_builtin_families_expose_uniform_observables_api():
    fams = [
        codes.repetition_code(5),
        codes.ring_code(6),
        codes.rotated_surface_code(3),
        codes.unrotated_surface_code(3),
        codes.toric_code(3),
        codes.heavy_hex_code(3),
    ]
    for code in fams:
        L = code.logicals_matrix()
        if code.logicals is None:
            assert L is None
            assert code.num_logical_qubits == 0
        else:
            assert L is not None
            assert L.shape[1] == code.n_qubits
            assert L.shape[0] == code.num_logical_qubits == len(code.logicals)
    cx, _ = codes.bivariate_bicycle_code(6, 6, BB72_A, BB72_B)
    assert cx.logicals_matrix().shape == (12, 72)
    hx, _ = codes.hypergraph_product(np.array([[1, 1, 0], [0, 1, 1]], np.uint8))
    assert hx.logicals_matrix() is not None


# ---------------------------------------------------------------------------
# End-to-end: the logical-flip metric actually measures decode failures
# ---------------------------------------------------------------------------
def test_bb72_logical_flip_metric_end_to_end():
    from qector_decoder_v3.bposd import BpOsdDecoder

    cx, _cz = codes.bivariate_bicycle_code(6, 6, BB72_A, BB72_B)
    H = cx.parity_check_matrix()
    Lx = cx.logicals_matrix()
    rng = np.random.default_rng(11)
    dec = BpOsdDecoder(H, error_rate=0.02, max_iter=20, osd_order=0)
    shots, fails, unfaithful = 200, 0, 0
    for _ in range(shots):
        e = (rng.random(cx.n_qubits) < 0.005).astype(np.uint8)
        s = (H @ e) & 1
        c = np.asarray(dec.decode(s), dtype=np.uint8)
        r = (c ^ e).astype(np.uint8)
        if not np.array_equal((H @ c) & 1, s):
            unfaithful += 1
            continue
        if np.any((Lx @ r) & 1):
            fails += 1
    assert unfaithful == 0  # BP-OSD is syndrome-faithful on every shot
    # At p=0.005 on [[72,12,6]] logical failures must be rare; a broken
    # metric (e.g. duals not in ker(Hz)) would flag ~half the shots.
    assert fails / shots < 0.05
