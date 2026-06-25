"""Tests for qector_decoder_v3.codes — code-family helpers.

Every matching-graph generator is validated for (a) graphlikeness and (b)
syndrome-faithfulness against the live decoders.  The hypergraph-product helper
is checked for the CSS commutation condition ``Hx Hz^T = 0 (mod 2)``.
"""

import numpy as np
import pytest

import qector_decoder_v3 as qd
from qector_decoder_v3 import codes


MATCHING_CODES = [
    codes.repetition_code(7),
    codes.repetition_code(15),
    codes.ring_code(12),
    codes.rotated_surface_code(5),
    codes.rotated_surface_code(7),
    codes.unrotated_surface_code(4),
    codes.unrotated_surface_code(5),
    codes.toric_code(4),
    codes.toric_code(5),
    codes.heavy_hex_code(5),
]


@pytest.mark.parametrize("code", MATCHING_CODES, ids=[c.name for c in MATCHING_CODES])
def test_is_matching_graph(code):
    assert code.is_matching_graph(), (code.name, code.max_qubit_degree())
    assert code.max_qubit_degree() <= 2


@pytest.mark.parametrize("code", MATCHING_CODES, ids=[c.name for c in MATCHING_CODES])
def test_parity_check_matrix_shape(code):
    H = code.parity_check_matrix()
    assert H.shape == (code.n_checks, code.n_qubits)
    assert H.dtype == np.uint8
    # check_to_qubits and H agree
    for ci, qs in enumerate(code.check_to_qubits):
        assert sorted(np.nonzero(H[ci])[0].tolist()) == sorted(qs)


@pytest.mark.parametrize(
    "decoder", ["UnionFind", "FastUnionFind", "Blossom", "SparseBlossom"]
)
@pytest.mark.parametrize("code", MATCHING_CODES, ids=[c.name for c in MATCHING_CODES])
def test_codes_are_syndrome_faithful(code, decoder):
    H = code.parity_check_matrix()
    dec = {
        "UnionFind": qd.UnionFindDecoder,
        "FastUnionFind": qd.FastUnionFindDecoder,
        "Blossom": qd.BlossomDecoder,
        "SparseBlossom": qd.SparseBlossomDecoder,
    }[decoder](code.check_to_qubits, code.n_qubits)
    rng = np.random.default_rng(2024)
    for _ in range(150):
        e = (rng.random(code.n_qubits) < 0.08).astype(np.uint8)
        s = (H @ e) & 1
        corr = np.asarray(dec.decode(s.astype(np.uint8))).astype(np.uint8)
        assert corr.shape == (code.n_qubits,)
        assert np.array_equal((H @ corr) & 1, s), f"{decoder} on {code.name}"


def test_from_parity_check_matrix_roundtrip():
    H = np.array([[1, 1, 0, 0], [0, 1, 1, 0], [0, 0, 1, 1]], dtype=np.uint8)
    code = codes.from_parity_check_matrix(H, name="manual")
    assert code.n_qubits == 4
    assert code.n_checks == 3
    assert np.array_equal(code.parity_check_matrix(), H)


def test_from_parity_check_matrix_accepts_scipy_sparse():
    sp = pytest.importorskip("scipy.sparse")
    H = np.array([[1, 1, 0], [0, 1, 1]], dtype=np.uint8)
    code = codes.from_parity_check_matrix(sp.csr_matrix(H))
    assert np.array_equal(code.parity_check_matrix(), H)


def test_hypergraph_product_is_valid_css():
    # Seed: a small parity check (the [3,1] repetition check matrix).
    H1 = np.array([[1, 1, 0], [0, 1, 1]], dtype=np.uint8)
    code_x, code_z = codes.hypergraph_product(H1)
    Hx = code_x.parity_check_matrix()
    Hz = code_z.parity_check_matrix()
    # Same number of physical qubits in both sectors.
    assert Hx.shape[1] == Hz.shape[1]
    # CSS commutation condition.
    assert np.array_equal(
        (Hx @ Hz.T) % 2, np.zeros((Hx.shape[0], Hz.shape[0]), np.uint8)
    )


def test_logicals_matrix_and_helpers():
    code = codes.repetition_code(9)
    L = code.logicals_matrix()
    assert L is not None and L.shape == (1, 9)
    rng = np.random.default_rng(0)
    e = code.random_error(0.2, rng)
    assert e.shape == (9,)
    s = code.syndrome(e)
    assert np.array_equal(s, (code.parity_check_matrix() @ e) & 1)


def test_repetition_logical_observable_is_valid():
    """The repetition observable {0} must distinguish trivial vs logical residue."""
    code = codes.repetition_code(11)
    H = code.parity_check_matrix()
    L = code.logicals_matrix()
    dec = qd.BlossomDecoder(code.check_to_qubits, code.n_qubits)
    rng = np.random.default_rng(5)
    # At very low error rate the decoder should rarely flip the logical.
    failures = 0
    n = 400
    for _ in range(n):
        e = (rng.random(code.n_qubits) < 0.03).astype(np.uint8)
        s = (H @ e) & 1
        c = np.asarray(dec.decode(s.astype(np.uint8))).astype(np.uint8)
        residue = c ^ e
        if bool(((L @ residue) & 1).any()):
            failures += 1
    # Logical failure rate must be well below chance (0.5) for a working observable.
    assert failures / n < 0.2


def test_invalid_arguments():
    with pytest.raises(ValueError):
        codes.repetition_code(1)
    with pytest.raises(ValueError):
        codes.toric_code(1)
    with pytest.raises(ValueError):
        codes.heavy_hex_code(4)  # must be odd
    with pytest.raises(ValueError):
        codes.from_parity_check_matrix(np.zeros((3,), dtype=np.uint8))
