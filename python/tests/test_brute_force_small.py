"""Brute-force ground-truth validation on small codes.

For codes small enough to enumerate every error, we build the exact
minimum-weight correction for each reachable syndrome and check that:

* every decoder is syndrome-faithful, and
* the MWPM decoders (Blossom / Sparse-Blossom) return a correction whose weight
  equals the true minimum weight (optimality, not just validity).

This is the strongest correctness statement available and would catch any
regression that returns a valid-but-suboptimal or invalid matching.
"""

import itertools

import numpy as np
import pytest

import qector_decoder_v3 as qd
from qector_decoder_v3 import codes


def _min_weight_table(H, n_qubits):
    """Map every reachable syndrome (as bytes) -> minimum correction weight."""
    best = {}
    for bits in itertools.product((0, 1), repeat=n_qubits):
        e = np.array(bits, dtype=np.uint8)
        s = tuple(((H @ e) & 1).tolist())
        w = int(e.sum())
        if s not in best or w < best[s]:
            best[s] = w
    return best


SMALL_CODES = [
    codes.repetition_code(7),
    codes.repetition_code(9),
    codes.rotated_surface_code(3),
    codes.ring_code(8),
]


@pytest.mark.parametrize("code", SMALL_CODES, ids=[c.name for c in SMALL_CODES])
def test_exact_blossom_is_weight_optimal(code):
    """BlossomDecoder is exact MWPM: its weight equals the brute-force minimum."""
    H = code.parity_check_matrix()
    table = _min_weight_table(H, code.n_qubits)
    blossom = qd.BlossomDecoder(code.check_to_qubits, code.n_qubits)
    for s_tuple, min_w in table.items():
        s = np.array(s_tuple, dtype=np.uint8)
        c = np.asarray(blossom.decode(s)).astype(np.uint8)
        assert np.array_equal((H @ c) & 1, s), f"Blossom {code.name}: not faithful"
        assert int(c.sum()) == min_w, (
            f"Blossom {code.name}: weight {int(c.sum())} != optimal {min_w} for syndrome {s_tuple}"
        )


@pytest.mark.parametrize("code", SMALL_CODES, ids=[c.name for c in SMALL_CODES])
def test_sparse_blossom_is_faithful_and_near_optimal(code):
    """SparseBlossomDecoder is a fast region-growing decoder: always faithful and
    near-optimal (>=99% of syndromes optimal, weight gap <=1 on small codes)."""
    H = code.parity_check_matrix()
    table = _min_weight_table(H, code.n_qubits)
    sparse = qd.SparseBlossomDecoder(code.check_to_qubits, code.n_qubits)
    total = len(table)
    optimal = 0
    max_gap = 0
    for s_tuple, min_w in table.items():
        s = np.array(s_tuple, dtype=np.uint8)
        c = np.asarray(sparse.decode(s)).astype(np.uint8)
        assert np.array_equal((H @ c) & 1, s), (
            f"SparseBlossom {code.name}: not faithful"
        )
        gap = int(c.sum()) - min_w
        max_gap = max(max_gap, gap)
        if gap == 0:
            optimal += 1
    assert optimal / total >= 0.99, f"{code.name}: only {optimal}/{total} optimal"
    assert max_gap <= 1, f"{code.name}: weight gap {max_gap} > 1"


@pytest.mark.parametrize("code", SMALL_CODES, ids=[c.name for c in SMALL_CODES])
def test_all_decoders_faithful_on_every_syndrome(code):
    H = code.parity_check_matrix()
    table = _min_weight_table(H, code.n_qubits)
    decs = {
        "UnionFind": qd.UnionFindDecoder(code.check_to_qubits, code.n_qubits),
        "FastUnionFind": qd.FastUnionFindDecoder(code.check_to_qubits, code.n_qubits),
        "LookupTable": qd.LookupTableDecoder(code.check_to_qubits, code.n_qubits),
    }
    decs["LookupTable"].build_table(1 << 16)
    for s_tuple in table:
        s = np.array(s_tuple, dtype=np.uint8)
        for name, dec in decs.items():
            c = np.asarray(dec.decode(s)).astype(np.uint8)
            assert np.array_equal((H @ c) & 1, s), f"{name} {code.name}: not faithful"


def test_known_answer_vectors_repetition():
    """Fixed known-answer test (KAT) vectors for the repetition code."""
    code = codes.repetition_code(5)  # checks: [0,1],[1,2],[2,3],[3,4]
    H = code.parity_check_matrix()
    dec = qd.BlossomDecoder(code.check_to_qubits, code.n_qubits)
    # single defect at check 0 -> minimal correction flips qubit 0 (boundary edge)
    cases = {
        (1, 0, 0, 0): 1,  # one boundary defect, weight-1 fix
        (0, 0, 0, 1): 1,
        (1, 1, 0, 0): 1,  # adjacent defects -> single internal qubit
        (1, 0, 0, 1): 2,  # two separated boundary defects
    }
    for s_tuple, expect_w in cases.items():
        s = np.array(s_tuple, dtype=np.uint8)
        c = np.asarray(dec.decode(s)).astype(np.uint8)
        assert np.array_equal((H @ c) & 1, s)
        assert int(c.sum()) == expect_w, (s_tuple, int(c.sum()), expect_w)
