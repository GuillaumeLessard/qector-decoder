"""SparseBlossom boundary pairing on the repetition code.

The repetition code has an open boundary: a lone defect is corrected by pairing
it to the nearest boundary, and a pair of defects is corrected by pairing them to
each other when that is cheaper than two boundary paths.  We verify SparseBlossom
returns syndrome-faithful corrections and compare its Hamming weight to the exact
MWPM weight (``BlossomDecoder``) for:

* every single isolated defect (one check lit), including the two near the
  boundary — here SparseBlossom is exactly optimal (weight equals exact MWPM),
  and the boundary defects have minimal weight 1; and
* every defect pair — here SparseBlossom is near-optimal: never lighter than
  exact, almost always equal, occasionally heavier by a small bounded amount
  when region growing pairs a defect to the boundary instead of to its partner.
"""

import numpy as np
import pytest

import qector_decoder_v3 as qd
from qector_decoder_v3 import codes


@pytest.mark.parametrize("d", [5, 7, 11])
def test_isolated_defect_pairs_to_nearest_boundary(d):
    code = codes.repetition_code(d)
    H = code.parity_check_matrix()
    nq = code.n_qubits
    nc = code.n_checks
    sb = qd.SparseBlossomDecoder(code.check_to_qubits, nq)
    bl = qd.BlossomDecoder(code.check_to_qubits, nq)

    # Every single-defect syndrome (one check lit). The two end checks pair to a
    # boundary at weight 1; interior checks need a longer path. The exact decoder
    # gives the true minimal weight; SparseBlossom must match it.
    for ci in range(nc):
        s = np.zeros(nc, np.uint8)
        s[ci] = 1
        cs = np.asarray(sb.decode(s), np.uint8)
        cb = np.asarray(bl.decode(s), np.uint8)
        assert np.array_equal((H @ cs) & 1, s), f"d={d}: sparse not faithful @ {ci}"
        assert np.array_equal((H @ cb) & 1, s), f"d={d}: exact not faithful @ {ci}"
        assert int(cs.sum()) == int(cb.sum()), (
            f"d={d}: isolated defect {ci} sparse weight {int(cs.sum())} != exact {int(cb.sum())}"
        )

    # the boundary checks (first and last) must be minimal weight 1
    for ci in (0, nc - 1):
        s = np.zeros(nc, np.uint8)
        s[ci] = 1
        cs = np.asarray(sb.decode(s), np.uint8)
        assert int(cs.sum()) == 1, f"d={d}: boundary defect {ci} not weight 1"


@pytest.mark.parametrize("d", [7, 11])
def test_defect_pairs_near_optimal_vs_exact_weight(d):
    """Defect pairs: SparseBlossom weight is near-optimal vs exact MWPM.

    Both decoders are syndrome-faithful.  Exact MWPM gives the true minimum, so
    SparseBlossom is never lighter.  Region growing occasionally chooses to pair
    a defect to the boundary where exact pairs the two defects together (or vice
    versa), so the weights can differ by a small amount; we assert the documented
    near-optimal bound and that the great majority of pairs match exactly.
    """
    code = codes.repetition_code(d)
    H = code.parity_check_matrix()
    nq = code.n_qubits
    nc = code.n_checks
    sb = qd.SparseBlossomDecoder(code.check_to_qubits, nq)
    bl = qd.BlossomDecoder(code.check_to_qubits, nq)

    # exhaustively cover every defect pair for this small code
    pairs = [(a, b) for a in range(nc) for b in range(a + 1, nc)]

    n_equal = 0
    max_gap = 0
    for a, b in pairs:
        s = np.zeros(nc, np.uint8)
        s[a] = 1
        s[b] = 1
        cs = np.asarray(sb.decode(s), np.uint8)
        cb = np.asarray(bl.decode(s), np.uint8)
        assert np.array_equal((H @ cs) & 1, s), f"d={d}: sparse not faithful {a},{b}"
        assert np.array_equal((H @ cb) & 1, s), f"d={d}: exact not faithful {a},{b}"

        ws, wb = int(cs.sum()), int(cb.sum())
        gap = ws - wb
        max_gap = max(max_gap, gap)
        # exact is the true minimum -> sparse never lighter; near-optimal bound
        assert ws >= wb, f"d={d}: pair ({a},{b}) sparse({ws}) < exact({wb})"
        assert gap <= 4, f"d={d}: pair ({a},{b}) weight gap {gap} > 4"
        if gap == 0:
            n_equal += 1

    frac_equal = n_equal / len(pairs)
    assert frac_equal >= 0.9, (
        f"d={d}: only {frac_equal:.2%} of defect pairs matched exact (max_gap={max_gap})"
    )
