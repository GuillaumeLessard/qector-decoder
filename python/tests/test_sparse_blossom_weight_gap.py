"""SparseBlossom is near-optimal, not exact: bound its weight vs exact Blossom.

On small matching codes we compare, per syndrome, the Hamming weight of the
SparseBlossom (region-growing) correction against the *exact* MWPM weight from
``BlossomDecoder`` on the SAME syndrome.  Both must be syndrome-faithful.

Documented bound: SparseBlossom is region-growing and so is allowed to be at most
a little heavier than exact.  Empirically on these small codes it is essentially
always equal; we assert the conservative per-shot bound ``sparse_w <= exact_w + 2``
and that the fraction of shots with ``sparse_w == exact_w`` is high (>= 0.9).
SparseBlossom is never *lighter* than exact MWPM, so we also assert
``sparse_w >= exact_w`` (exact is the true minimum).
"""

import numpy as np
import pytest

import qector_decoder_v3 as qd
from qector_decoder_v3 import codes


@pytest.mark.parametrize(
    "code",
    [
        codes.repetition_code(5),
        codes.repetition_code(7),
        codes.repetition_code(9),
        codes.rotated_surface_code(3),
        codes.rotated_surface_code(5),
    ],
    ids=lambda c: c.name,
)
def test_sparse_blossom_weight_within_bound(code):
    H = code.parity_check_matrix()
    nq = code.n_qubits
    sb = qd.SparseBlossomDecoder(code.check_to_qubits, nq)
    bl = qd.BlossomDecoder(code.check_to_qubits, nq)

    rng = np.random.default_rng(abs(hash(code.name)) % (2**32))
    n_shots = 200
    n_equal = 0
    max_gap = 0
    for _ in range(n_shots):
        e = (rng.random(nq) < 0.12).astype(np.uint8)
        s = ((H @ e) & 1).astype(np.uint8)

        cs = np.asarray(sb.decode(s), np.uint8)
        cb = np.asarray(bl.decode(s), np.uint8)

        # both faithful
        assert np.array_equal((H @ cs) & 1, s), f"{code.name}: sparse H@c != s"
        assert np.array_equal((H @ cb) & 1, s), f"{code.name}: exact H@c != s"

        ws = int(cs.sum())
        wb = int(cb.sum())
        gap = ws - wb
        max_gap = max(max_gap, gap)

        # exact is the true minimum -> sparse never lighter
        assert ws >= wb, f"{code.name}: sparse({ws}) < exact({wb})"
        # documented near-optimality bound
        assert gap <= 2, f"{code.name}: weight gap {gap} > 2 (s={s})"

        if ws == wb:
            n_equal += 1

    frac_equal = n_equal / n_shots
    assert frac_equal >= 0.9, f"{code.name}: only {frac_equal:.2%} of shots had sparse_w == exact_w (max_gap={max_gap})"
