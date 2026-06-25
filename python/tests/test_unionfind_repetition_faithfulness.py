"""Union-Find decoders are syndrome-faithful on repetition codes.

Both ``UnionFindDecoder`` and ``FastUnionFindDecoder`` are exercised on
repetition codes of distance 7, 11 and 21 over many random reachable syndromes,
asserting the core invariant ``(H @ corr) & 1 == s``.
"""

import numpy as np
import pytest

import qector_decoder_v3 as qd
from qector_decoder_v3 import codes

UF_DECODERS = {
    "UnionFind": qd.UnionFindDecoder,
    "FastUnionFind": qd.FastUnionFindDecoder,
}


@pytest.mark.parametrize("d", [7, 11, 21])
@pytest.mark.parametrize("dec_name", list(UF_DECODERS))
def test_union_find_repetition_faithful(d, dec_name):
    code = codes.repetition_code(d)
    H = code.parity_check_matrix()
    nq = code.n_qubits
    dec = UF_DECODERS[dec_name](code.check_to_qubits, nq)

    rng = np.random.default_rng((d * 1000) ^ hash(dec_name) % (2**32))
    n_shots = 200
    nonzero = 0
    for _ in range(n_shots):
        e = (rng.random(nq) < 0.1).astype(np.uint8)
        s = ((H @ e) & 1).astype(np.uint8)
        c = np.asarray(dec.decode(s), np.uint8)
        assert c.shape == (nq,)
        assert np.array_equal((H @ c) & 1, s), f"{dec_name} on rep({d}): H@c != s"
        if s.any():
            nonzero += 1
    assert nonzero > n_shots // 4


@pytest.mark.parametrize("d", [7, 11, 21])
@pytest.mark.parametrize("dec_name", list(UF_DECODERS))
def test_union_find_repetition_every_single_defect(d, dec_name):
    """Every single-check-lit syndrome must be corrected faithfully."""
    code = codes.repetition_code(d)
    H = code.parity_check_matrix()
    nq = code.n_qubits
    nc = code.n_checks
    dec = UF_DECODERS[dec_name](code.check_to_qubits, nq)
    for ci in range(nc):
        s = np.zeros(nc, np.uint8)
        s[ci] = 1
        c = np.asarray(dec.decode(s), np.uint8)
        assert np.array_equal((H @ c) & 1, s), (
            f"{dec_name} on rep({d}): defect {ci} not faithful"
        )
