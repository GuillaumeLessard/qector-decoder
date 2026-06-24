"""Union-Find decoders are syndrome-faithful on toric codes.

The toric code is a periodic (boundary-free) matching graph.  Both
``UnionFindDecoder`` and ``FastUnionFindDecoder`` are exercised on L = 4, 5, 6
over many random reachable syndromes, asserting ``(H @ corr) & 1 == s``.
"""
import numpy as np
import pytest

import qector_decoder_v3 as qd
from qector_decoder_v3 import codes

UF_DECODERS = {
    "UnionFind": qd.UnionFindDecoder,
    "FastUnionFind": qd.FastUnionFindDecoder,
}


@pytest.mark.parametrize("L", [4, 5, 6])
@pytest.mark.parametrize("dec_name", list(UF_DECODERS))
def test_union_find_toric_faithful(L, dec_name):
    code = codes.toric_code(L)
    H = code.parity_check_matrix()
    nq = code.n_qubits
    dec = UF_DECODERS[dec_name](code.check_to_qubits, nq)

    rng = np.random.default_rng((L * 7919) ^ (hash(dec_name) % (2**32)))
    n_shots = 200
    nonzero = 0
    for _ in range(n_shots):
        e = (rng.random(nq) < 0.1).astype(np.uint8)
        s = ((H @ e) & 1).astype(np.uint8)
        c = np.asarray(dec.decode(s), np.uint8)
        assert c.shape == (nq,)
        assert np.array_equal((H @ c) & 1, s), (
            f"{dec_name} on toric({L}): H@c != s"
        )
        if s.any():
            nonzero += 1
    assert nonzero > n_shots // 4
