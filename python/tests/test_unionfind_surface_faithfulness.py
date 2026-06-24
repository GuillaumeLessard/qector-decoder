"""Union-Find decoders are syndrome-faithful on surface codes.

Both ``UnionFindDecoder`` and the SIMD ``FastUnionFindDecoder`` are exercised on
rotated (d=5, d=7) and unrotated (d=5) surface codes over many random reachable
syndromes (p~0.1), asserting the core invariant ``(H @ corr) & 1 == s``.
"""

import numpy as np
import pytest

import qector_decoder_v3 as qd
from qector_decoder_v3 import codes

UF_DECODERS = {
    "UnionFind": qd.UnionFindDecoder,
    "FastUnionFind": qd.FastUnionFindDecoder,
}


@pytest.mark.parametrize(
    "code",
    [
        codes.rotated_surface_code(5),
        codes.rotated_surface_code(7),
        codes.unrotated_surface_code(5),
    ],
    ids=lambda c: c.name,
)
@pytest.mark.parametrize("dec_name", list(UF_DECODERS))
def test_union_find_surface_faithful(code, dec_name):
    H = code.parity_check_matrix()
    nq = code.n_qubits
    dec = UF_DECODERS[dec_name](code.check_to_qubits, nq)

    rng = np.random.default_rng((abs(hash(code.name)) ^ hash(dec_name)) % (2**32))
    n_shots = 200
    nonzero = 0
    for _ in range(n_shots):
        e = (rng.random(nq) < 0.1).astype(np.uint8)
        s = ((H @ e) & 1).astype(np.uint8)
        c = np.asarray(dec.decode(s), np.uint8)
        assert c.shape == (nq,)
        assert np.array_equal((H @ c) & 1, s), f"{dec_name} on {code.name}: H@c != s"
        if s.any():
            nonzero += 1
    assert nonzero > n_shots // 4
