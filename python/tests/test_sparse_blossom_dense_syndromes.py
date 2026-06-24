"""SparseBlossom stays syndrome-faithful under very dense syndromes.

At physical error rates of p = 0.40-0.45 a large fraction of checks fire, which
stresses region growing far harder than the usual sub-threshold regime.  We draw
high-density *reachable* syndromes (so a valid correction always exists) and
assert the core invariant ``(H @ corr) & 1 == s`` holds on every shot.
"""

import numpy as np
import pytest

import qector_decoder_v3 as qd
from qector_decoder_v3 import codes


@pytest.mark.parametrize(
    "code",
    [
        codes.rotated_surface_code(5),
        codes.rotated_surface_code(7),
        codes.toric_code(5),
    ],
    ids=lambda c: c.name,
)
@pytest.mark.parametrize("p", [0.40, 0.45])
def test_sparse_blossom_dense_faithful(code, p):
    H = code.parity_check_matrix()
    nq = code.n_qubits
    dec = qd.SparseBlossomDecoder(code.check_to_qubits, nq)

    rng = np.random.default_rng((abs(hash(code.name)) ^ int(p * 1000)) % (2**32))
    n_shots = 150
    dense_seen = 0
    for _ in range(n_shots):
        e = (rng.random(nq) < p).astype(np.uint8)
        s = ((H @ e) & 1).astype(np.uint8)
        c = np.asarray(dec.decode(s), np.uint8)
        assert c.shape == (nq,)
        assert np.array_equal((H @ c) & 1, s), f"{code.name} p={p}: H@c != s"
        if s.sum() >= code.n_checks // 4:
            dense_seen += 1
    # confirm we genuinely exercised dense syndromes, not sparse ones
    assert dense_seen > n_shots // 2, f"{code.name} p={p}: expected many dense syndromes, saw {dense_seen}"
