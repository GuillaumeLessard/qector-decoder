"""SparseBlossom syndrome-faithfulness across the real QEC matching codes.

Pure-numpy / synthetic (no stim): for each code we draw many random reachable
syndromes from random data errors (p~0.1) and assert the core invariant
``(H @ corr) & 1 == s`` for the region-growing SparseBlossom decoder.
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
        codes.unrotated_surface_code(5),
        codes.toric_code(5),
        codes.repetition_code(11),
    ],
    ids=lambda c: c.name,
)
def test_sparse_blossom_syndrome_faithful(code):
    H = code.parity_check_matrix()
    nq = code.n_qubits
    dec = qd.SparseBlossomDecoder(code.check_to_qubits, nq)

    rng = np.random.default_rng(abs(hash(code.name)) % (2**32))
    n_shots = 200
    nonzero = 0
    for _ in range(n_shots):
        e = (rng.random(nq) < 0.1).astype(np.uint8)
        s = ((H @ e) & 1).astype(np.uint8)
        c = np.asarray(dec.decode(s), np.uint8)
        assert c.shape == (nq,)
        assert np.array_equal((H @ c) & 1, s), f"{code.name}: H@c != s"
        if s.any():
            nonzero += 1
    # the syndromes are genuinely non-trivial, not all zeros
    assert nonzero > n_shots // 4
