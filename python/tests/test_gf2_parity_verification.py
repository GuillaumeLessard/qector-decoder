"""GF(2) parity verification: H·c == s for every decoder.

Every decoder must return corrections satisfying the syndrome under GF(2)
parity-check (the fundamental correctness property).
"""

import numpy as np
import pytest

import qector_decoder_v3 as qd


def _surface5_code():
    c2q = []
    d = 5
    for r in range(d - 1):
        for c in range(d - 1):
            if (r + c) % 2 == 0:
                c2q.append([r * d + c, r * d + c + 1, (r + 1) * d + c, (r + 1) * d + c + 1])
    H = np.zeros((len(c2q), d * d), dtype=np.uint8)
    for ci, qs in enumerate(c2q):
        for q in qs:
            H[ci, q] ^= 1
    return c2q, d * d, H


def _reachable_syndromes(H, nq, n_shots, p, seed):
    rng = np.random.default_rng(seed)
    errors = (rng.random((n_shots, nq)) < p).astype(np.uint8)
    return errors, (errors @ H.T) & 1


@pytest.mark.skipif(not hasattr(qd, "BlossomDecoder"), reason="BlossomDecoder not available")
def test_blossom_correction_parity():
    c2q, nq, H = _surface5_code()
    dec = qd.BlossomDecoder(c2q, nq)
    _, syns = _reachable_syndromes(H, nq, 30, 0.08, 1)
    for s in syns:
        c = np.asarray(dec.decode(s.astype(np.uint8)), dtype=np.uint8).reshape(-1)
        assert np.array_equal((H @ c) & 1, s)


@pytest.mark.skipif(not hasattr(qd, "FastUnionFindDecoder"), reason="FastUnionFindDecoder not available")
def test_fastuf_correction_parity():
    c2q, nq, H = _surface5_code()
    dec = qd.FastUnionFindDecoder(c2q, nq)
    _, syns = _reachable_syndromes(H, nq, 30, 0.08, 2)
    for s in syns:
        c = np.asarray(dec.decode(s.astype(np.uint8)), dtype=np.uint8).reshape(-1)
        assert np.array_equal((H @ c) & 1, s)


@pytest.mark.skipif(not hasattr(qd, "BPOSDDecoder"), reason="BPOSDDecoder not available")
def test_bposd_correction_parity():
    c2q, nq, H = _surface5_code()
    dec = qd.BPOSDDecoder(c2q, nq, 0.08)
    _, syns = _reachable_syndromes(H, nq, 20, 0.08, 3)
    for s in syns:
        c = np.asarray(dec.decode(s.astype(np.uint8)), dtype=np.uint8).reshape(-1)
        assert np.array_equal((H @ c) & 1, s)


@pytest.mark.skipif(not hasattr(qd, "SparseBlossomDecoder"), reason="SparseBlossomDecoder not available")
def test_sparse_blossom_correction_parity():
    c2q, nq, H = _surface5_code()
    dec = qd.SparseBlossomDecoder(c2q, nq)
    _, syns = _reachable_syndromes(H, nq, 30, 0.08, 4)
    for s in syns:
        c = np.asarray(dec.decode(s.astype(np.uint8)), dtype=np.uint8).reshape(-1)
        assert np.array_equal((H @ c) & 1, s)
