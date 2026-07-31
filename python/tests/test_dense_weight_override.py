"""Tests for SparseBlossomDecoder weight-override (decode_with_weights)."""

import numpy as np
import pytest
import qector_decoder_v3 as qd


@pytest.fixture
def surface5_code():
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


@pytest.mark.skipif(not hasattr(qd, "SparseBlossomDecoder"), reason="SparseBlossomDecoder not available")
def test_sparse_blossom_decode_with_weights(surface5_code):
    c2q, nq, H = surface5_code
    dec = qd.SparseBlossomDecoder(c2q, nq)
    rng = np.random.default_rng(3)
    e = (rng.random(nq) < 0.08).astype(np.uint8)
    s = (H @ e) & 1
    weights = [(i, 1.0) for i in range(nq)]
    c = np.asarray(dec.decode_with_weights(s.astype(np.uint8), weights), dtype=np.uint8).reshape(-1)
    assert c.shape == (nq,)


@pytest.mark.skipif(not hasattr(qd, "SparseBlossomDecoder"), reason="SparseBlossomDecoder not available")
def test_sparse_blossom_weight_override_does_not_break_faithfulness(surface5_code):
    c2q, nq, H = surface5_code
    dec = qd.SparseBlossomDecoder(c2q, nq)
    rng = np.random.default_rng(7)
    for _ in range(20):
        e = (rng.random(nq) < 0.08).astype(np.uint8)
        s = (H @ e) & 1
        weights = [(i, float(rng.random())) for i in range(nq)]
        c = np.asarray(dec.decode_with_weights(s.astype(np.uint8), weights), dtype=np.uint8).reshape(-1)
        assert np.array_equal((H @ c) & 1, s)
