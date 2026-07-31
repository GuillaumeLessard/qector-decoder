"""Tests for decoder-divergent corrections on designed syndromes.

Different decoders (Blossom vs UF, BP-OSD vs Blossom) can give different
valid corrections for the same syndrome.  This file tests that both are
syndrome-faithful (H·c == s) even when they disagree.
"""

import numpy as np
import pytest
import qector_decoder_v3 as qd


def _ring_code(n):
    return [[i, (i + 1) % n] for i in range(n)], n


def _H(c2q, nq):
    H = np.zeros((len(c2q), nq), dtype=np.uint8)
    for ci, qs in enumerate(c2q):
        for q in qs:
            H[ci, q] ^= 1
    return H


@pytest.mark.skipif(
    not (hasattr(qd, "BlossomDecoder") and hasattr(qd, "FastUnionFindDecoder")),
    reason="BlossomDecoder or FastUnionFindDecoder not available",
)
def test_divergent_blossom_vs_uf():
    c2q, nq = _ring_code(8)
    H = _H(c2q, nq)
    bdec = qd.BlossomDecoder(c2q, nq)
    udec = qd.FastUnionFindDecoder(c2q, nq)
    rng = np.random.default_rng(9)
    for _ in range(30):
        e = (rng.random(nq) < 0.1).astype(np.uint8)
        s = (H @ e) & 1
        cb = np.asarray(bdec.decode(s.astype(np.uint8)), dtype=np.uint8).reshape(-1)
        cu = np.asarray(udec.decode(s.astype(np.uint8)), dtype=np.uint8).reshape(-1)
        assert np.array_equal((H @ cb) & 1, s)
        assert np.array_equal((H @ cu) & 1, s)


@pytest.mark.skipif(
    not (hasattr(qd, "BlossomDecoder") and hasattr(qd, "BPOSDDecoder")),
    reason="BlossomDecoder or BPOSDDecoder not available",
)
def test_divergent_bposd_vs_blossom():
    c2q, nq = _ring_code(6)
    H = _H(c2q, nq)
    bdec = qd.BlossomDecoder(c2q, nq)
    pdec = qd.BPOSDDecoder(c2q, nq, 0.08)
    rng = np.random.default_rng(11)
    for _ in range(20):
        e = (rng.random(nq) < 0.08).astype(np.uint8)
        s = (H @ e) & 1
        cb = np.asarray(bdec.decode(s.astype(np.uint8)), dtype=np.uint8).reshape(-1)
        cp = np.asarray(pdec.decode(s.astype(np.uint8)), dtype=np.uint8).reshape(-1)
        assert np.array_equal((H @ cb) & 1, s)
        assert np.array_equal((H @ cp) & 1, s)
