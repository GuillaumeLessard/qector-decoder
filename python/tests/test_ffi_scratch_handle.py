"""FFI scratch-handle test: FastUnionFind thread-local scratch arena.

The native FastUnionFindDecoder uses a thread-local scratch buffer to avoid
re-allocation.  These tests exercise repeated decode calls to confirm the
scratch handle does not corrupt or crash.
"""

import numpy as np
import pytest

import qector_decoder_v3 as qd


@pytest.mark.skipif(not hasattr(qd, "FastUnionFindDecoder"), reason="FastUnionFindDecoder not available")
def test_fastuf_thread_local_scratch():
    c2q = [[0, 1], [1, 2], [2, 3], [3, 0]]
    dec = qd.FastUnionFindDecoder(c2q)
    rng = np.random.default_rng(1)
    for _ in range(50):
        s = rng.integers(0, 2, size=dec.n_checks, dtype=np.uint8)
        c = np.asarray(dec.decode(s), dtype=np.uint8)
        assert c.shape == (dec.n_qubits,)


@pytest.mark.skipif(not hasattr(qd, "FastUnionFindDecoder"), reason="FastUnionFindDecoder not available")
def test_fastuf_arena_decode():
    c2q = [[0, 1], [1, 2], [2, 3], [3, 0]]
    uf = qd.UnionFindDecoder(c2q)
    fuf = qd.FastUnionFindDecoder(c2q)
    rng = np.random.default_rng(5)
    for _ in range(20):
        s = rng.integers(0, 2, size=uf.n_checks, dtype=np.uint8)
        c_uf = np.asarray(uf.decode(s), dtype=np.uint8)
        c_fuf = np.asarray(fuf.decode(s), dtype=np.uint8)
        np.testing.assert_array_equal(c_fuf, c_uf)
