"""BP-OSD on a hypergraph-product code built from a small seed matrix.

The hypergraph product of a single classical seed H1 yields a CSS quantum code.
This file checks the CSS commutation relation and that ``BpOsdDecoder`` is
syndrome-faithful over many random errors.
"""

import numpy as np

from qector_decoder_v3 import codes
from qector_decoder_v3.bposd import BpOsdDecoder


def _hgp():
    # 3x3 cyclic-ish seed (each row/col weight 2) -> small, well-conditioned HGP.
    H1 = np.array([[1, 1, 0], [0, 1, 1], [1, 0, 1]], dtype=np.uint8)
    return codes.hypergraph_product(H1)


def test_hypergraph_product_is_valid_css():
    cx, cz = _hgp()
    Hx = cx.parity_check_matrix().astype(np.uint8)
    Hz = cz.parity_check_matrix().astype(np.uint8)
    assert Hx.shape[1] == Hz.shape[1]
    assert Hx.shape[1] > 0
    assert np.array_equal((Hx @ Hz.T) % 2, np.zeros((Hx.shape[0], Hz.shape[0]), np.uint8))


def test_bposd_hgp_faithful_per_shot():
    cx, _ = _hgp()
    H = cx.parity_check_matrix().astype(np.uint8)
    nq = H.shape[1]
    dec = BpOsdDecoder(H, error_rate=0.05, max_iter=30, osd_order=0)
    rng = np.random.default_rng(21)
    for _ in range(120):
        e = (rng.random(nq) < 0.05).astype(np.uint8)
        s = ((H @ e) & 1).astype(np.uint8)
        c = np.asarray(dec.decode(s)).astype(np.uint8)
        assert c.shape == (nq,)
        assert np.array_equal((H @ c) & 1, s)


def test_bposd_hgp_faithful_batch():
    cx, _ = _hgp()
    H = cx.parity_check_matrix().astype(np.uint8)
    nq = H.shape[1]
    dec = BpOsdDecoder(H, error_rate=0.05, max_iter=30, osd_order=0)
    rng = np.random.default_rng(22)
    shots = 120
    errs = (rng.random((shots, nq)) < 0.05).astype(np.uint8)
    syns = ((errs @ H.T) & 1).astype(np.uint8)
    out = np.asarray(dec.batch_decode(syns)).astype(np.uint8)
    assert out.shape == (shots, nq)
    for i in range(shots):
        assert np.array_equal((H @ out[i]) & 1, syns[i])
