"""BP-OSD on a (random) bicycle code from the bicycle_code family.

``codes.bicycle_code`` builds a CSS code from a random circulant block (seeded
for reproducibility).  Checks CSS commutation and BP-OSD syndrome-faithfulness.
"""

import numpy as np
import pytest

from qector_decoder_v3 import codes
from qector_decoder_v3.bposd import BpOsdDecoder


def _bicycle():
    return codes.bicycle_code(12, weight=6, seed=1)


def test_bicycle_is_valid_css():
    bx, bz = _bicycle()
    Hx = bx.parity_check_matrix().astype(np.uint8)
    Hz = bz.parity_check_matrix().astype(np.uint8)
    assert Hx.shape[1] == Hz.shape[1]
    assert Hx.shape[1] > 0
    assert np.array_equal((Hx @ Hz.T) % 2, np.zeros((Hx.shape[0], Hz.shape[0]), np.uint8))


def test_bposd_bicycle_faithful_per_shot():
    bx, _ = _bicycle()
    H = bx.parity_check_matrix().astype(np.uint8)
    nq = H.shape[1]
    dec = BpOsdDecoder(H, error_rate=0.05, max_iter=30, osd_order=0)
    rng = np.random.default_rng(31)
    for _ in range(120):
        e = (rng.random(nq) < 0.05).astype(np.uint8)
        s = ((H @ e) & 1).astype(np.uint8)
        c = np.asarray(dec.decode(s)).astype(np.uint8)
        assert c.shape == (nq,)
        assert np.array_equal((H @ c) & 1, s)


def test_bposd_bicycle_faithful_batch():
    bx, _ = _bicycle()
    H = bx.parity_check_matrix().astype(np.uint8)
    nq = H.shape[1]
    dec = BpOsdDecoder(H, error_rate=0.05, max_iter=30, osd_order=0)
    rng = np.random.default_rng(32)
    shots = 120
    errs = (rng.random((shots, nq)) < 0.05).astype(np.uint8)
    syns = ((errs @ H.T) & 1).astype(np.uint8)
    out = np.asarray(dec.batch_decode(syns)).astype(np.uint8)
    assert out.shape == (shots, nq)
    for i in range(shots):
        assert np.array_equal((H @ out[i]) & 1, syns[i])
