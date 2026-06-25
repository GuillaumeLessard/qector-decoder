"""BP-OSD on the bivariate-bicycle [[72,12]] code.

Validates that the BB(6,6) construction is a well-formed CSS code on 72 qubits
and that ``qd.bposd.BpOsdDecoder`` is syndrome-faithful (``(H @ c) & 1 == s``)
over many random errors at p=0.05, both per-shot and in batch mode.
"""

import numpy as np

from qector_decoder_v3 import codes
from qector_decoder_v3.bposd import BpOsdDecoder


def _bb72():
    return codes.bivariate_bicycle_code(6, 6, [("x", 3), ("y", 1), ("y", 2)], [("y", 3), ("x", 1), ("x", 2)])


def test_bb72_is_valid_css():
    cx, cz = _bb72()
    Hx = cx.parity_check_matrix().astype(np.uint8)
    Hz = cz.parity_check_matrix().astype(np.uint8)
    assert Hx.shape[1] == 72
    assert Hz.shape[1] == 72
    # CSS commutation: Hx @ Hz.T == 0 (mod 2).
    assert np.array_equal((Hx @ Hz.T) % 2, np.zeros((Hx.shape[0], Hz.shape[0]), np.uint8))


def test_bposd_bb72_faithful_per_shot():
    cx, _ = _bb72()
    H = cx.parity_check_matrix().astype(np.uint8)
    nq = H.shape[1]
    dec = BpOsdDecoder(H, error_rate=0.05, max_iter=30, osd_order=0)
    rng = np.random.default_rng(1)
    for _ in range(150):
        e = (rng.random(nq) < 0.05).astype(np.uint8)
        s = ((H @ e) & 1).astype(np.uint8)
        c = np.asarray(dec.decode(s)).astype(np.uint8)
        assert c.shape == (nq,)
        assert np.array_equal((H @ c) & 1, s)


def test_bposd_bb72_faithful_batch():
    cx, _ = _bb72()
    H = cx.parity_check_matrix().astype(np.uint8)
    nq = H.shape[1]
    dec = BpOsdDecoder(H, error_rate=0.05, max_iter=30, osd_order=0)
    rng = np.random.default_rng(2)
    shots = 150
    errs = (rng.random((shots, nq)) < 0.05).astype(np.uint8)
    syns = ((errs @ H.T) & 1).astype(np.uint8)
    out = np.asarray(dec.batch_decode(syns)).astype(np.uint8)
    assert out.shape == (shots, nq)
    for i in range(shots):
        assert np.array_equal((H @ out[i]) & 1, syns[i])
