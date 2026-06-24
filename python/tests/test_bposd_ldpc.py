"""Tests for BP-OSD (qector_decoder_v3.bposd) and the LDPC code generators.

Faithfulness is checked directly; where the `ldpc` package is installed, QECTOR's
BP-OSD logical error rate is cross-validated against the reference BP-OSD.
"""
import numpy as np
import pytest

from qector_decoder_v3 import codes
from qector_decoder_v3.bposd import BpOsdDecoder


def _bb72():
    return codes.bivariate_bicycle_code(
        6, 6, [("x", 3), ("y", 1), ("y", 2)], [("y", 3), ("x", 1), ("x", 2)]
    )


def test_bb_code_is_valid_css():
    cx, cz = _bb72()
    Hx, Hz = cx.parity_check_matrix(), cz.parity_check_matrix()
    assert Hx.shape[1] == Hz.shape[1] == 72
    assert np.array_equal((Hx @ Hz.T) % 2, np.zeros((Hx.shape[0], Hz.shape[0]), np.uint8))


def test_bicycle_code_is_valid_css():
    bx, bz = codes.bicycle_code(12, weight=6, seed=1)
    Hx, Hz = bx.parity_check_matrix(), bz.parity_check_matrix()
    assert np.array_equal((Hx @ Hz.T) % 2, np.zeros((Hx.shape[0], Hz.shape[0]), np.uint8))


@pytest.mark.parametrize("osd_order", [0, 4])
def test_bposd_is_faithful_on_bb_code(osd_order):
    cx, _ = _bb72()
    H = cx.parity_check_matrix()
    nq = H.shape[1]
    dec = BpOsdDecoder(H, error_rate=0.05, max_iter=30, osd_order=osd_order)
    rng = np.random.default_rng(1)
    for _ in range(200):
        e = (rng.random(nq) < 0.05).astype(np.uint8)
        s = (H @ e) & 1
        c = np.asarray(dec.decode(s)).astype(np.uint8)
        assert np.array_equal((H @ c) & 1, s)


def test_bposd_faithful_on_hypergraph_product():
    H1 = np.array([[1, 1, 0], [0, 1, 1], [1, 0, 1]], dtype=np.uint8)
    cx, _ = codes.hypergraph_product(H1)
    H = cx.parity_check_matrix()
    dec = BpOsdDecoder(H, error_rate=0.05, max_iter=30, osd_order=0)
    rng = np.random.default_rng(2)
    for _ in range(150):
        e = (rng.random(H.shape[1]) < 0.05).astype(np.uint8)
        s = (H @ e) & 1
        c = np.asarray(dec.decode(s)).astype(np.uint8)
        assert np.array_equal((H @ c) & 1, s)


def test_bposd_batch():
    cx, _ = _bb72()
    H = cx.parity_check_matrix()
    dec = BpOsdDecoder(H, error_rate=0.05)
    rng = np.random.default_rng(3)
    errs = (rng.random((50, H.shape[1])) < 0.05).astype(np.uint8)
    syns = (errs @ H.T) & 1
    out = dec.batch_decode(syns.astype(np.uint8))
    assert out.shape == (50, H.shape[1])
    for i in range(50):
        assert np.array_equal((H @ out[i]) & 1, syns[i])


def _gf2_rowspace_basis(M):
    rows = []
    for r in (np.asarray(M) % 2).astype(np.uint8):
        v = r.copy()
        for b in rows:
            if v[np.argmax(b)]:
                v ^= b
        if v.any():
            rows.append(v)
    return rows


def _in_span(basis, v):
    v = (np.asarray(v) % 2).astype(np.uint8)
    for b in basis:
        if v[np.argmax(b)]:
            v ^= b
    return not v.any()


def test_bposd_logical_error_rate_competitive_with_ldpc():
    """QECTOR BP-OSD logical error rate is competitive with the `ldpc` package.

    Logical failure uses the correct CSS criterion: the residual ``c ^ e`` lies in
    ker(Hx) (always, since both reproduce the syndrome), and is a *logical* failure
    iff it is NOT in the row space of the Z-stabilisers ``Hz`` (i.e. not a product
    of stabilisers). Counting ``c != e`` would overcount harmless stabiliser shifts.
    """
    pytest.importorskip("ldpc")
    from ldpc import BpOsdDecoder as RefBpOsd

    cx, cz = _bb72()
    Hx = cx.parity_check_matrix().astype(np.uint8)
    Hz = cz.parity_check_matrix().astype(np.uint8)
    nq = Hx.shape[1]
    Zbasis = _gf2_rowspace_basis(Hz)
    p = 0.03
    shots = 1200
    rng = np.random.default_rng(7)
    errs = (rng.random((shots, nq)) < p).astype(np.uint8)
    syns = ((errs @ Hx.T) & 1).astype(np.uint8)

    q = BpOsdDecoder(Hx, error_rate=p, max_iter=30, osd_order=10)
    ref = RefBpOsd(Hx, error_rate=p, max_iter=30, osd_order=10,
                   bp_method="product_sum", osd_method="osd_cs")

    q_fail = r_fail = 0
    for i in range(shots):
        cq = np.asarray(q.decode(syns[i])).astype(np.uint8)
        assert np.array_equal((Hx @ cq) & 1, syns[i])
        if not _in_span(Zbasis, cq ^ errs[i]):
            q_fail += 1
        cr = np.asarray(ref.decode(syns[i])).astype(np.uint8)
        if not _in_span(Zbasis, cr ^ errs[i]):
            r_fail += 1

    # QECTOR BP-OSD logical error rate within 1.5x of the reference (+slack).
    assert q_fail <= int(1.5 * r_fail) + 10, (q_fail, r_fail)
