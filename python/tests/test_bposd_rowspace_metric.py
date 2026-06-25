"""The correct LDPC logical-failure metric: stabiliser-rowspace membership of the residual.

Section 6 (logical-observable proof). For a CSS LDPC code decoded over ``Hx`` by BP-OSD,
the correct logical-failure criterion for the residual ``r = c ^ e`` is:

* ``r`` is *always* in ``ker(Hx)`` (BP-OSD is syndrome-faithful), and
* ``r`` is a *logical failure* iff it is NOT in the GF(2) row space of the Z-stabilisers
  ``Hz`` -- i.e. it is not a product of stabilisers.

The naive ``c != e`` metric over-counts, because a non-zero residual that lies in the
``Hz`` row space is a harmless stabiliser. We demonstrate that such residuals exist
(deterministically, by planting a stabiliser error), so the rowspace metric is required.

Code: bivariate-bicycle ``[[72, 12]]``.
"""

import numpy as np

from qector_decoder_v3 import codes
from qector_decoder_v3.bposd import BpOsdDecoder


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


def _bb72():
    return codes.bivariate_bicycle_code(
        6, 6, [("x", 3), ("y", 1), ("y", 2)], [("y", 3), ("x", 1), ("x", 2)]
    )


def test_residual_always_in_ker_hx():
    cx, _ = _bb72()
    Hx = cx.parity_check_matrix().astype(np.uint8)
    nq = Hx.shape[1]
    p = 0.03
    dec = BpOsdDecoder(Hx, error_rate=p, max_iter=30, osd_order=0)
    rng = np.random.default_rng(11)
    zero = np.zeros(Hx.shape[0], np.uint8)
    for _ in range(300):
        e = (rng.random(nq) < p).astype(np.uint8)
        s = ((Hx @ e) & 1).astype(np.uint8)
        c = np.asarray(dec.decode(s), np.uint8)
        assert np.array_equal((Hx @ c) & 1, s)  # syndrome-faithful
        r = c ^ e
        assert np.array_equal((Hx @ r) & 1, zero)  # residual in ker(Hx) always


def test_logical_failure_is_rowspace_membership():
    cx, cz = _bb72()
    Hx = cx.parity_check_matrix().astype(np.uint8)
    Hz = cz.parity_check_matrix().astype(np.uint8)
    nq = Hx.shape[1]
    Zbasis = _gf2_rowspace_basis(Hz)
    p = 0.03
    dec = BpOsdDecoder(Hx, error_rate=p, max_iter=30, osd_order=4)
    rng = np.random.default_rng(13)

    shots = 600
    failures = 0
    for _ in range(shots):
        e = (rng.random(nq) < p).astype(np.uint8)
        s = ((Hx @ e) & 1).astype(np.uint8)
        c = np.asarray(dec.decode(s), np.uint8)
        r = c ^ e
        # logical failure := residual NOT in Hz row space.
        if not _in_span(Zbasis, r):
            failures += 1
    ler = failures / shots
    # A valid probability and competitive (BP-OSD at p=0.03 on BB[[72,12]]).
    assert 0.0 <= ler <= 1.0
    assert ler < 0.5


def test_harmless_stabilizer_residual_exists_deterministically():
    """A planted stabiliser error yields a non-zero residual that IS in the Hz rowspace.

    This is the case the naive ``c != e`` metric overcounts: ``c != e`` (failure flagged)
    yet ``r`` is a product of stabilisers (no logical failure under the rowspace metric).
    Built deterministically from a row of ``Hz`` so it never flakes.
    """
    cx, cz = _bb72()
    Hx = cx.parity_check_matrix().astype(np.uint8)
    Hz = cz.parity_check_matrix().astype(np.uint8)
    Zbasis = _gf2_rowspace_basis(Hz)
    dec = BpOsdDecoder(Hx, error_rate=0.03, max_iter=30, osd_order=0)

    zero_hx = np.zeros(Hx.shape[0], np.uint8)
    found = 0
    for row in range(Hz.shape[0]):
        st = Hz[row].astype(np.uint8)
        if not st.any():
            continue
        # A Z-stabiliser is in ker(Hx) (CSS) and trivially in the Hz row space.
        assert np.array_equal((Hx @ st) & 1, zero_hx)
        assert _in_span(Zbasis, st)

        # Plant e = st: zero syndrome -> decoder returns all-zeros -> r = st.
        s = ((Hx @ st) & 1).astype(np.uint8)
        assert not s.any()
        c = np.asarray(dec.decode(s), np.uint8)
        r = c ^ st

        # Concrete harmless-but-different case.
        if r.any() and _in_span(Zbasis, r) and not np.array_equal(c, st):
            found += 1
            # The rowspace metric correctly reports NO logical failure here,
            # whereas naive c != e would (wrongly) count a failure.
            assert _in_span(Zbasis, r)  # no logical failure
            assert not np.array_equal(c, st)  # but c != e

    assert found > 0, "expected at least one non-zero in-rowspace residual"


def test_some_random_residual_in_rowspace():
    """Across random shots, at least some non-zero residuals land in the Hz rowspace.

    These are harmless stabilisers that ``c != e`` would overcount. We mix planted
    stabilisers into the error so the case is reliably exercised without large shot
    counts (a pure depolarising channel produces them only rarely at this distance).
    """
    cx, cz = _bb72()
    Hx = cx.parity_check_matrix().astype(np.uint8)
    Hz = cz.parity_check_matrix().astype(np.uint8)
    nq = Hx.shape[1]
    Zbasis = _gf2_rowspace_basis(Hz)
    dec = BpOsdDecoder(Hx, error_rate=0.03, max_iter=30, osd_order=0)
    rng = np.random.default_rng(17)

    nonzero_in_span = 0
    naive_only = 0
    shots = 400
    for _ in range(shots):
        # Light error + an added stabiliser (same syndrome as the light error alone).
        e_light = (rng.random(nq) < 0.01).astype(np.uint8)
        st = Hz[rng.integers(0, Hz.shape[0])].astype(np.uint8)
        e = e_light ^ st
        s = ((Hx @ e) & 1).astype(np.uint8)
        c = np.asarray(dec.decode(s), np.uint8)
        assert np.array_equal((Hx @ c) & 1, s)
        r = c ^ e
        if r.any() and _in_span(Zbasis, r):
            nonzero_in_span += 1
            if not np.array_equal(c, e):
                naive_only += 1

    # The rowspace metric is genuinely needed: these are non-zero, in-span residuals.
    assert nonzero_in_span > 0
    # And the naive c != e metric would have flagged them as failures.
    assert naive_only > 0
