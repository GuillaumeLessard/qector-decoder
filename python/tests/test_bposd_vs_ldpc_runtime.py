"""QECTOR BP-OSD vs the reference ``ldpc`` BP-OSD on the BB [[72,12]] code.

Cross-validates QECTOR's ``BpOsdDecoder`` against ``ldpc.BpOsdDecoder`` at
p=0.03 over ~800 shots:

* QECTOR must be syndrome-faithful on every shot.
* The logical-failure metric is the *correct* CSS criterion: the residual
  ``c ^ e`` is in ker(Hx) (always, both reproduce the syndrome) and is a
  *logical* failure iff it is NOT in the GF(2) row space of the Z-stabilisers
  ``Hz`` (i.e. it is not a product of stabilisers).  Counting ``c != e`` would
  overcount harmless stabiliser shifts and is forbidden.
* QECTOR's logical error count must stay within 1.5x of the reference (+slack).
* Wall-clock time of both is measured and the ratio is printed, but runtime is
  NOT asserted on (machine load varies) -- only that both ran faithfully.
"""
import time

import numpy as np
import pytest


def _bb72():
    from qector_decoder_v3 import codes

    return codes.bivariate_bicycle_code(
        6, 6, [("x", 3), ("y", 1), ("y", 2)], [("y", 3), ("x", 1), ("x", 2)]
    )


def _gf2_rowspace_basis(M):
    """Reduced GF(2) basis of the row space of ``M`` (pivot-indexed elimination)."""
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
    """True iff ``v`` lies in the GF(2) span described by ``basis``."""
    v = (np.asarray(v) % 2).astype(np.uint8)
    for b in basis:
        if v[np.argmax(b)]:
            v ^= b
    return not v.any()


def test_bposd_vs_ldpc_runtime():
    pytest.importorskip("ldpc")
    from ldpc import BpOsdDecoder as RefBpOsd

    from qector_decoder_v3.bposd import BpOsdDecoder

    cx, cz = _bb72()
    Hx = cx.parity_check_matrix().astype(np.uint8)
    Hz = cz.parity_check_matrix().astype(np.uint8)
    nq = Hx.shape[1]
    Zbasis = _gf2_rowspace_basis(Hz)

    p = 0.03
    shots = 800
    rng = np.random.default_rng(7)
    errs = (rng.random((shots, nq)) < p).astype(np.uint8)
    syns = ((errs @ Hx.T) & 1).astype(np.uint8)

    q = BpOsdDecoder(Hx, error_rate=p, max_iter=30, osd_order=0)
    ref = RefBpOsd(
        Hx,
        error_rate=p,
        max_iter=30,
        osd_order=0,
        bp_method="product_sum",
        osd_method="osd_cs",
    )

    # --- QECTOR ---
    t0 = time.perf_counter()
    q_fail = 0
    for i in range(shots):
        cq = np.asarray(q.decode(syns[i])).astype(np.uint8)
        assert np.array_equal((Hx @ cq) & 1, syns[i])  # faithful every shot
        if not _in_span(Zbasis, cq ^ errs[i]):
            q_fail += 1
    q_time = time.perf_counter() - t0

    # --- reference ldpc ---
    t0 = time.perf_counter()
    r_fail = 0
    for i in range(shots):
        cr = np.asarray(ref.decode(syns[i])).astype(np.uint8)
        if not _in_span(Zbasis, cr ^ errs[i]):
            r_fail += 1
    r_time = time.perf_counter() - t0

    ratio = q_time / r_time if r_time > 0 else float("inf")
    print(
        f"\nBP-OSD BB[[72,12]] p={p} shots={shots}: "
        f"QECTOR fail={q_fail} ({q_time:.3f}s), "
        f"ldpc fail={r_fail} ({r_time:.3f}s), time ratio={ratio:.2f}x"
    )

    # Both produced output over a non-trivial workload.
    assert q_time > 0.0 and r_time > 0.0
    # QECTOR logical error rate within 1.5x of the reference (+slack); NOT a
    # runtime assertion (deliberately not hard-failing on wall clock).
    assert q_fail <= int(1.5 * r_fail) + 10, (q_fail, r_fail)
