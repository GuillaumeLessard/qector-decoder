"""BP-OSD across OSD orders on the BB [[72,12]] code.

For osd_order in (0, 1, 2, 4) the decoder must remain syndrome-faithful, and a
higher OSD order must never produce a *heavier* correction than OSD-0 on the
same shot: OSD greedily improves on the order-0 pivot solution, so the Hamming
weight is monotonically non-increasing in the order.
"""

import numpy as np

from qector_decoder_v3 import codes
from qector_decoder_v3.bposd import BpOsdDecoder


def _bb72():
    return codes.bivariate_bicycle_code(6, 6, [("x", 3), ("y", 1), ("y", 2)], [("y", 3), ("x", 1), ("x", 2)])


def test_bposd_osd_orders_faithful_and_weight_monotone():
    cx, _ = _bb72()
    H = cx.parity_check_matrix().astype(np.uint8)
    nq = H.shape[1]
    p = 0.05
    orders = (0, 1, 2, 4)
    decs = {k: BpOsdDecoder(H, error_rate=p, max_iter=30, osd_order=k) for k in orders}

    rng = np.random.default_rng(41)
    shots = 80
    checked_nonzero = 0
    for _ in range(shots):
        e = (rng.random(nq) < p).astype(np.uint8)
        s = ((H @ e) & 1).astype(np.uint8)

        weights = {}
        for k in orders:
            c = np.asarray(decs[k].decode(s)).astype(np.uint8)
            # Every order must be syndrome-faithful.
            assert np.array_equal((H @ c) & 1, s)
            weights[k] = int(c.sum())

        w0 = weights[0]
        for k in (1, 2, 4):
            # Higher OSD order never yields a heavier correction than OSD-0.
            assert weights[k] <= w0, (k, weights[k], w0)
        if w0 > 0:
            checked_nonzero += 1

    # Make sure the weight comparison was exercised on real (non-trivial) shots.
    assert checked_nonzero > 0
