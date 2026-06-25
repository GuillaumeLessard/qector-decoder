"""BP-OSD belief-propagation modes on the BB [[72,12]] code.

Both ``bp_method="sum_product"`` and ``bp_method="min_sum"`` must yield
syndrome-faithful corrections over many random errors at p=0.05.
"""

import numpy as np
import pytest

from qector_decoder_v3 import codes
from qector_decoder_v3.bposd import BpOsdDecoder


def _bb72():
    return codes.bivariate_bicycle_code(
        6, 6, [("x", 3), ("y", 1), ("y", 2)], [("y", 3), ("x", 1), ("x", 2)]
    )


@pytest.mark.parametrize("bp_method", ["sum_product", "min_sum"])
def test_bposd_bp_mode_faithful(bp_method):
    cx, _ = _bb72()
    H = cx.parity_check_matrix().astype(np.uint8)
    nq = H.shape[1]
    dec = BpOsdDecoder(
        H, error_rate=0.05, max_iter=30, osd_order=0, bp_method=bp_method
    )
    rng = np.random.default_rng(51)
    for _ in range(120):
        e = (rng.random(nq) < 0.05).astype(np.uint8)
        s = ((H @ e) & 1).astype(np.uint8)
        c = np.asarray(dec.decode(s)).astype(np.uint8)
        assert c.shape == (nq,)
        assert np.array_equal((H @ c) & 1, s)
