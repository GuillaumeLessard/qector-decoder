"""Regression: GPU probes must degrade to False, never raise.

Exposed by scripts/competitive_extended.py (K3dev E4/C1): on a host without
cupy, ``BpOsdDecoder.batch_decode`` died with ``ModuleNotFoundError`` because
``gpu_backend._cupy`` only caught ``RuntimeError``, not ``ImportError``.
"""

from __future__ import annotations

import numpy as np
from qector_decoder_v3 import codes
from qector_decoder_v3 import gpu_backend as gb
from qector_decoder_v3.bposd import BpOsdDecoder


def test_gpu_probes_never_raise():
    # On this machine cupy is not installed: both must return False, not raise.
    assert gb.has_cupy() in (True, False)
    if not gb.has_cupy():
        assert gb.gpu_available() is False


def test_bposd_batch_decode_works_without_cupy():
    cx, _ = codes.bivariate_bicycle_code(6, 6, [("x", 3), ("y", 1), ("y", 2)], [("y", 3), ("x", 1), ("x", 2)])
    H = cx.parity_check_matrix()
    dec = BpOsdDecoder(H, error_rate=0.03, max_iter=10)  # use_gpu=None (auto)
    rng = np.random.default_rng(0)
    # Sample real errors: BB72's Hx has rank 30 (36 rows), so arbitrary random
    # syndromes are generally unsatisfiable and faithfulness is undefined there.
    E = (rng.random((8, H.shape[1])) < 0.03).astype(np.uint8)
    S = ((E @ H.T) & 1).astype(np.uint8)
    C = np.asarray(dec.batch_decode(S), dtype=np.uint8)
    assert C.shape == (8, H.shape[1])
    assert np.all(((C @ H.T) & 1) == S)  # faithful on satisfiable syndromes
