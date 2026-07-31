"""B2-05: DEM per-edge weights must drive matching, and make_decoder must thread them.

The semantic guarantee is verified against the installed native ``BlossomDecoder``
(which already accepts ``edge_weights``). The ``make_decoder`` wiring lives in
repo ``dem.py`` and is asserted by source inspection so the test stays green
before the next wheel install (conftest prefers the installed package).
"""

from __future__ import annotations

import math
from pathlib import Path

import numpy as np

from qector_decoder_v3 import BlossomDecoder, dem

# Syndrome [1, 0, 1]:
#   direct:   mech 0 (D0 D2, p=0.001) — one expensive edge
#   via D1:   mech 1 + mech 2 (p=0.4 each) — two cheap edges
# Uniform weights pick the direct edge (cost 1 < 2).
# DEM weights pick the two-hop path (2·ln(1.5) ≪ ln(999)).
SKEWED_DEM = """
error(0.001) D0 D2
error(0.4) D0 D1
error(0.4) D1 D2
"""


def test_dem_weights_change_blossom_matching():
    m = dem.parse_dem(SKEWED_DEM)
    H = m.check_matrix()
    syndrome = np.array([1, 0, 1], dtype=np.uint8)
    w = m.weights().tolist()
    assert w[0] > w[1] + w[2]

    corr_w = np.asarray(
        BlossomDecoder(m.check_to_qubits(), m.num_errors, w).decode(syndrome),
        dtype=np.uint8,
    ).reshape(-1)
    corr_u = np.asarray(
        BlossomDecoder(m.check_to_qubits(), m.num_errors).decode(syndrome),
        dtype=np.uint8,
    ).reshape(-1)

    assert np.array_equal((H @ corr_w) & 1, syndrome)
    assert np.array_equal((H @ corr_u) & 1, syndrome)
    assert corr_w.tolist() == [0, 1, 1], f"weighted correction={corr_w.tolist()}"
    assert corr_u.tolist() == [1, 0, 0], f"uniform correction={corr_u.tolist()}"


def test_repo_make_decoder_threads_weights():
    """Lock the B2-05 wiring in the repo's dem.py (may not yet be installed)."""
    path = Path(__file__).resolve().parents[1] / "qector_decoder_v3" / "dem.py"
    src = path.read_text(encoding="utf-8")
    assert "w = self.weights().tolist()" in src
    assert "BlossomDecoder(c2q, nq, w)" in src
    assert "SparseBlossomDecoder(c2q, nq, w)" in src


def test_to_code_still_attaches_qubit_weights():
    m = dem.parse_dem(SKEWED_DEM)
    code = m.to_code()
    assert code.qubit_weights is not None
    assert len(code.qubit_weights) == m.num_errors
    assert math.isclose(float(code.qubit_weights[0]), float(m.weights()[0]), rel_tol=1e-9)
