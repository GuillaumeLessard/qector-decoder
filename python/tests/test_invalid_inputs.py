"""Section 14: input validation.

Invalid inputs must raise a clean ValueError/TypeError -- never panic, abort or
segfault. Each case is wrapped in ``pytest.raises``.
"""
import numpy as np
import pytest

import qector_decoder_v3 as qd
from qector_decoder_v3 import codes, dem, result, bposd, pymatching_compat


def _small_dem():
    text = (
        "error(0.1) D0 L0\n"
        "error(0.1) D0 D1\n"
        "error(0.1) D1 L0\n"
    )
    return dem.parse_dem(text)


def test_bposd_decoder_1d_H_raises():
    H1d = np.array([1, 1, 0], np.uint8)
    with pytest.raises((ValueError, TypeError)):
        bposd.BpOsdDecoder(H1d)


def test_qd_bposd_decoder_1d_H_raises():
    H1d = np.array([1, 1, 0], np.uint8)
    with pytest.raises((ValueError, TypeError)):
        qd.BPOSDDecoder(H1d)


def test_matching_from_1d_check_matrix_raises():
    H1d = np.array([1, 1, 0], np.uint8)
    with pytest.raises((ValueError, TypeError)):
        pymatching_compat.Matching.from_check_matrix(H1d)


def test_predicted_observables_wrong_length_raises():
    dm = _small_dem()
    bad = np.zeros(dm.num_errors + 5, np.uint8)
    with pytest.raises((ValueError, TypeError)):
        dm.predicted_observables(bad)


def test_decode_with_diagnostics_unknown_kind_raises():
    code = codes.repetition_code(3)
    H = code.parity_check_matrix()
    e = np.zeros(code.n_qubits, np.uint8)
    e[0] = 1
    s = ((H @ e) & 1).astype(np.uint8)
    with pytest.raises(ValueError):
        result.decode_with_diagnostics(code, s, kind="nonsense")


def test_dem_make_decoder_unknown_kind_raises():
    dm = _small_dem()
    with pytest.raises(ValueError):
        dm.make_decoder("nonsense")
