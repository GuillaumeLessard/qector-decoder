"""Section 14: input validation.

Invalid inputs must raise a clean ValueError/TypeError -- never panic, abort or
segfault. Each case is wrapped in ``pytest.raises``.
"""

import numpy as np
import pytest
import qector_decoder_v3 as qd
from qector_decoder_v3 import bposd, codes, dem, pymatching_compat, result


def _small_dem():
    text = "error(0.1) D0 L0\nerror(0.1) D0 D1\nerror(0.1) D1 L0\n"
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


# ---------------------------------------------------------------------------
# UF / FastUF input validation matrix (from v0.6.2 todo.md)
# Covers: empty, negative, duplicates, range, type coercion (np.int*, bool_),
# hypergraph (high qubit degree) rejection.
# ---------------------------------------------------------------------------


def test_uf_validation_empty_top_level():
    with pytest.raises(ValueError, match="non-empty"):
        qd.UnionFindDecoder([], 4)


def test_uf_validation_empty_check():
    with pytest.raises(ValueError, match="All checks must be non-empty"):
        qd.UnionFindDecoder([[0, 1], []], 3)


def test_uf_validation_negative_index():
    with pytest.raises(ValueError, match="Negative"):
        qd.FastUnionFindDecoder([[-1, 0]], 2)


def test_uf_validation_duplicate_in_check():
    with pytest.raises(ValueError, match="Duplicate"):
        qd.UnionFindDecoder([[0, 0, 1]], 3)


def test_uf_validation_out_of_range():
    with pytest.raises(ValueError, match=">= n_qubits"):
        qd.UnionFindDecoder([[0, 5]], n_qubits=3)


def test_uf_validation_type_coercion_np_int_and_bool():
    # np.int32, np.bool_ etc must be accepted and normalized
    checks_i32 = np.array([[0, 1], [1, 2]], dtype=np.int32).tolist()
    dec = qd.UnionFindDecoder(checks_i32, 3)
    assert dec.n_qubits == 3

    # bool_ in checks (0/1)
    checks_bool = [[np.bool_(0), np.bool_(1)], [np.bool_(1), np.int_(2)]]
    dec2 = qd.FastUnionFindDecoder(checks_bool, 3)
    syn = np.array([1, 0], dtype=np.uint8)
    corr = dec2.decode(syn)
    assert corr.shape == (3,)


def test_unionfind_rejects_hypergraph_weight4():
    # High-degree qubit (the actual condition that used to be accepted silently).
    # A qubit participating in >2 checks must be rejected with clear message.
    bad_checks = [[0, 1], [0, 2], [0, 3]]  # qubit 0 in 3 checks
    with pytest.raises(ValueError, match="hyperedge|weight|participates in"):
        qd.UnionFindDecoder(bad_checks, 4)
    with pytest.raises(ValueError, match="hyperedge|weight|participates in"):
        qd.FastUnionFindDecoder(bad_checks, 4)


def test_uf_rejects_non_uint8_syndrome_shape():
    dec = qd.UnionFindDecoder([[0, 1]], 2)
    with pytest.raises((ValueError, TypeError)):
        dec.decode(np.array([1.0, 0.0]))  # wrong dtype triggers in wrapper or core
