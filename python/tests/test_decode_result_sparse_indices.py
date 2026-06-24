"""Section 15: DecodeResult.sparse_indices consistency.

``sparse_indices`` must equal ``np.nonzero(correction)[0]`` and its length must
equal ``hamming_weight``.
"""
import numpy as np
import pytest

from qector_decoder_v3 import codes, result


@pytest.mark.parametrize("flips", [[], [0], [0, 3], [1, 2, 4]])
def test_sparse_indices_match_nonzero(flips):
    code = codes.repetition_code(6)
    H = code.parity_check_matrix()
    e = np.zeros(code.n_qubits, np.uint8)
    for q in flips:
        e[q] = 1
    s = ((H @ e) & 1).astype(np.uint8)

    res = result.decode_with_diagnostics(code, s, kind="blossom")
    corr = np.asarray(res.as_uint8(), np.uint8)

    expected = np.nonzero(corr)[0]
    got = np.asarray(res.sparse_indices)
    assert np.array_equal(got, expected), f"{got} != {expected}"


@pytest.mark.parametrize("flips", [[], [0], [0, 3], [1, 2, 4]])
def test_sparse_indices_length_equals_hamming_weight(flips):
    code = codes.repetition_code(6)
    H = code.parity_check_matrix()
    e = np.zeros(code.n_qubits, np.uint8)
    for q in flips:
        e[q] = 1
    s = ((H @ e) & 1).astype(np.uint8)

    res = result.decode_with_diagnostics(code, s, kind="blossom")
    assert len(res.sparse_indices) == res.hamming_weight
