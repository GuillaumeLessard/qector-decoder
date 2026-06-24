"""Section 14: batch input shapes.

``batch_decode`` family requires 2-D input; 1-D or 3-D must raise ValueError.
A valid 2-D batch must return shape ``(n_shots, n_qubits)`` with every row
syndrome-faithful.
"""

import numpy as np
import pytest

import qector_decoder_v3 as qd
from qector_decoder_v3 import codes, bposd, pymatching_compat


def test_matching_decode_batch_rejects_1d_and_3d():
    code = codes.repetition_code(4)
    H = code.parity_check_matrix()
    m = pymatching_compat.Matching.from_check_matrix(H)
    with pytest.raises(ValueError):
        m.decode_batch(np.zeros(code.n_checks, np.uint8))
    with pytest.raises(ValueError):
        m.decode_batch(np.zeros((2, 2, code.n_checks), np.uint8))


def test_bposd_batch_decode_rejects_non_2d():
    code = codes.repetition_code(4)
    H = code.parity_check_matrix()
    bp = bposd.BpOsdDecoder(H)
    with pytest.raises(ValueError):
        bp.batch_decode(np.zeros(code.n_checks, np.uint8))
    with pytest.raises(ValueError):
        bp.batch_decode(np.zeros((2, 2, code.n_checks), np.uint8))


@pytest.mark.parametrize("DecCls", [qd.CPUBatchDecoder, qd.BlossomDecoder])
def test_valid_2d_batch_shape_and_faithfulness(DecCls):
    code = codes.repetition_code(5)
    H = code.parity_check_matrix()
    nq = code.n_qubits
    rng = np.random.default_rng(11)
    n_shots = 16
    E = (rng.random((n_shots, nq)) < 0.1).astype(np.uint8)
    S = ((E @ H.T) & 1).astype(np.uint8)

    dec = DecCls(code.check_to_qubits, nq)
    out = np.asarray(dec.batch_decode(S), np.uint8).reshape(n_shots, -1)
    assert out.shape == (n_shots, nq)
    for i in range(n_shots):
        assert np.array_equal((H @ out[i]) & 1, S[i]), f"row {i} not faithful"
