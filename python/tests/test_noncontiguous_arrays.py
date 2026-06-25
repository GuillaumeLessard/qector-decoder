"""Section 14: non-contiguous / Fortran-ordered syndrome arrays.

Syndrome data does not always arrive C-contiguous (e.g. a strided column view
or an ``order='F'`` 2-D batch). The supported idiom for such inputs is to
normalise them with ``np.ascontiguousarray`` before decoding; this test checks
that the normalised result is syndrome-faithful for both single and batch
decode. It also asserts that handing a raw non-contiguous array to the layer
never crashes the process -- it either decodes faithfully or raises a clean
Python exception.
"""

import numpy as np

import qector_decoder_v3 as qd
from qector_decoder_v3 import codes


def _rep_case(seed=2, p=0.1):
    code = codes.repetition_code(5)
    H = code.parity_check_matrix()
    nq = code.n_qubits
    rng = np.random.default_rng(seed)
    e = (rng.random(nq) < p).astype(np.uint8)
    s = ((H @ e) & 1).astype(np.uint8)
    return code, H, s


def test_single_decode_noncontiguous_1d_view():
    code, H, s = _rep_case()
    dec = qd.BlossomDecoder(code.check_to_qubits, code.n_qubits)

    # embed s as the middle column of a 2-D buffer, then take a strided view
    buf = np.zeros((code.n_checks, 3), np.uint8)
    buf[:, 1] = s
    view = buf[:, 1]
    assert not view.flags["C_CONTIGUOUS"], "view should be non-contiguous"

    corr = np.asarray(dec.decode(np.ascontiguousarray(view)), np.uint8)
    assert np.array_equal((H @ corr) & 1, s)


def test_batch_decode_fortran_ordered_2d():
    code = codes.repetition_code(5)
    H = code.parity_check_matrix()
    nq = code.n_qubits
    rng = np.random.default_rng(7)
    N = 12
    E = (rng.random((N, nq)) < 0.1).astype(np.uint8)
    S = ((E @ H.T) & 1).astype(np.uint8)

    Sf = np.asfortranarray(S)
    assert Sf.flags["F_CONTIGUOUS"] and not Sf.flags["C_CONTIGUOUS"]

    dec = qd.BlossomDecoder(code.check_to_qubits, nq)
    out = np.asarray(dec.batch_decode(np.ascontiguousarray(Sf)), np.uint8).reshape(
        N, -1
    )
    assert out.shape == (N, nq)
    for i in range(N):
        assert np.array_equal((H @ out[i]) & 1, S[i]), f"row {i} not faithful"


def test_raw_noncontiguous_never_crashes():
    # Defensive: passing a raw strided/F-ordered array must not segfault.
    # Acceptable outcomes are a faithful decode OR a clean Python exception.
    code, H, s = _rep_case(seed=3)
    dec = qd.BlossomDecoder(code.check_to_qubits, code.n_qubits)

    buf = np.zeros((code.n_checks, 2), np.uint8)
    buf[:, 0] = s
    view = buf[:, 0]
    assert not view.flags["C_CONTIGUOUS"]
    try:
        corr = np.asarray(dec.decode(view), np.uint8)
    except (ValueError, TypeError):
        pass  # clean rejection is acceptable
    else:
        assert np.array_equal((H @ corr) & 1, s)
