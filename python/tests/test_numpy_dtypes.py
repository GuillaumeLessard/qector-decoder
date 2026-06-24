"""Section 14: numpy dtype handling.

A decoder must accept syndromes expressed in uint8, bool, int32 and int64 and
still produce a syndrome-faithful correction. The batch path converts the dtype
internally; the single-decode path accepts native uint8 directly. Faithfulness
is asserted via the core invariant ``(H @ c) & 1 == s`` against the canonical
uint8 syndrome.
"""

import numpy as np
import pytest

import qector_decoder_v3 as qd
from qector_decoder_v3 import codes


DECODER_CLASSES = [
    qd.BlossomDecoder,
    qd.SparseBlossomDecoder,
    qd.UnionFindDecoder,
]

DTYPES = [np.uint8, bool, np.int32, np.int64]


def _build_case(code, seed=0, p=0.12):
    H = code.parity_check_matrix()
    nq = code.n_qubits
    rng = np.random.default_rng(seed)
    e = (rng.random(nq) < p).astype(np.uint8)
    s = ((H @ e) & 1).astype(np.uint8)
    return H, s


@pytest.mark.parametrize(
    "code_factory",
    [
        lambda: codes.repetition_code(5),
        lambda: codes.rotated_surface_code(3),
    ],
)
@pytest.mark.parametrize("DecCls", DECODER_CLASSES)
@pytest.mark.parametrize("dtype", DTYPES)
def test_batch_decode_accepts_each_dtype(code_factory, DecCls, dtype):
    code = code_factory()
    H, s = _build_case(code)
    dec = DecCls(code.check_to_qubits, code.n_qubits)

    # one-row batch in the requested dtype; the layer converts internally
    batch = np.ascontiguousarray(s.reshape(1, -1).astype(dtype))
    out = np.asarray(dec.batch_decode(batch), np.uint8).reshape(1, -1)
    corr = out[0]
    assert np.array_equal((H @ corr) & 1, s), f"{DecCls.__name__} dtype={np.dtype(dtype)} not syndrome-faithful"


@pytest.mark.parametrize(
    "code_factory",
    [
        lambda: codes.repetition_code(5),
        lambda: codes.rotated_surface_code(3),
    ],
)
@pytest.mark.parametrize("DecCls", DECODER_CLASSES)
def test_single_decode_uint8(code_factory, DecCls):
    code = code_factory()
    H, s = _build_case(code, seed=1)
    dec = DecCls(code.check_to_qubits, code.n_qubits)
    corr = np.asarray(dec.decode(s.astype(np.uint8)), np.uint8)
    assert np.array_equal((H @ corr) & 1, s)
