"""Tests for qector_decoder_v3.predecoder."""

import numpy as np

from qector_decoder_v3 import codes
from qector_decoder_v3.predecoder import PredecodedDecoder, quantize_weights


def test_predecoder_is_faithful():
    code = codes.rotated_surface_code(7)
    H = code.parity_check_matrix()
    dec = PredecodedDecoder(code.check_to_qubits, code.n_qubits, backend="blossom")
    rng = np.random.default_rng(0)
    for _ in range(300):
        e = (rng.random(code.n_qubits) < 0.07).astype(np.uint8)
        s = (H @ e) & 1
        c = np.asarray(dec.decode(s.astype(np.uint8))).astype(np.uint8)
        assert np.array_equal((H @ c) & 1, s)


def test_predecoder_resolves_adjacent_pairs():
    code = codes.repetition_code(15)
    H = code.parity_check_matrix()
    dec = PredecodedDecoder(code.check_to_qubits, code.n_qubits, backend="union_find")
    # a single error gives two adjacent defects -> predecoder resolves them
    e = np.zeros(code.n_qubits, np.uint8)
    e[5] = 1
    s = (H @ e) & 1
    c = np.asarray(dec.decode(s)).astype(np.uint8)
    assert np.array_equal((H @ c) & 1, s)
    assert dec.last_predecoded >= 2


def test_predecoder_all_backends_faithful():
    code = codes.rotated_surface_code(5)
    H = code.parity_check_matrix()
    rng = np.random.default_rng(1)
    syns = (((rng.random((200, code.n_qubits)) < 0.08).astype(np.uint8)) @ H.T) & 1
    for backend in ("blossom", "union_find", "sparse_blossom"):
        dec = PredecodedDecoder(code.check_to_qubits, code.n_qubits, backend=backend)
        out = dec.batch_decode(syns.astype(np.uint8))
        for i in range(len(syns)):
            assert np.array_equal((H @ out[i]) & 1, syns[i]), backend


def test_quantize_weights_handles_inf():
    q = quantize_weights([0.1, 2.5, 7.0, np.inf, 0.0])
    assert q.dtype == np.int64
    assert (q >= 1).all()
    assert q[3] == q.max()  # inf -> top level
    # monotonic in the finite weights
    assert q[0] <= q[1] <= q[2]
