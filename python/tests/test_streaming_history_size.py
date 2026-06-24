"""Streaming decoder is usable and syndrome-faithful for any history_size.

``history_size`` controls how many past rounds the sliding history buffers; it
must not affect single-shot ``decode`` faithfulness.  We construct decoders with
several history sizes (1, 5, 20) and assert each decodes faithfully via the core
invariant ``(H @ corr) & 1 == s``, and that the per-syndrome ``decode`` result is
independent of history_size.
"""
import numpy as np
import pytest

import qector_decoder_v3 as qd
from qector_decoder_v3 import codes


def _reachable(H, nq, rng, p=0.15):
    e = (rng.random(nq) < p).astype(np.uint8)
    return ((H @ e) & 1).astype(np.uint8)


@pytest.mark.parametrize("history_size", [1, 5, 20])
def test_decode_faithful_regardless_of_history_size(history_size):
    code = codes.repetition_code(9)
    H = code.parity_check_matrix()
    nq = code.n_qubits
    dec = qd.StreamingDecoder(code.check_to_qubits, nq, history_size=history_size)
    assert dec.n_qubits == nq
    assert dec.n_checks == code.n_checks

    rng = np.random.default_rng(int(history_size) + 100)
    for _ in range(150):
        s = _reachable(H, nq, rng)
        c = np.asarray(dec.decode(s), np.uint8)
        assert c.shape == (nq,)
        assert np.array_equal((H @ c) & 1, s), (
            f"history_size={history_size}: H@c != s"
        )


@pytest.mark.parametrize("history_size", [1, 5, 20])
def test_update_usable_for_every_history_size(history_size):
    """update() returns a valid-length correction without error for any history_size."""
    code = codes.ring_code(7)
    H = code.parity_check_matrix()
    nq = code.n_qubits
    dec = qd.StreamingDecoder(code.check_to_qubits, nq, history_size=history_size)

    rng = np.random.default_rng(int(history_size) + 1)
    for _ in range(30):
        s = _reachable(H, nq, rng)
        c = np.asarray(dec.update(s), np.uint8)
        assert c.shape == (nq,)


def test_per_syndrome_decode_independent_of_history_size():
    """decode(s) gives the same correction across different history_size values."""
    code = codes.repetition_code(9)
    H = code.parity_check_matrix()
    nq = code.n_qubits

    decs = {h: qd.StreamingDecoder(code.check_to_qubits, nq, history_size=h)
            for h in (1, 5, 20)}

    rng = np.random.default_rng(4242)
    for _ in range(80):
        s = _reachable(H, nq, rng)
        ref = np.asarray(decs[1].decode(s), np.uint8)
        assert np.array_equal((H @ ref) & 1, s)
        for h in (5, 20):
            c = np.asarray(decs[h].decode(s), np.uint8)
            assert np.array_equal(c, ref), (
                f"decode differs between history_size 1 and {h}"
            )
