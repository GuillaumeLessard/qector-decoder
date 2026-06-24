"""Sliding-window decoder: round bookkeeping and per-round faithfulness.

Asserts the configured ``window_size`` / ``decay_factor`` are stored, that
``current_round`` increments once per ``update`` and resets to 0 on ``flush``,
that every returned correction has length ``n_qubits``, and that the single-shot
``decode`` obeys the core invariant ``(H @ corr) & 1 == s``.
"""
import numpy as np
import pytest

import qector_decoder_v3 as qd
from qector_decoder_v3 import codes


def _reachable(H, nq, rng, p=0.15):
    e = (rng.random(nq) < p).astype(np.uint8)
    return ((H @ e) & 1).astype(np.uint8)


def test_stored_config_and_round_counting():
    code = codes.repetition_code(7)
    H = code.parity_check_matrix()
    nq = code.n_qubits
    W, f = 6, 0.7
    dec = qd.SlidingWindowDecoder(code.check_to_qubits, nq, window_size=W,
                                  decay_factor=f)
    assert dec.window_size == W
    assert dec.decay_factor == pytest.approx(f)
    assert dec.current_round == 0
    assert dec.n_qubits == nq

    rng = np.random.default_rng(11)
    n_rounds = 25
    for i in range(n_rounds):
        s = _reachable(H, nq, rng)
        corr = np.asarray(dec.update(s), np.uint8)
        assert corr.shape == (nq,)
        assert dec.current_round == i + 1
    assert dec.current_round == n_rounds


def test_flush_resets_current_round():
    code = codes.ring_code(8)
    nq = code.n_qubits
    dec = qd.SlidingWindowDecoder(code.check_to_qubits, nq, window_size=4,
                                  decay_factor=0.5)
    H = code.parity_check_matrix()
    rng = np.random.default_rng(3)
    for _ in range(7):
        dec.update(_reachable(H, nq, rng))
    assert dec.current_round == 7
    dec.flush()
    assert dec.current_round == 0
    # usable again after flush
    dec.update(_reachable(H, nq, rng))
    assert dec.current_round == 1


@pytest.mark.parametrize(
    "code",
    [codes.repetition_code(7), codes.ring_code(7)],
    ids=lambda c: c.name,
)
def test_sliding_decode_is_syndrome_faithful(code):
    H = code.parity_check_matrix()
    nq = code.n_qubits
    dec = qd.SlidingWindowDecoder(code.check_to_qubits, nq, window_size=5,
                                  decay_factor=0.8)
    rng = np.random.default_rng(77)
    for _ in range(150):
        s = _reachable(H, nq, rng)
        c = np.asarray(dec.decode(s), np.uint8)
        assert c.shape == (nq,)
        assert np.array_equal((H @ c) & 1, s), f"{code.name}: H@c != s"
