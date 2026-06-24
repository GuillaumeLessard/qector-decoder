import numpy as np
import qector_decoder_v3 as qd


def test_sliding_window_decoder():
    """Sliding window with decay factor."""
    c2q = [[0, 1], [1, 2], [2, 3], [3, 0]]
    dec = qd.SlidingWindowDecoder(c2q, window_size=3, decay_factor=0.5)
    assert dec.window_size == 3
    assert dec.decay_factor == 0.5

    for i in range(5):
        syndrome = np.array([1, 0, 0, 0], dtype=np.uint8)
        correction = dec.update(syndrome)
        assert len(correction) == dec.n_qubits
    assert dec.current_round == 5


def test_sliding_window_flush():
    """Sliding window flush resets state."""
    c2q = [[0, 1], [1, 2], [2, 3], [3, 0]]
    dec = qd.SlidingWindowDecoder(c2q, window_size=3, decay_factor=0.5)
    dec.update(np.array([1, 0, 0, 0], dtype=np.uint8))
    dec.flush()
    assert dec.current_round == 0


def test_sliding_window_decoder_direct():
    """Direct decode on sliding window decoder."""
    c2q = [[0, 1], [1, 2], [2, 3], [3, 0]]
    dec = qd.SlidingWindowDecoder(c2q, window_size=3, decay_factor=0.5)
    syndrome = np.array([1, 0, 1, 0], dtype=np.uint8)
    correction = dec.decode(syndrome)
    assert len(correction) == dec.n_qubits
