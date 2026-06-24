"""Streaming decoder flush resets all carried state.

After feeding several ``update(...)`` rounds, ``flush()`` must clear the buffered
history so that the next decode reproduces exactly what a freshly-constructed
decoder produces — i.e. no stale carryover survives a flush.  Asserted via
bit-identical corrections and the core invariant ``(H @ corr) & 1 == s``.
"""
import numpy as np

import qector_decoder_v3 as qd
from qector_decoder_v3 import codes


def _reachable(H, nq, rng, p=0.2):
    e = (rng.random(nq) < p).astype(np.uint8)
    return ((H @ e) & 1).astype(np.uint8)


def test_flush_clears_stale_state():
    code = codes.repetition_code(7)
    H = code.parity_check_matrix()
    nq = code.n_qubits
    dec = qd.StreamingDecoder(code.check_to_qubits, nq, history_size=6)

    rng = np.random.default_rng(2024)

    # warm the decoder up with several rounds of accumulated history
    for _ in range(10):
        dec.update(_reachable(H, nq, rng))

    # after flush, decoding any syndrome must match a brand-new decoder
    for _ in range(50):
        s = _reachable(H, nq, rng)
        dec.flush()
        c_after_flush = np.asarray(dec.decode(s), np.uint8)

        ref = qd.StreamingDecoder(code.check_to_qubits, nq, history_size=6)
        c_ref = np.asarray(ref.decode(s), np.uint8)

        assert np.array_equal((H @ c_after_flush) & 1, s)
        assert np.array_equal(c_after_flush, c_ref), (
            "flush() left stale state: post-flush decode != fresh decode"
        )


def test_flush_reproducibility_across_repeated_flush():
    """Flushing repeatedly between updates yields a reproducible correction."""
    code = codes.ring_code(8)
    H = code.parity_check_matrix()
    nq = code.n_qubits
    dec = qd.StreamingDecoder(code.check_to_qubits, nq, history_size=5)

    rng = np.random.default_rng(55)
    fixed = _reachable(H, nq, rng)

    results = []
    for _ in range(8):
        # interleave noise updates with flush + the same fixed syndrome
        dec.update(_reachable(H, nq, rng))
        dec.update(_reachable(H, nq, rng))
        dec.flush()
        c = np.asarray(dec.update(fixed), np.uint8)
        assert np.array_equal((H @ c) & 1, fixed)
        results.append(c)

    first = results[0]
    for c in results[1:]:
        assert np.array_equal(c, first), (
            "flush() not reproducible: same fixed syndrome gave differing corrections"
        )
