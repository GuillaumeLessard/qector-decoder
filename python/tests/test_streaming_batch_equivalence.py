"""Streaming decoder: per-syndrome ``decode`` faithfulness and the equivalence
``flush()`` + ``update(s)`` == fresh ``decode(s)``.

``StreamingDecoder.update`` accumulates a sliding history, so it is *not*
faithful to a single round's syndrome once several rounds are buffered.  But:

* ``decode(s)`` is a stateless, syndrome-faithful single-shot decode, and
* on a freshly-constructed / just-flushed decoder, ``update(s)`` decodes against
  empty history and so reproduces ``decode(s)`` bit-for-bit.

Both facts are asserted here with the core invariant ``(H @ corr) & 1 == s``.
"""

import numpy as np
import pytest

import qector_decoder_v3 as qd
from qector_decoder_v3 import codes


def _reachable(H, nq, rng, p):
    e = (rng.random(nq) < p).astype(np.uint8)
    return ((H @ e) & 1).astype(np.uint8)


@pytest.mark.parametrize(
    "code",
    [codes.repetition_code(7), codes.ring_code(7)],
    ids=lambda c: c.name,
)
def test_streaming_decode_is_syndrome_faithful(code):
    H = code.parity_check_matrix()
    nq = code.n_qubits
    dec = qd.StreamingDecoder(code.check_to_qubits, nq, history_size=10)
    assert dec.n_qubits == nq
    assert dec.n_checks == code.n_checks

    rng = np.random.default_rng(12345)
    seen_nonzero = 0
    for _ in range(200):
        s = _reachable(H, nq, rng, 0.15)
        corr = np.asarray(dec.decode(s), np.uint8)
        assert corr.shape == (nq,)
        assert np.array_equal((H @ corr) & 1, s), f"{code.name}: H@c != s"
        if s.any():
            seen_nonzero += 1
    # the random syndromes are not all trivially zero
    assert seen_nonzero > 0


@pytest.mark.parametrize(
    "code",
    [codes.repetition_code(7), codes.ring_code(7)],
    ids=lambda c: c.name,
)
def test_flushed_update_equals_fresh_decode(code):
    """On flushed state, update(s) is bit-identical to a fresh decoder's decode(s)."""
    H = code.parity_check_matrix()
    nq = code.n_qubits
    dec = qd.StreamingDecoder(code.check_to_qubits, nq, history_size=8)

    rng = np.random.default_rng(999)
    matches = 0
    for _ in range(60):
        s = _reachable(H, nq, rng, 0.2)
        dec.flush()
        upd = np.asarray(dec.update(s), np.uint8)
        fresh = qd.StreamingDecoder(code.check_to_qubits, nq, history_size=8)
        ref = np.asarray(fresh.decode(s), np.uint8)
        assert np.array_equal((H @ ref) & 1, s)
        assert np.array_equal(upd, ref), f"{code.name}: flushed update != fresh decode for s={s}"
        matches += 1
    assert matches == 60


def test_decode_is_stateless_under_prior_updates():
    """decode(s) is unaffected by previously-buffered update() history."""
    code = codes.repetition_code(7)
    H = code.parity_check_matrix()
    nq = code.n_qubits
    dirty = qd.StreamingDecoder(code.check_to_qubits, nq, history_size=10)
    clean = qd.StreamingDecoder(code.check_to_qubits, nq, history_size=10)

    rng = np.random.default_rng(7)
    # pollute the dirty decoder's history
    for _ in range(6):
        dirty.update(_reachable(H, nq, rng, 0.2))

    for _ in range(40):
        s = _reachable(H, nq, rng, 0.2)
        cd = np.asarray(dirty.decode(s), np.uint8)
        cc = np.asarray(clean.decode(s), np.uint8)
        assert np.array_equal((H @ cd) & 1, s)
        assert np.array_equal(cd, cc), "decode() leaked state from update() history"
