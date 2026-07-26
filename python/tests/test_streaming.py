"""Tests for the higher-level streaming orchestration layer
(:mod:`qector_decoder_v3.streaming`).

These exercise the *Python* :class:`StreamingSession` / :func:`sliding_window_decode`
orchestration — distinct from the compiled-core ``StreamingDecoder`` /
``SlidingWindowDecoder`` covered elsewhere.  The load-bearing assertions are:

* **validity** — every committed correction satisfies ``H @ c == s (mod 2)``, for
  any inner decoder (this bar is never weakened);
* **full-decode equivalence** — for a *stateless* inner decoder (Union-Find family,
  exact ``BlossomDecoder``) windowed streaming reproduces a single full per-round
  decode bit-for-bit, for any window size and for single-shot and batched streams;
* **honest stateful caveat** — for a region-growing decoder
  (``SparseBlossomDecoder``) the committed corrections stay valid but need not be
  window-invariant; we assert validity only, matching the module docstring;
* **real telemetry** — measured round/window counts and per-window wall times are
  self-consistent (no fabricated latency).

Equivalence assertions pass an *explicit* stateless decoder so they do not depend
on the (sibling-owned, optional) ``routing`` module that selects the default.
"""

from __future__ import annotations

import numpy as np
import pytest
import qector_decoder_v3 as qd
from qector_decoder_v3 import codes
from qector_decoder_v3.streaming import (
    StreamingResult,
    StreamingSession,
    StreamingTelemetry,
    sliding_window_decode,
)


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------
def _reachable_stream(code, n_rounds: int, p: float, seed: int) -> np.ndarray:
    """A (n_rounds, n_checks) stream of always-valid (reachable) syndromes."""
    H = code.parity_check_matrix()
    rng = np.random.default_rng(seed)
    rows = []
    for _ in range(n_rounds):
        e = (rng.random(code.n_qubits) < p).astype(np.uint8)
        rows.append(((H @ e) & 1).astype(np.uint8))
    return np.stack(rows)


def _full_decode_reference(rounds: np.ndarray, decoder) -> np.ndarray:
    """Single full decode: decode each round independently, stack the corrections."""
    return np.stack([np.asarray(decoder.decode(s), np.uint8) for s in rounds])


# matching-graph codes where H @ c == s is well-defined for these decoders
_CODES = [
    codes.repetition_code(7),
    codes.ring_code(7),
    codes.rotated_surface_code(5),
]

# decoders that are *stateless* (pure function of the syndrome) — the regime in
# which windowed streaming equals a single full decode bit-for-bit.
_STATELESS = ["FastUnionFindDecoder", "BlossomDecoder", "UnionFindDecoder"]


def _fresh(name: str, code):
    return getattr(qd, name)(code.check_to_qubits, code.n_qubits)


# ---------------------------------------------------------------------------
# does not shadow the Rust primitives
# ---------------------------------------------------------------------------
def test_does_not_shadow_rust_streaming_primitives():
    """The orchestration names are new; the Rust core classes are untouched."""
    assert StreamingSession is not qd.StreamingDecoder
    assert StreamingSession is not qd.SlidingWindowDecoder
    assert sliding_window_decode is not qd.SlidingWindowDecoder
    assert qd.StreamingDecoder.__module__ == "qector_decoder_v3"
    assert qd.SlidingWindowDecoder.__module__ == "qector_decoder_v3"


# ---------------------------------------------------------------------------
# validity (H @ c == s) on every committed round
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("code", _CODES, ids=lambda c: c.name)
def test_session_committed_rounds_are_syndrome_faithful(code):
    H = code.parity_check_matrix()
    rounds = _reachable_stream(code, n_rounds=40, p=0.12, seed=1)

    sess = StreamingSession(code, window_size=5)
    assert sess.n_checks == code.n_checks
    assert sess.n_qubits == code.n_qubits

    for r in rounds:
        sess.push_round(r)
    sess.flush()

    corr = sess.committed_corrections()
    syn = sess.committed_syndromes()
    assert corr.shape == (40, code.n_qubits)
    assert syn.shape == (40, code.n_checks)
    recon = (corr @ H.T) & 1
    assert np.array_equal(recon, syn), f"{code.name}: H @ c != s on a committed round"
    assert syn.any()  # some non-trivial syndromes were exercised


@pytest.mark.parametrize("code", _CODES, ids=lambda c: c.name)
def test_sliding_window_decode_is_valid(code):
    H = code.parity_check_matrix()
    rounds = _reachable_stream(code, n_rounds=30, p=0.15, seed=2)
    res = sliding_window_decode(rounds, code=code, window_size=4)
    assert isinstance(res, StreamingResult)
    assert res.corrections.shape == (30, code.n_qubits)
    assert res.is_valid(H)


def test_sparse_blossom_stays_valid_even_if_not_window_invariant():
    """Honest stateful caveat: region-growing decoder is valid but not invariant.

    ``SparseBlossomDecoder`` is order/grouping dependent, so different window sizes
    may yield different *but equally valid* corrections.  We assert validity for
    each, and document (not require) that they can differ.
    """
    code = codes.rotated_surface_code(5)
    H = code.parity_check_matrix()
    rounds = _reachable_stream(code, n_rounds=24, p=0.15, seed=99)

    r_small = sliding_window_decode(rounds, code=code, window_size=2, decoder=_fresh("SparseBlossomDecoder", code))
    r_big = sliding_window_decode(rounds, code=code, window_size=999, decoder=_fresh("SparseBlossomDecoder", code))
    assert r_small.is_valid(H)
    assert r_big.is_valid(H)
    # both valid; equality is NOT asserted (stateful decoder), matching the docs.


# ---------------------------------------------------------------------------
# full-decode equivalence (the core theorem, stateless decoders)
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("code", _CODES, ids=lambda c: c.name)
@pytest.mark.parametrize("decoder_name", _STATELESS)
@pytest.mark.parametrize("window_size", [1, 3, 8, 1000])
def test_windowed_equals_single_full_decode(code, decoder_name, window_size):
    """Windowed streaming reproduces a single full per-round decode, bit-for-bit."""
    rounds = _reachable_stream(code, n_rounds=24, p=0.12, seed=3)

    # reference: decode every round once with a fresh (stateless) decoder
    full = _full_decode_reference(rounds, _fresh(decoder_name, code))

    # streamed via sliding_window_decode (uses batch_decode internally)
    res = sliding_window_decode(rounds, code=code, window_size=window_size, decoder=_fresh(decoder_name, code))
    assert np.array_equal(res.corrections, full), (
        f"{code.name}/{decoder_name} w={window_size}: sliding_window_decode != full"
    )

    # streamed via the incremental commit-buffer session (single decode per round)
    sess = StreamingSession(code, window_size=window_size, decoder=_fresh(decoder_name, code))
    streamed = sess.run(rounds).corrections
    assert np.array_equal(streamed, full), f"{code.name}/{decoder_name} w={window_size}: StreamingSession.run != full"


def test_session_and_function_paths_agree_for_stateless_decoder():
    """The two orchestration paths produce identical corrections (stateless decoder)."""
    code = codes.rotated_surface_code(5)
    rounds = _reachable_stream(code, n_rounds=20, p=0.13, seed=33)

    sess = StreamingSession(code, window_size=4, decoder=_fresh("FastUnionFindDecoder", code))
    sess_corr = sess.run(rounds).corrections
    func_corr = sliding_window_decode(
        rounds, code=code, window_size=1000, decoder=_fresh("FastUnionFindDecoder", code)
    ).corrections
    assert np.array_equal(sess_corr, func_corr)


def test_commit_latency_and_ordering():
    """Rounds commit in arrival order, lagging by exactly ``window_size``."""
    code = codes.repetition_code(7)
    rounds = _reachable_stream(code, n_rounds=10, p=0.2, seed=4)
    sess = StreamingSession(code, window_size=3, decoder=_fresh("FastUnionFindDecoder", code))

    committed_so_far = 0
    for i, r in enumerate(rounds):
        released = sess.push_round(r)
        if i < 3:
            assert released == []  # nothing commits until the window fills
        else:
            assert len(released) == 1
        committed_so_far += len(released)
        assert sess.committed_count == committed_so_far
        assert sess.pending == min(i + 1, 3)

    assert sess.committed_count == 7  # 10 rounds, window 3 -> 7 committed, 3 pending
    assert sess.pending == 3
    tail = sess.flush()
    assert len(tail) == 3
    assert sess.committed_count == 10
    ref = _full_decode_reference(rounds, _fresh("FastUnionFindDecoder", code))
    assert np.array_equal(sess.committed_corrections(), ref)


# ---------------------------------------------------------------------------
# batched streaming over many shots
# ---------------------------------------------------------------------------
def test_batched_streaming_shapes_and_validity():
    code = codes.rotated_surface_code(5)
    H = code.parity_check_matrix()
    n_shots, n_rounds = 6, 12
    rng = np.random.default_rng(5)
    batch = np.stack(
        [_reachable_stream(code, n_rounds, p=0.1, seed=int(rng.integers(1 << 30))) for _ in range(n_shots)]
    )
    assert batch.shape == (n_shots, n_rounds, code.n_checks)

    res = sliding_window_decode(batch, code=code, window_size=4)
    assert res.corrections.shape == (n_shots, n_rounds, code.n_qubits)
    assert res.is_valid(H)
    assert res.telemetry.rounds == n_shots * n_rounds
    assert res.telemetry.committed == n_shots * n_rounds


@pytest.mark.parametrize("decoder_name", _STATELESS)
def test_batched_equals_per_shot_full_decode(decoder_name):
    """Each shot/round in a batch matches its independent full decode (stateless)."""
    code = codes.repetition_code(9)
    n_shots, n_rounds = 5, 15
    rng = np.random.default_rng(6)
    batch = np.stack(
        [_reachable_stream(code, n_rounds, p=0.15, seed=int(rng.integers(1 << 30))) for _ in range(n_shots)]
    )
    res = sliding_window_decode(batch, code=code, window_size=4, decoder=_fresh(decoder_name, code))

    ref_dec = _fresh(decoder_name, code)
    for s in range(n_shots):
        expected = _full_decode_reference(batch[s], ref_dec)
        assert np.array_equal(res.corrections[s], expected), f"shot {s} mismatch"


# ---------------------------------------------------------------------------
# telemetry is real and self-consistent
# ---------------------------------------------------------------------------
def test_telemetry_counts_are_real():
    code = codes.rotated_surface_code(5)
    rounds = _reachable_stream(code, n_rounds=20, p=0.1, seed=7)

    res = sliding_window_decode(rounds, code=code, window_size=4)
    tel = res.telemetry
    assert isinstance(tel, StreamingTelemetry)
    assert tel.rounds == 20
    assert tel.windows == 5  # 20 rounds / window 4 -> 5 windows
    assert len(tel.per_window_seconds) == tel.windows
    assert all(t >= 0.0 for t in tel.per_window_seconds)
    assert tel.decode_seconds == pytest.approx(sum(tel.per_window_seconds))
    assert tel.mean_window_seconds == pytest.approx(tel.decode_seconds / tel.windows)
    assert "active_module" in tel.backend
    for key in ("h2d", "d2h", "gpu_calls", "fallbacks"):
        assert key in tel.gpu


def test_session_telemetry_one_window_per_round():
    code = codes.repetition_code(7)
    rounds = _reachable_stream(code, n_rounds=12, p=0.15, seed=8)
    sess = StreamingSession(code, window_size=4, decoder=_fresh("FastUnionFindDecoder", code))
    res = sess.run(rounds)
    # incremental session decodes each round once -> one window per round
    assert res.telemetry.rounds == 12
    assert res.telemetry.windows == 12
    assert res.telemetry.committed == 12
    assert len(res.telemetry.per_window_seconds) == 12


def test_reset_clears_state_and_telemetry():
    code = codes.repetition_code(7)
    rounds = _reachable_stream(code, n_rounds=6, p=0.2, seed=20)
    sess = StreamingSession(code, window_size=2, decoder=_fresh("FastUnionFindDecoder", code))
    sess.run(rounds)
    assert sess.committed_count == 6
    sess.reset()
    assert sess.committed_count == 0
    assert sess.pending == 0
    assert sess.telemetry.rounds == 0
    assert sess.telemetry.windows == 0


# ---------------------------------------------------------------------------
# decoder selection / routing fallback
# ---------------------------------------------------------------------------
def test_default_decoder_is_valid_and_known_type():
    """The default decoder (routing-chosen or FastUnionFind fallback) works.

    The ``routing`` module is optional and sibling-owned, so we do not assert a
    specific class; we require the default to be one of the package's decoder
    classes and to produce valid corrections.
    """
    code = codes.rotated_surface_code(5)
    H = code.parity_check_matrix()
    sess = StreamingSession(code, window_size=3)
    assert hasattr(sess.decoder, "decode")
    # the chosen decoder is a real exported decoder class instance
    assert type(sess.decoder).__name__ in {
        "FastUnionFindDecoder",
        "UnionFindDecoder",
        "BlossomDecoder",
        "SparseBlossomDecoder",
        "BPOSDDecoder",
        "BpOsdDecoder",
        "AutoDecoder",
        "LookupTableDecoder",
    }
    res = sess.run(_reachable_stream(code, 12, 0.12, seed=21))
    assert res.is_valid(H)


def test_custom_inner_decoder_is_honoured():
    code = codes.rotated_surface_code(5)
    H = code.parity_check_matrix()
    rounds = _reachable_stream(code, n_rounds=16, p=0.1, seed=9)

    inner = qd.BlossomDecoder(code.check_to_qubits, code.n_qubits)
    res = sliding_window_decode(rounds, code=code, window_size=4, decoder=inner)
    assert res.is_valid(H)

    sess = StreamingSession(code, window_size=4, decoder=inner)
    assert sess.decoder is inner
    assert sess.run(rounds).is_valid(H)


def test_bad_decoder_rejected():
    code = codes.repetition_code(7)

    class NotADecoder:
        pass

    with pytest.raises(TypeError):
        StreamingSession(code, decoder=NotADecoder())


def test_raw_check_to_qubits_accepted():
    """The function accepts a raw check_to_qubits + n_qubits instead of a Code."""
    code = codes.repetition_code(7)
    H = code.parity_check_matrix()
    rounds = _reachable_stream(code, n_rounds=10, p=0.15, seed=22)
    res = sliding_window_decode(rounds, check_to_qubits=code.check_to_qubits, n_qubits=code.n_qubits, window_size=3)
    assert res.is_valid(H)


# ---------------------------------------------------------------------------
# logical flips
# ---------------------------------------------------------------------------
def test_logical_flips_match_manual_parity():
    code = codes.repetition_code(7)  # ships a logical observable [[0]]
    assert code.logicals is not None
    rounds = _reachable_stream(code, n_rounds=18, p=0.2, seed=10)

    res = sliding_window_decode(rounds, code=code, window_size=5, decoder=_fresh("FastUnionFindDecoder", code))
    assert res.logical_flips is not None
    assert res.logical_flips.shape == (18, 1)

    L = code.logicals_matrix()
    manual = (res.corrections @ L.T) & 1
    assert np.array_equal(res.logical_flips, manual)


def test_no_logicals_means_none():
    code = codes.unrotated_surface_code(4)  # no logicals attached
    assert code.logicals is None
    rounds = _reachable_stream(code, n_rounds=10, p=0.1, seed=11)
    res = sliding_window_decode(rounds, code=code, window_size=3)
    assert res.logical_flips is None


# ---------------------------------------------------------------------------
# GPU-preference toggle must not change the (deterministic) result
# ---------------------------------------------------------------------------
def test_prefer_gpu_toggle_invariant_result():
    from qector_decoder_v3 import gpu_backend as gb

    code = codes.rotated_surface_code(5)
    H = code.parity_check_matrix()
    rounds = _reachable_stream(code, n_rounds=14, p=0.12, seed=12)

    saved = gb.get_prefer_gpu()
    try:
        gb.set_prefer_gpu(False)
        cpu = sliding_window_decode(
            rounds,
            code=code,
            window_size=4,
            prefer_gpu=False,
            decoder=_fresh("FastUnionFindDecoder", code),
        )
        assert cpu.is_valid(H, prefer_gpu=False)
        res_default = sliding_window_decode(
            rounds, code=code, window_size=4, decoder=_fresh("FastUnionFindDecoder", code)
        )
        # decoding is deterministic regardless of the GPU-preference policy
        assert np.array_equal(cpu.corrections, res_default.corrections)
    finally:
        gb.set_prefer_gpu(saved)


# ---------------------------------------------------------------------------
# input validation
# ---------------------------------------------------------------------------
def test_input_validation():
    code = codes.repetition_code(7)
    with pytest.raises(ValueError):
        StreamingSession(code, window_size=0)
    with pytest.raises(ValueError):
        sliding_window_decode(np.zeros((4, code.n_checks), np.uint8), code=code, window_size=0)
    with pytest.raises(ValueError):
        sliding_window_decode(np.zeros((4, 4, code.n_checks + 3), np.uint8), code=code)
    with pytest.raises(ValueError):
        sliding_window_decode(np.zeros((4, code.n_checks), np.uint8))  # no code/checks
    sess = StreamingSession(code, window_size=2, decoder=_fresh("FastUnionFindDecoder", code))
    with pytest.raises(ValueError):
        sess.push_round(np.zeros((code.n_checks + 1,), np.uint8))
