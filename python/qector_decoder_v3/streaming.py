"""
qector_decoder_v3.streaming - a higher-level Python streaming / multi-round
decode *orchestration* layer.

The compiled Rust core already exports two low-level streaming primitives,
``StreamingDecoder`` (circular history + OR accumulation) and
``SlidingWindowDecoder`` (exponential-decay weighted window).  This module does
**not** shadow or reuse those names.  Instead it provides a thin, dependency-free
Python orchestration built *on top of* the ordinary single-shot decoders:

* :class:`StreamingSession` - feed syndrome rounds in one at a time; each round is
  decoded immediately by an inner decoder and held in a sliding *commit buffer* of
  ``window_size`` rounds.  Once a round ages out of the window it is **committed**
  (released, in order).  :meth:`StreamingSession.flush` commits whatever remains.
* :func:`sliding_window_decode` - a batched convenience that decodes an entire
  multi-round stream (single shot *or* many shots) window-by-window and returns a
  :class:`StreamingResult`.

Decoding model (read this - it is an honest, deliberately simple model)
----------------------------------------------------------------------
Each *round* carries one spatial syndrome ``s_t`` over the code's ``n_checks``
checks, and is decoded **independently** by the inner decoder:
``c_t = inner.decode(s_t)``.  This is the phenomenological, perfect-measurement
("no time-like errors") regime: rounds do not share decoding state, so the
committed correction for a round is exactly the inner decoder's faithful decode
of that round's syndrome.  Two consequences this layer guarantees and tests:

* **Validity.**  Because every inner decoder in this package returns a GF(2)-exact
  correction (``H @ c == s (mod 2)``), every *committed* correction is valid by
  construction.  This layer never weakens that bar.
* **Window-invariance / full-decode equivalence (stateless decoders).**  For a
  *stateless* inner decoder - i.e. ``decode``/``batch_decode`` is a pure function of
  the syndrome, true of the package's Union-Find family and exact
  ``BlossomDecoder`` - the committed correction for round ``t`` is ``inner.decode(s_t)``
  regardless of ``window_size`` or whether the rounds were decoded one-at-a-time
  (:class:`StreamingSession`) or in batched windows (:func:`sliding_window_decode`).
  So streaming reproduces a single full per-round decode bit-for-bit; the window
  only controls *commit latency*, batching granularity, and telemetry.

  A *stateful* region-growing decoder (e.g. ``SparseBlossomDecoder``) may return a
  **different but equally valid** coset member depending on decode order / window
  grouping.  Validity (``H @ c == s``) still holds for every committed round; only
  bit-for-bit window-invariance is forfeit, and the layer never claims otherwise.

This layer does **not** claim to implement full space-time (circuit-level)
matching with time-like edges - that would be a different, heavier construction.

Telemetry is **real**: per-window wall time is measured with
:func:`time.perf_counter` around the inner-decoder call only, and the GPU helpers
from :mod:`qector_decoder_v3.gpu_backend` are used for the vectorised
logical-flip / validity math (CuPy when a device is usable, NumPy otherwise).

Example
-------
>>> import numpy as np
>>> from qector_decoder_v3 import codes
>>> from qector_decoder_v3.streaming import StreamingSession, sliding_window_decode
>>> code = codes.repetition_code(7)
>>> H = code.parity_check_matrix()
>>> rng = np.random.default_rng(0)
>>> rounds = np.stack([(H @ code.random_error(0.1, rng)) & 1 for _ in range(20)])
>>> res = sliding_window_decode(rounds, code=code, window_size=4)
>>> res.is_valid(H)
True
>>> res.telemetry.rounds
20
"""

from __future__ import annotations

import time
from collections import deque
from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from . import gpu_backend as _gb

__all__ = [
    "StreamingResult",
    "StreamingSession",
    "StreamingTelemetry",
    "sliding_window_decode",
]


# ---------------------------------------------------------------------------
# Inner-decoder resolution
# ---------------------------------------------------------------------------
def _resolve_default_decoder(check_to_qubits: list[list[int]], n_qubits: int) -> Any:
    """Pick the default inner decoder.

    Uses ``routing.recommend_decoder`` when a ``qector_decoder_v3.routing`` module
    exposing that callable is importable, otherwise falls back to the
    :class:`~qector_decoder_v3.FastUnionFindDecoder`.  ``recommend_decoder`` returns
    the *name* of a decoder class, which is instantiated as
    ``Cls(check_to_qubits, n_qubits)`` and sanity-checked with a trivial decode.
    Any failure (routing absent, odd signature, non-graphlike-only decoder, bad
    instance) degrades cleanly to ``FastUnionFindDecoder`` - the routing module is
    an optional convenience, never a hard dependency.
    """
    try:  # optional routing hook - guard the import + call + instantiation fully
        from . import routing as _routing  # type: ignore[attr-defined]

        rec = getattr(_routing, "recommend_decoder", None)
        if callable(rec):
            choice = rec(n_qubits=n_qubits)
            inst = _coerce_decoder(choice, check_to_qubits, n_qubits)
            if inst is not None and _decoder_works(inst, len(check_to_qubits), n_qubits):
                return inst
    except (
        ImportError,
        AttributeError,
        TypeError,
        ValueError,
        RuntimeError,
    ):  # pragma: no cover - routing API is sibling-owned / may churn
        pass

    from . import FastUnionFindDecoder

    return FastUnionFindDecoder(check_to_qubits, n_qubits)


def _decoder_works(inst: Any, n_checks: int, n_qubits: int) -> bool:
    """Return ``True`` iff ``inst`` decodes an all-zero syndrome to an ``(n_qubits,)`` vector."""
    try:
        out = np.asarray(inst.decode(np.zeros(n_checks, dtype=np.uint8)))
        return out.shape == (n_qubits,)
    except (AttributeError, TypeError, ValueError, RuntimeError):  # pragma: no cover - defensive
        return False


def _coerce_decoder(choice: Any, check_to_qubits: list[list[int]], n_qubits: int) -> Any | None:
    """Turn a ``recommend_decoder`` return value into a decoder instance, or ``None``.

    Accepts an already-built instance (has ``.decode``), a class/factory callable
    (instantiated as ``choice(check_to_qubits, n_qubits)``), or a string naming a
    decoder class exported by the package.
    """
    if choice is None:
        return None
    if hasattr(choice, "decode"):
        return choice
    if isinstance(choice, str):
        import qector_decoder_v3 as _pkg_mod

        cls = getattr(_pkg_mod, choice, None)
        if cls is None:
            return None
        return cls(check_to_qubits, n_qubits)
    if callable(choice):
        return choice(check_to_qubits, n_qubits)
    return None


# ---------------------------------------------------------------------------
# Input normalisation
# ---------------------------------------------------------------------------
def _checks_and_qubits(code_or_checks: Any, n_qubits: int | None) -> tuple[list[list[int]], int, np.ndarray | None]:
    """Normalise the code argument to ``(check_to_qubits, n_qubits, logicals_matrix)``.

    Accepts a :class:`qector_decoder_v3.codes.Code`-like object (has
    ``check_to_qubits`` / ``n_qubits``) or a raw ``check_to_qubits`` list with an
    explicit ``n_qubits``.
    """
    logicals_mat: np.ndarray | None = None
    if hasattr(code_or_checks, "check_to_qubits") and hasattr(code_or_checks, "n_qubits"):
        c2q = [list(map(int, c)) for c in code_or_checks.check_to_qubits]
        nq = int(code_or_checks.n_qubits)
        getter = getattr(code_or_checks, "logicals_matrix", None)
        if callable(getter):
            lm = getter()
            if lm is not None:
                logicals_mat = np.asarray(lm, dtype=np.uint8)
        return c2q, nq, logicals_mat

    c2q = [list(map(int, c)) for c in code_or_checks]
    if n_qubits is None:
        inferred = 1 + max((max(c) for c in c2q if c), default=-1)
        nq = int(inferred)
    else:
        nq = int(n_qubits)
    return c2q, nq, logicals_mat


def _to_host_u8(a: Any) -> np.ndarray:
    """Return ``a`` as a contiguous host ``uint8`` array (CuPy → host via gpu_backend)."""
    host = _gb.asnumpy(a) if _gb.is_on_gpu(a) else np.asarray(a)
    return np.ascontiguousarray(host).astype(np.uint8, copy=False)


def _logicals_from(logicals: Any, n_qubits: int) -> np.ndarray | None:
    """Coerce a logicals spec to a ``(n_logicals, n_qubits)`` uint8 matrix or ``None``."""
    if logicals is None:
        return None
    arr = np.asarray(logicals)
    if arr.ndim == 1:  # list of qubit indices -> single observable
        L = np.zeros((1, n_qubits), dtype=np.uint8)
        for q in arr.tolist():
            L[0, int(q)] ^= 1
        return L
    if arr.dtype == object or (arr.ndim == 2 and arr.shape[1] != n_qubits):
        # list[list[int]] of qubit-index sets
        rows = list(logicals)
        L = np.zeros((len(rows), n_qubits), dtype=np.uint8)
        for i, qs in enumerate(rows):
            for q in qs:
                L[i, int(q)] ^= 1
        return L
    return arr.astype(np.uint8, copy=False)


# ---------------------------------------------------------------------------
# Telemetry & result containers
# ---------------------------------------------------------------------------
@dataclass
class StreamingTelemetry:
    """Real, measured counters for a streaming decode run.

    Attributes
    ----------
    rounds:
        Total syndrome rounds processed (across all shots).
    windows:
        Number of inner-decoder invocations (size-1 windows for
        :class:`StreamingSession`, one per temporal chunk for
        :func:`sliding_window_decode`).
    committed:
        Number of rounds released from the commit buffer.
    decode_seconds:
        Sum of per-window wall times (``time.perf_counter``), inner decode only.
    per_window_seconds:
        Per-window wall times, in order.
    gpu:
        GPU transfer/compute counter deltas observed during the run, taken from
        :data:`qector_decoder_v3.gpu_backend.TELEMETRY`.
    backend:
        One-glance GPU capability snapshot (``gpu_backend.get_backend().summary()``).
    """

    rounds: int = 0
    windows: int = 0
    committed: int = 0
    decode_seconds: float = 0.0
    per_window_seconds: list[float] = field(default_factory=list)
    gpu: dict[str, int] = field(default_factory=dict)
    backend: dict[str, Any] = field(default_factory=dict)
    rounds_valid: int | None = None

    @property
    def mean_window_seconds(self) -> float:
        """Mean inner-decode wall time per window (0.0 if no windows ran)."""
        return self.decode_seconds / self.windows if self.windows else 0.0

    def record_window(self, seconds: float, rounds: int = 1) -> None:
        """Append one window's measured wall time and round count."""
        self.windows += 1
        self.rounds += int(rounds)
        self.decode_seconds += float(seconds)
        self.per_window_seconds.append(float(seconds))

    def as_dict(self) -> dict[str, Any]:
        """JSON-friendly view for logs / diagnostics."""
        return {
            "rounds": self.rounds,
            "windows": self.windows,
            "committed": self.committed,
            "decode_seconds": self.decode_seconds,
            "mean_window_seconds": self.mean_window_seconds,
            "per_window_seconds": list(self.per_window_seconds),
            "gpu": dict(self.gpu),
            "backend": dict(self.backend),
            "rounds_valid": self.rounds_valid,
        }


@dataclass
class StreamingResult:
    """Outcome of a full streaming decode.

    Attributes
    ----------
    corrections:
        ``(n_rounds, n_qubits)`` for a single shot, or
        ``(n_shots, n_rounds, n_qubits)`` for a batch.  ``uint8``.
    syndromes:
        The syndrome rounds that were decoded, same leading shape as
        ``corrections`` but with ``n_checks`` trailing - used for validity checks.
    telemetry:
        The :class:`StreamingTelemetry` for the run.
    logical_flips:
        ``(... , n_logicals)`` parity of each correction against the logical
        observables, or ``None`` if no logicals were supplied.
    """

    corrections: np.ndarray
    syndromes: np.ndarray
    telemetry: StreamingTelemetry
    logical_flips: np.ndarray | None = None

    def is_valid(self, H: Any, *, prefer_gpu: bool = True) -> bool:
        """Return ``True`` iff every committed round satisfies ``H @ c == s (mod 2)``.

        The check is vectorised over all rounds (and shots) via
        :func:`qector_decoder_v3.gpu_backend.get_array_module`, so it runs on the
        GPU when one is usable and on NumPy otherwise.
        """
        H = np.asarray(H, dtype=np.uint8)
        corr = self.corrections.reshape(-1, self.corrections.shape[-1])
        syn = self.syndromes.reshape(-1, self.syndromes.shape[-1])
        if corr.shape[0] == 0:
            self.telemetry.rounds_valid = 0
            return True
        xp = _gb.get_array_module(prefer_gpu=prefer_gpu)
        Hx = xp.asarray(H)
        cx = xp.asarray(corr)
        sx = xp.asarray(syn)
        recon = (cx @ Hx.T) & 1
        valid_mask = (recon == sx).all(axis=1)
        ok = bool(valid_mask.all())
        self.telemetry.rounds_valid = int(valid_mask.sum())
        if xp is not np:  # pragma: no cover - GPU-only path
            _gb.note_gpu_call()
        return ok


# ---------------------------------------------------------------------------
# StreamingSession - incremental window + commit
# ---------------------------------------------------------------------------
class StreamingSession:
    """Incremental multi-round decode with a sliding commit buffer.

    Feed one syndrome round at a time with :meth:`push_round`.  Each round is
    decoded immediately by the inner decoder and buffered; when more than
    ``window_size`` rounds are buffered the oldest is **committed** (released, in
    arrival order).  Call :meth:`flush` at end-of-stream to commit the remainder.

    Parameters
    ----------
    code_or_checks:
        A :class:`qector_decoder_v3.codes.Code`-like object, or a raw
        ``check_to_qubits`` list (then pass ``n_qubits``).
    n_qubits:
        Number of data qubits; required only when passing a raw check list and it
        cannot be inferred.
    window_size:
        Commit latency in rounds (``>= 1``).  A round commits once
        ``window_size`` newer rounds have arrived; smaller = lower latency.
    decoder:
        Inner decoder instance (anything with ``.decode``).  Defaults to
        ``routing.recommend_decoder(...)`` if available, else ``FastUnionFindDecoder``.
    logicals:
        Optional logical observables - a ``(n_logicals, n_qubits)`` matrix or a
        list of qubit-index sets - used to compute logical flips on committed rounds.
    prefer_gpu:
        If not ``None``, sets the global GPU preference for the duration of the
        session's vectorised math (logical flips / validity).  Detection still
        gates actual GPU use; NumPy is used whenever no device is available.
    """

    def __init__(
        self,
        code_or_checks: Any,
        n_qubits: int | None = None,
        *,
        window_size: int = 8,
        decoder: Any | None = None,
        logicals: Any = None,
        prefer_gpu: bool | None = None,
    ) -> None:
        if int(window_size) < 1:
            raise ValueError("window_size must be >= 1")
        c2q, nq, code_logicals = _checks_and_qubits(code_or_checks, n_qubits)
        self._check_to_qubits = c2q
        self._n_qubits = nq
        self._n_checks = len(c2q)
        self._window_size = int(window_size)
        self._decoder = decoder if decoder is not None else _resolve_default_decoder(c2q, nq)
        if not hasattr(self._decoder, "decode"):
            raise TypeError("decoder must expose a .decode(syndrome) method")

        chosen_logicals = logicals if logicals is not None else code_logicals
        self._logicals = _logicals_from(chosen_logicals, nq)
        self._prefer_gpu = prefer_gpu

        # sliding buffer of (index, syndrome, correction) not yet committed
        self._buffer: deque[tuple[int, np.ndarray, np.ndarray]] = deque()
        self._next_index = 0
        self._committed_syn: list[np.ndarray] = []
        self._committed_corr: list[np.ndarray] = []

        self.telemetry = StreamingTelemetry(backend=_gb.get_backend().summary())
        self._gpu_start = dict(_gb.TELEMETRY)

    # -- introspection -----------------------------------------------------
    @property
    def n_qubits(self) -> int:
        return self._n_qubits

    @property
    def n_checks(self) -> int:
        return self._n_checks

    @property
    def window_size(self) -> int:
        return self._window_size

    @property
    def decoder(self) -> Any:
        return self._decoder

    @property
    def pending(self) -> int:
        """Number of decoded-but-not-yet-committed rounds in the window."""
        return len(self._buffer)

    @property
    def committed_count(self) -> int:
        return len(self._committed_corr)

    # -- core --------------------------------------------------------------
    def _decode_one(self, syndrome: Any) -> np.ndarray:
        """Decode a single round, timing the inner call with ``perf_counter``."""
        s = _to_host_u8(syndrome)
        if s.shape != (self._n_checks,):
            raise ValueError(f"syndrome must have shape ({self._n_checks},), got {s.shape}")
        t0 = time.perf_counter()
        raw = self._decoder.decode(s)
        dt = time.perf_counter() - t0
        corr = _to_host_u8(raw)
        if corr.shape != (self._n_qubits,):
            raise ValueError(f"decoder returned shape {corr.shape}, expected ({self._n_qubits},)")
        self.telemetry.record_window(dt, rounds=1)
        return corr

    def push_round(self, syndrome: Any) -> list[np.ndarray]:
        """Decode one round and buffer it; return any newly-committed corrections.

        Returns a list (length 0 or 1) of corrections released from the back of the
        window this step.  Use :meth:`flush` to drain the rest at end-of-stream.
        """
        corr = self._decode_one(syndrome)
        s = _to_host_u8(syndrome)
        self._buffer.append((self._next_index, s, corr))
        self._next_index += 1
        released: list[np.ndarray] = []
        while len(self._buffer) > self._window_size:
            _, syn_old, corr_old = self._buffer.popleft()
            self._committed_syn.append(syn_old)
            self._committed_corr.append(corr_old)
            self.telemetry.committed += 1
            released.append(corr_old)
        return released

    def flush(self) -> list[np.ndarray]:
        """Commit every remaining buffered round, in order; return their corrections."""
        released: list[np.ndarray] = []
        while self._buffer:
            _, syn_old, corr_old = self._buffer.popleft()
            self._committed_syn.append(syn_old)
            self._committed_corr.append(corr_old)
            self.telemetry.committed += 1
            released.append(corr_old)
        return released

    def reset(self) -> None:
        """Clear all buffered and committed state and zero the telemetry."""
        self._buffer.clear()
        self._next_index = 0
        self._committed_syn.clear()
        self._committed_corr.clear()
        self.telemetry = StreamingTelemetry(backend=_gb.get_backend().summary())
        self._gpu_start = dict(_gb.TELEMETRY)

    # -- aggregate views ---------------------------------------------------
    def committed_corrections(self) -> np.ndarray:
        """All committed corrections, stacked ``(n_committed, n_qubits)`` uint8."""
        if not self._committed_corr:
            return np.zeros((0, self._n_qubits), dtype=np.uint8)
        return np.stack(self._committed_corr).astype(np.uint8)

    def committed_syndromes(self) -> np.ndarray:
        """All committed syndromes, stacked ``(n_committed, n_checks)`` uint8."""
        if not self._committed_syn:
            return np.zeros((0, self._n_checks), dtype=np.uint8)
        return np.stack(self._committed_syn).astype(np.uint8)

    def _finalise_gpu_telemetry(self) -> None:
        self.telemetry.gpu = {k: int(_gb.TELEMETRY.get(k, 0) - self._gpu_start.get(k, 0)) for k in _gb.TELEMETRY}

    def run(self, syndrome_rounds: Any) -> StreamingResult:
        """Stream a full ``(n_rounds, n_checks)`` syndrome array and flush.

        Pushes each row in turn, flushes, and returns a :class:`StreamingResult`
        whose ``corrections`` are the committed per-round corrections in order.
        """
        rounds = _to_host_u8(syndrome_rounds)
        if rounds.ndim != 2 or rounds.shape[1] != self._n_checks:
            raise ValueError(f"syndrome_rounds must have shape (n_rounds, {self._n_checks}), got {rounds.shape}")
        for r in rounds:
            self.push_round(r)
        self.flush()
        self._finalise_gpu_telemetry()
        corr = self.committed_corrections()
        syn = self.committed_syndromes()
        flips = _compute_logical_flips(corr, self._logicals, self._prefer_gpu)
        return StreamingResult(
            corrections=corr,
            syndromes=syn,
            telemetry=self.telemetry,
            logical_flips=flips,
        )


# ---------------------------------------------------------------------------
# Logical-flip helper (GPU-aware)
# ---------------------------------------------------------------------------
def _compute_logical_flips(
    corrections: np.ndarray, logicals: np.ndarray | None, prefer_gpu: bool | None
) -> np.ndarray | None:
    """Parity of each correction against the logical observables (or ``None``).

    ``corrections`` may be ``(..., n_qubits)``; result is ``(..., n_logicals)``.
    Uses :func:`gpu_backend.get_array_module` so the matmul runs on CuPy when a
    device is usable and NumPy otherwise.
    """
    if logicals is None or corrections.size == 0:
        return None
    flag = True if prefer_gpu is None else bool(prefer_gpu)
    xp = _gb.get_array_module(prefer_gpu=flag)
    Lx = xp.asarray(np.asarray(logicals, dtype=np.uint8))
    cx = xp.asarray(corrections.astype(np.uint8, copy=False))
    flips = (cx @ Lx.T) & 1
    if xp is not np:  # pragma: no cover - GPU-only path
        _gb.note_gpu_call()
        flips = _gb.asnumpy(flips)
    return np.asarray(flips, dtype=np.uint8)


# ---------------------------------------------------------------------------
# Batched, windowed convenience
# ---------------------------------------------------------------------------
def sliding_window_decode(
    syndrome_rounds: Any,
    code: Any = None,
    *,
    check_to_qubits: Sequence[Sequence[int]] | None = None,
    n_qubits: int | None = None,
    window_size: int = 8,
    decoder: Any | None = None,
    logicals: Any = None,
    prefer_gpu: bool | None = None,
) -> StreamingResult:
    """Decode a whole multi-round syndrome stream window-by-window.

    Parameters
    ----------
    syndrome_rounds:
        ``(n_rounds, n_checks)`` for a single shot, or ``(n_shots, n_rounds,
        n_checks)`` for a batch of shots.
    code:
        A :class:`qector_decoder_v3.codes.Code`-like object describing the code
        (provides ``check_to_qubits`` / ``n_qubits`` / optional logicals).  If
        omitted, pass ``check_to_qubits`` (+ ``n_qubits``).
    check_to_qubits, n_qubits:
        Raw code description, used when ``code`` is not given.
    window_size:
        Number of consecutive rounds decoded together as one window (``>= 1``).
        Each window is one inner-decoder (batch) invocation and one telemetry
        ``window``.  For a *stateless* inner decoder the per-round result is
        identical for any ``window_size`` (window-invariance); for a stateful
        region-growing decoder the choice may change which equally-valid coset
        member is returned.  Either way every result satisfies ``H @ c == s``.
    decoder:
        Inner decoder (anything with ``.decode``; ``.batch_decode`` is used when
        present for speed).  Defaults via ``routing.recommend_decoder`` else
        ``FastUnionFindDecoder``.
    logicals:
        Optional logical observables for computing per-round logical flips.
    prefer_gpu:
        Temporarily overrides the global GPU preference for the vectorised math.

    Returns
    -------
    StreamingResult
        With ``corrections`` shaped like the input (minus the check axis, plus a
        qubit axis) and real per-window timing telemetry.
    """
    if int(window_size) < 1:
        raise ValueError("window_size must be >= 1")

    if code is not None:
        c2q, nq, code_logicals = _checks_and_qubits(code, None)
    elif check_to_qubits is not None:
        c2q, nq, code_logicals = _checks_and_qubits(check_to_qubits, n_qubits)
    else:
        raise ValueError("provide either `code` or `check_to_qubits`")

    rounds = _to_host_u8(syndrome_rounds)
    single_shot = rounds.ndim == 2
    if single_shot:
        rounds = rounds[None, ...]  # (1, T, C)
    if rounds.ndim != 3 or rounds.shape[2] != len(c2q):
        raise ValueError(
            f"syndrome_rounds must be (T, {len(c2q)}) or (S, T, {len(c2q)}), got {np.asarray(syndrome_rounds).shape}"
        )

    n_shots, n_rounds, n_checks = rounds.shape
    inner = decoder if decoder is not None else _resolve_default_decoder(c2q, nq)
    if not hasattr(inner, "decode"):
        raise TypeError("decoder must expose a .decode(syndrome) method")
    has_batch = hasattr(inner, "batch_decode")

    telem = StreamingTelemetry(backend=_gb.get_backend().summary())
    gpu_start = dict(_gb.TELEMETRY)

    corrections = np.zeros((n_shots, n_rounds, nq), dtype=np.uint8)

    # Walk the time axis in windows; each window is one (batched) inner call.
    for start in range(0, n_rounds, int(window_size)):
        stop = min(start + int(window_size), n_rounds)
        chunk = rounds[:, start:stop, :]  # (S, w, C)
        w = stop - start
        flat = np.ascontiguousarray(chunk.reshape(n_shots * w, n_checks))
        t0 = time.perf_counter()
        if has_batch and flat.shape[0] > 1:
            out = _to_host_u8(inner.batch_decode(flat))
        else:
            out = np.stack([_to_host_u8(inner.decode(row)) for row in flat])
        dt = time.perf_counter() - t0
        out = out.reshape(n_shots, w, nq)
        corrections[:, start:stop, :] = out
        telem.record_window(dt, rounds=n_shots * w)

    telem.committed = n_shots * n_rounds
    telem.gpu = {k: int(_gb.TELEMETRY.get(k, 0) - gpu_start.get(k, 0)) for k in _gb.TELEMETRY}

    chosen_logicals = logicals if logicals is not None else code_logicals
    Lmat = _logicals_from(chosen_logicals, nq)
    flips = _compute_logical_flips(corrections, Lmat, prefer_gpu)

    syndromes = rounds.astype(np.uint8, copy=False)
    if single_shot:
        corrections = corrections[0]
        syndromes = syndromes[0]
        if flips is not None:
            flips = flips[0]

    return StreamingResult(
        corrections=corrections,
        syndromes=syndromes,
        telemetry=telem,
        logical_flips=flips,
    )
