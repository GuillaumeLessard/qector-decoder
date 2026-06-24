"""
qector_decoder_v3.result — Rich decode results, diagnostics and debug output.

The bare decoders return a correction vector.  Research and engineering use often
needs more: the correction in several encodings, the implied logical-observable
flips, the matching weight, timing, and which backend actually ran.  This module
wraps a single decode (or a backend) and produces a :class:`DecodeResult` with
all of that, plus a human-readable :func:`DecodeResult.explain` debug view and a
JSON export.

Example
-------
>>> from qector_decoder_v3 import codes
>>> from qector_decoder_v3.result import decode_with_diagnostics
>>> code = codes.repetition_code(9)
>>> s = code.syndrome(code.random_error(0.1))
>>> res = decode_with_diagnostics(code, s, kind="blossom")
>>> res.correction            # uint8 vector
>>> res.sparse_indices        # nonzero qubit indices
>>> res.bit_packed            # np.packbits view
>>> res.to_json()
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from typing import Any, Optional, Sequence

import numpy as np

__all__ = ["DecodeResult", "decode_with_diagnostics"]


@dataclass
class DecodeResult:
    """Result of decoding a single syndrome, with diagnostics and metadata."""

    correction: np.ndarray
    syndrome: np.ndarray
    n_qubits: int
    n_checks: int
    weight: Optional[float] = None
    logical_flips: Optional[np.ndarray] = None
    decode_seconds: Optional[float] = None
    backend: str = "cpu"
    fallback: bool = False
    fallback_reason: str = ""
    syndrome_valid: Optional[bool] = None
    metadata: dict = field(default_factory=dict)

    # -- alternative encodings --------------------------------------------
    @property
    def sparse_indices(self) -> np.ndarray:
        """Indices of the flipped qubits (sparse representation)."""
        return np.nonzero(self.correction)[0].astype(np.int64)

    @property
    def bit_packed(self) -> np.ndarray:
        """Correction packed 8 bits per byte (``np.packbits``)."""
        return np.packbits(self.correction.astype(np.uint8))

    @property
    def hamming_weight(self) -> int:
        """Number of flipped qubits in the correction."""
        return int(self.correction.sum())

    def as_uint8(self) -> np.ndarray:
        return self.correction.astype(np.uint8, copy=False)

    # -- validation --------------------------------------------------------
    def verify(self, H: np.ndarray) -> bool:
        """Check ``H @ correction == syndrome (mod 2)`` and cache the result."""
        ok = bool(np.array_equal((H @ self.as_uint8()) & 1, self.syndrome.astype(np.uint8)))
        self.syndrome_valid = ok
        return ok

    # -- serialisation -----------------------------------------------------
    def to_dict(self) -> dict:
        return {
            "n_qubits": self.n_qubits,
            "n_checks": self.n_checks,
            "hamming_weight": self.hamming_weight,
            "sparse_indices": self.sparse_indices.tolist(),
            "weight": self.weight,
            "logical_flips": None
            if self.logical_flips is None
            else self.logical_flips.astype(int).tolist(),
            "decode_seconds": self.decode_seconds,
            "backend": self.backend,
            "fallback": self.fallback,
            "fallback_reason": self.fallback_reason,
            "syndrome_valid": self.syndrome_valid,
            "metadata": _jsonable(self.metadata),
        }

    def to_json(self, indent: int = 2) -> str:
        return json.dumps(self.to_dict(), indent=indent)

    # -- debugging ---------------------------------------------------------
    def explain(self) -> str:
        """Human-readable explanation of the decode path."""
        lines = [
            "=== QECTOR decode diagnostics ===",
            f"backend            : {self.backend}",
            f"qubits / checks    : {self.n_qubits} / {self.n_checks}",
            f"syndrome weight    : {int(self.syndrome.sum())}",
            f"correction weight  : {self.hamming_weight}",
            f"flipped qubits     : {self.sparse_indices.tolist()[:32]}"
            + (" ..." if self.hamming_weight > 32 else ""),
        ]
        if self.weight is not None:
            lines.append(f"matching weight    : {self.weight:.4f}")
        if self.logical_flips is not None:
            lines.append(f"logical flips      : {self.logical_flips.astype(int).tolist()}")
        if self.decode_seconds is not None:
            lines.append(f"decode time        : {self.decode_seconds * 1e6:.2f} us")
        lines.append(f"syndrome valid     : {self.syndrome_valid}")
        if self.fallback:
            lines.append(f"FALLBACK           : {self.fallback_reason}")
        return "\n".join(lines)

    def __repr__(self) -> str:  # pragma: no cover - cosmetic
        return (
            f"DecodeResult(weight={self.hamming_weight}, backend={self.backend!r}, "
            f"valid={self.syndrome_valid})"
        )


def _jsonable(obj: Any) -> Any:
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        return float(obj)
    if isinstance(obj, dict):
        return {k: _jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_jsonable(v) for v in obj]
    return obj


def decode_with_diagnostics(
    code: Any,
    syndrome: Sequence[int],
    kind: str = "blossom",
    decoder: Any = None,
    logicals: Optional[np.ndarray] = None,
) -> DecodeResult:
    """Decode one syndrome and return a fully-populated :class:`DecodeResult`.

    Parameters
    ----------
    code : qector_decoder_v3.codes.Code
        The code being decoded (provides ``H`` and, optionally, logicals).
    syndrome : array-like
        Binary syndrome of length ``code.n_checks``.
    kind : str
        Decoder family to instantiate when ``decoder`` is not given.
    decoder : object, optional
        A pre-built decoder exposing ``decode(syndrome)`` — reuse this for the
        hot path instead of rebuilding.
    logicals : numpy.ndarray, optional
        ``(n_logicals, n_qubits)`` observable matrix; defaults to
        ``code.logicals_matrix()``.
    """
    from . import (
        BlossomDecoder,
        BPOSDDecoder,
        FastUnionFindDecoder,
        SparseBlossomDecoder,
        UnionFindDecoder,
    )

    s = np.asarray(syndrome, dtype=np.uint8).reshape(-1)
    c2q, nq = code.check_to_qubits, code.n_qubits

    if decoder is None:
        builders = {
            "union_find": lambda: UnionFindDecoder(c2q, nq),
            "fast_union_find": lambda: FastUnionFindDecoder(c2q, nq),
            "blossom": lambda: BlossomDecoder(c2q, nq),
            "sparse_blossom": lambda: SparseBlossomDecoder(c2q, nq),
            "bp_osd": lambda: BPOSDDecoder(c2q, nq, 0.05),
        }
        if kind not in builders:
            raise ValueError(f"unknown decoder kind: {kind!r}")
        decoder = builders[kind]()
        backend = kind
    else:
        backend = type(decoder).__name__

    t0 = time.perf_counter()
    correction = np.asarray(decoder.decode(s), dtype=np.uint8).reshape(-1)
    dt = time.perf_counter() - t0

    H = code.parity_check_matrix()
    weight = None
    if code.qubit_weights is not None and code.qubit_weights.shape[0] == nq:
        weight = float(np.dot(code.qubit_weights, correction))
    else:
        weight = float(correction.sum())

    L = logicals if logicals is not None else code.logicals_matrix()
    logical_flips = None
    if L is not None and L.shape[1] == nq:
        logical_flips = (L @ correction) & 1

    res = DecodeResult(
        correction=correction,
        syndrome=s,
        n_qubits=nq,
        n_checks=code.n_checks,
        weight=weight,
        logical_flips=logical_flips,
        decode_seconds=dt,
        backend=backend,
    )
    res.verify(H)
    return res
