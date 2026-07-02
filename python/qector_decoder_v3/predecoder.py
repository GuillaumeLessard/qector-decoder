"""
qector_decoder_v3.predecoder — a fast, faithful matching predecoder.

A predecoder cheaply resolves the "easy" part of a syndrome before handing the
hard residual to a full decoder, cutting latency on sparse syndromes. This
implements the standard *local matching* predecoder: greedily commit graph edges
whose two endpoints are both lit detectors (an adjacent defect pair), which clears
those two defects exactly with no side effects, then decode whatever remains.

The result is always syndrome-faithful: committed edges toggle exactly their two
checks, and the residual is decoded by an exact MWPM decoder, so
``H @ correction == syndrome`` holds. Pair it with ``UnionFindDecoder`` (speed) or
``BlossomDecoder`` (exact residual).

Example
-------
>>> from qector_decoder_v3 import codes
>>> from qector_decoder_v3.predecoder import PredecodedDecoder
>>> code = codes.rotated_surface_code(7)
>>> dec = PredecodedDecoder(code.check_to_qubits, code.n_qubits, backend="blossom")
>>> correction = dec.decode(syndrome)
"""

from __future__ import annotations

from typing import TYPE_CHECKING, List, Union

import numpy as np

if TYPE_CHECKING:
    from . import BlossomDecoder, SparseBlossomDecoder, UnionFindDecoder

__all__ = ["PredecodedDecoder", "quantize_weights"]


def quantize_weights(weights, num_levels: int = 1024):
    """Quantise float matching weights to integers (PyMatching-style).

    Integer weights make comparisons exact and ties deterministic. Returns an
    int array scaled so the largest finite weight maps to ``num_levels``.
    """
    w = np.asarray(weights, dtype=np.float64)
    finite = w[np.isfinite(w)]
    if finite.size == 0:
        return np.full(w.shape, num_levels, dtype=np.int64)
    hi = float(np.max(np.abs(finite))) or 1.0
    scaled = np.where(np.isfinite(w), w / hi * num_levels, float(num_levels))
    q = np.round(scaled).astype(np.int64)
    q[q < 1] = 1
    return q


class PredecodedDecoder:
    """Local-matching predecoder + exact/fast residual decoder."""

    def __init__(self, check_to_qubits, n_qubits=None, backend: str = "blossom"):
        from . import BlossomDecoder, UnionFindDecoder, SparseBlossomDecoder

        if not check_to_qubits:
            raise ValueError("check_to_qubits must be non-empty")
        self._c2q: List[List[int]] = [[int(q) for q in c] for c in check_to_qubits]
        self.n_checks = len(self._c2q)
        if n_qubits is None:
            n_qubits = max((max(c) for c in self._c2q if c), default=-1) + 1
        self.n_qubits = int(n_qubits)

        # qubit -> the checks it touches (graph edges have exactly 2)
        self._qubit_checks: List[List[int]] = [[] for _ in range(self.n_qubits)]
        for ci, qs in enumerate(self._c2q):
            for q in qs:
                self._qubit_checks[q].append(ci)
        # for each pair of checks, an edge qubit connecting them (graph edges only)
        self._pair_edge: dict = {}
        for q, chk in enumerate(self._qubit_checks):
            if len(chk) == 2:
                key = (min(chk), max(chk))
                self._pair_edge.setdefault(key, q)

        _valid_backends = ("blossom", "union_find", "sparse_blossom")
        if backend not in _valid_backends:
            raise ValueError(f"backend must be one of {list(_valid_backends)}")
        self.backend = backend
        residual: Union[BlossomDecoder, UnionFindDecoder, SparseBlossomDecoder]
        if backend == "blossom":
            residual = BlossomDecoder(self._c2q, self.n_qubits)
        elif backend == "union_find":
            residual = UnionFindDecoder(self._c2q, self.n_qubits)
        else:
            residual = SparseBlossomDecoder(self._c2q, self.n_qubits)
        self._residual: Union[BlossomDecoder, UnionFindDecoder, SparseBlossomDecoder] = residual
        self.last_predecoded = 0  # number of defects resolved by the predecoder

    def _predecode(self, syndrome: np.ndarray):
        """Greedily commit adjacent defect-pair edges. Returns (committed, residual_syndrome)."""
        s = syndrome.copy()
        committed = np.zeros(self.n_qubits, dtype=np.uint8)
        matched = np.zeros(self.n_checks, dtype=bool)
        defects = np.nonzero(s)[0]
        resolved = 0
        for c in defects:
            if matched[c] or not s[c]:
                continue
            # find an adjacent unmatched defect via a single graph edge
            for q in self._c2q[c]:
                chk = self._qubit_checks[q]
                if len(chk) != 2:
                    continue
                other = chk[0] if chk[1] == c else chk[1]
                if s[other] and not matched[other] and other != c:
                    committed[q] ^= 1
                    s[c] = 0
                    s[other] = 0
                    matched[c] = matched[other] = True
                    resolved += 2
                    break
        self.last_predecoded = resolved
        return committed, s

    def decode(self, syndrome) -> np.ndarray:
        s = np.asarray(syndrome, dtype=np.uint8).reshape(-1)
        committed, residual = self._predecode(s)
        if not residual.any():
            return committed
        res_corr = np.asarray(self._residual.decode(residual), dtype=np.uint8).reshape(-1)
        return (committed ^ res_corr).astype(np.uint8)

    def batch_decode(self, syndromes) -> np.ndarray:
        arr = np.asarray(syndromes, dtype=np.uint8)
        if arr.ndim != 2:
            raise ValueError(f"syndromes must be 2D, got {arr.shape}")
        return np.stack([self.decode(arr[i]) for i in range(arr.shape[0])]).astype(np.uint8)
