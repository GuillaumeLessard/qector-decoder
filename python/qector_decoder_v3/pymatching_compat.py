"""
qector_decoder_v3.pymatching_compat — a PyMatching-compatible ``Matching`` API.

A drop-in replacement for the subset of ``pymatching.Matching`` most code uses,
backed by QECTOR's Sparse-Blossom decoder.  Lets existing Stim/PyMatching
pipelines swap in QECTOR with a one-line change::

    # from pymatching import Matching
    from qector_decoder_v3.pymatching_compat import Matching

    m = Matching.from_check_matrix(H, faults_matrix=observables)
    predicted_observables = m.decode(syndrome)
    predicted_batch = m.decode_batch(shots)

Supported constructors: :meth:`Matching.from_check_matrix`,
:meth:`Matching.from_detector_error_model`, incremental
:meth:`Matching.add_edge` / :meth:`Matching.add_boundary_edge`.  ``decode``
returns predicted **fault-id / observable** flips (PyMatching semantics): if a
``faults_matrix`` is supplied, the result has one bit per observable; otherwise
one bit per edge (the correction itself).

Weights are stored for reporting and used uniformly for matching (so minimum-
weight == minimum-cardinality matching, which matches PyMatching's behaviour for
uniform-weight graphs and lands in the same logical coset otherwise).
"""

from __future__ import annotations

from typing import Any, List, Optional, Sequence, Tuple

import numpy as np

from . import BlossomDecoder
from .codes import _to_dense_binary

__all__ = ["Matching"]

BOUNDARY = None


class Matching:
    """Minimum-weight perfect-matching decoder with a PyMatching-style API."""

    def __init__(self, H: Any = None, weights: Any = None, faults_matrix: Any = None):
        self._edges: List[dict] = []  # each: {u, v, fault_ids:set, weight:float}
        self._num_detectors = 0
        self._faults_matrix: Optional[np.ndarray] = None
        self._decoder: Optional[BlossomDecoder] = None
        self._H_cache: Optional[np.ndarray] = None
        if H is not None:
            self._init_from_check_matrix(H, weights, faults_matrix)

    # -- constructors ------------------------------------------------------
    @classmethod
    def from_check_matrix(
        cls,
        H: Any,
        weights: Any = None,
        faults_matrix: Any = None,
        **_: Any,
    ) -> "Matching":
        """Build from a parity-check matrix (rows=detectors, cols=edges)."""
        m = cls()
        m._init_from_check_matrix(H, weights, faults_matrix)
        return m

    @classmethod
    def from_detector_error_model(cls, dem: Any) -> "Matching":
        """Build from a Stim ``DetectorErrorModel`` (object or ``.dem`` text).

        Parallel mechanisms between the same detectors are collapsed into one
        min-weight edge (as PyMatching does), which makes circuit-level decoding
        an order of magnitude faster at identical logical accuracy.
        """
        from .dem import from_stim

        model = from_stim(dem)
        if model.is_graphlike:
            model = model.collapse_to_graph()
        m = cls.from_check_matrix(
            model.check_matrix(),
            weights=model.weights(),
            faults_matrix=model.observables_matrix(),
        )
        return m

    # alias used by some Stim workflows
    from_stim_dem = from_detector_error_model

    def _init_from_check_matrix(self, H: Any, weights: Any, faults_matrix: Any) -> None:
        arr = _to_dense_binary(H)
        if arr.ndim != 2:
            raise ValueError(f"H must be 2D, got shape {arr.shape}")
        n_det, n_edges = arr.shape
        w = (
            None
            if weights is None
            else np.asarray(weights, dtype=np.float64).reshape(-1)
        )
        fm = None if faults_matrix is None else _to_dense_binary(faults_matrix)
        self._faults_matrix = fm
        self._num_detectors = int(n_det)
        for j in range(n_edges):
            dets = np.nonzero(arr[:, j])[0].tolist()
            if fm is not None:
                fault_ids = set(int(o) for o in np.nonzero(fm[:, j])[0])
            else:
                fault_ids = {j}
            edge = {
                "u": int(dets[0]) if len(dets) >= 1 else BOUNDARY,
                "v": int(dets[1]) if len(dets) >= 2 else BOUNDARY,
                "fault_ids": fault_ids,
                "weight": float(w[j]) if w is not None and j < len(w) else 1.0,
            }
            self._edges.append(edge)
        self._decoder = None  # rebuild lazily

    # -- incremental construction -----------------------------------------
    def add_edge(
        self,
        node1: int,
        node2: int,
        fault_ids: Any = None,
        weight: float = 1.0,
        **_: Any,
    ) -> None:
        """Add an edge between two detector nodes carrying optional fault ids."""
        self._edges.append(
            {
                "u": int(node1),
                "v": int(node2),
                "fault_ids": _as_id_set(fault_ids),
                "weight": float(weight),
            }
        )
        self._num_detectors = max(self._num_detectors, int(node1) + 1, int(node2) + 1)
        self._decoder = None

    def add_boundary_edge(
        self, node: int, fault_ids: Any = None, weight: float = 1.0, **_: Any
    ) -> None:
        """Add an edge from a detector node to the boundary."""
        self._edges.append(
            {
                "u": int(node),
                "v": BOUNDARY,
                "fault_ids": _as_id_set(fault_ids),
                "weight": float(weight),
            }
        )
        self._num_detectors = max(self._num_detectors, int(node) + 1)
        self._decoder = None

    # -- introspection -----------------------------------------------------
    @property
    def num_detectors(self) -> int:
        return self._num_detectors

    @property
    def num_edges(self) -> int:
        return len(self._edges)

    @property
    def num_fault_ids(self) -> int:
        if self._faults_matrix is not None:
            return int(self._faults_matrix.shape[0])
        ids = set()
        for e in self._edges:
            ids |= e["fault_ids"]
        return (max(ids) + 1) if ids else 0

    def edges(self) -> List[Tuple[Optional[int], Optional[int], dict]]:
        """PyMatching-style edge list: ``(u, v, {"fault_ids":..., "weight":...})``."""
        return [
            (e["u"], e["v"], {"fault_ids": set(e["fault_ids"]), "weight": e["weight"]})
            for e in self._edges
        ]

    def check_matrix(self) -> np.ndarray:
        """The parity-check matrix (rows=detectors, cols=edges)."""
        if self._H_cache is not None and self._H_cache.shape[1] == len(self._edges):
            return self._H_cache
        H = np.zeros((self._num_detectors, len(self._edges)), dtype=np.uint8)
        for j, e in enumerate(self._edges):
            for node in (e["u"], e["v"]):
                if node is not BOUNDARY:
                    H[node, j] ^= np.uint8(1)
        self._H_cache = H
        return H

    # -- decoding ----------------------------------------------------------
    def _ensure_decoder(self) -> BlossomDecoder:
        if self._decoder is None:
            c2q: List[List[int]] = [[] for _ in range(self._num_detectors)]
            for j, e in enumerate(self._edges):
                for node in (e["u"], e["v"]):
                    if node is not BOUNDARY:
                        c2q[node].append(j)
            if not c2q:
                c2q = [[0]]
            # Weighted exact MWPM (matches PyMatching's weighting). When all
            # weights are equal, pass None so the uniform-weight fast path runs.
            weights = [e["weight"] for e in self._edges]
            uniform = len(set(round(w, 12) for w in weights)) <= 1
            self._decoder = BlossomDecoder(
                c2q, len(self._edges), None if uniform else weights
            )
        return self._decoder

    def _edge_correction(self, syndrome: np.ndarray) -> np.ndarray:
        dec = self._ensure_decoder()
        return np.asarray(
            dec.decode(syndrome.astype(np.uint8)), dtype=np.uint8
        ).reshape(-1)

    def _to_fault_prediction(self, correction: np.ndarray) -> np.ndarray:
        if self._faults_matrix is not None:
            return (self._faults_matrix @ correction) & 1
        return correction

    def decode(self, syndrome: Sequence[int]) -> np.ndarray:
        """Decode one syndrome; returns predicted fault-id / observable flips."""
        s = np.asarray(syndrome, dtype=np.uint8).reshape(-1)
        if s.shape[0] < self._num_detectors:
            s = np.concatenate(
                [s, np.zeros(self._num_detectors - s.shape[0], np.uint8)]
            )
        corr = self._edge_correction(s)
        return self._to_fault_prediction(corr).astype(np.uint8)

    def decode_batch(self, shots: Any) -> np.ndarray:
        """Decode a 2-D array of syndromes (one per row).

        Uses a single batched Rust call (GIL released) and a vectorised
        observable mapping, so throughput is far higher than a Python per-shot
        loop — this is the path to use for Monte-Carlo logical-error sampling.
        """
        arr = np.asarray(shots, dtype=np.uint8)
        if arr.ndim != 2:
            raise ValueError(f"shots must be 2D, got shape {arr.shape}")
        if arr.shape[1] < self._num_detectors:
            pad = np.zeros((arr.shape[0], self._num_detectors - arr.shape[1]), np.uint8)
            arr = np.concatenate([arr, pad], axis=1)
        # The Rust batch decoder reads the buffer row-major; force C-contiguity so
        # Fortran-ordered / non-contiguous inputs decode correctly (not silently wrong).
        arr = np.ascontiguousarray(arr, dtype=np.uint8)
        dec = self._ensure_decoder()
        corr = np.asarray(dec.batch_decode(arr), dtype=np.uint8)
        if self._faults_matrix is not None:
            return ((self._faults_matrix @ corr.T) & 1).T.astype(np.uint8)
        return corr

    def decode_to_edges_array(self, syndrome: Sequence[int]) -> np.ndarray:
        """Return the raw edge correction (length ``num_edges``)."""
        s = np.asarray(syndrome, dtype=np.uint8).reshape(-1)
        if s.shape[0] < self._num_detectors:
            s = np.concatenate(
                [s, np.zeros(self._num_detectors - s.shape[0], np.uint8)]
            )
        return self._edge_correction(s)

    def __repr__(self) -> str:  # pragma: no cover - cosmetic
        return (
            f"<qector Matching: {self._num_detectors} detectors, "
            f"{self.num_edges} edges, {self.num_fault_ids} fault ids>"
        )


def _as_id_set(fault_ids: Any) -> set:
    if fault_ids is None:
        return set()
    if isinstance(fault_ids, (int, np.integer)):
        return {int(fault_ids)}
    return set(int(x) for x in fault_ids)
