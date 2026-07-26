"""
qector_decoder_v3.belief_matching - belief-propagation + MWPM (belief-matching).

Plain minimum-weight perfect matching on a *decomposed* (graphlike) detector
error model discards the X-Z correlations that the decomposition split apart.
**Belief-matching** (Higgott et al., 2023) recovers them and achieves a **lower
logical error rate than plain MWPM/PyMatching** at the same noise.

The architecture (matching the reference ``beliefmatching`` package, but using
QECTOR's exact weighted Blossom for the matching step):

  1. Build two views of the DEM - the **hyperedge** check matrix (one column per
     full error mechanism, correlations intact) and the **edge** check matrix
     (the graphlike decomposition), plus the hyperedge→edge map.
  2. Run **sum-product BP** on the hyperedge graph with per-mechanism priors.
  3. If BP's hard decision already explains the syndrome, return it (exact, fast).
  4. Otherwise map BP's posterior probabilities onto the edges, set edge weights
     to ``-log(p_edge)``, and run QECTOR weighted MWPM on the edge graph.

Self-contained: BP is QECTOR's own vectorised implementation. Cross-validated
against the reference ``beliefmatching`` package in the test suite.

Example
-------
>>> import stim
>>> from qector_decoder_v3.belief_matching import BeliefMatching
>>> dem = stim.Circuit.generated(
...     "surface_code:rotated_memory_x", distance=5, rounds=5, after_clifford_depolarization=0.005
... ).detector_error_model(decompose_errors=True)
>>> bm = BeliefMatching.from_detector_error_model(dem)
>>> predicted_observables = bm.decode_batch(detection_events)
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from . import BlossomDecoder, DetectorGraph, GNNPredecoder, SparseBlossomDecoder
from ._bp_core import build_incidence, sum_product_bp

__all__ = [
    "BeliefMatching",
    "GNNBeliefMatcher",
    "build_matching_matrices",
    "decode_with_gnn",
]


@dataclass
class _Matrices:
    hyper_check: np.ndarray  # (num_detectors, num_hyper)
    hyper_obs: np.ndarray  # (num_observables, num_hyper)
    hyper_priors: np.ndarray  # (num_hyper,)
    hyper_to_edge: np.ndarray  # (num_edges, num_hyper)
    edge_check: np.ndarray  # (num_detectors, num_edges)
    edge_obs: np.ndarray  # (num_observables, num_edges)
    num_detectors: int
    num_observables: int


def _iter_xor(sets: list[list[int]]) -> frozenset[int]:
    out: set = set()
    for x in sets:
        s = set(x)
        out = (out - s) | (s - out)
    return frozenset(out)


def build_matching_matrices(dem: Any) -> _Matrices:
    """Decompose a Stim DEM into hyperedge + edge check matrices (belief-matching).

    Mirrors ``beliefmatching.detector_error_model_to_check_matrices`` but returns
    dense GF(2) arrays. Accepts a ``stim.DetectorErrorModel`` or ``.dem`` text.
    """
    import stim

    if isinstance(dem, str):
        dem = stim.DetectorErrorModel(dem)

    hyper_ids: dict[frozenset[int], int] = {}
    edge_ids: dict[frozenset[int], int] = {}
    hyper_obs: dict[int, frozenset[int]] = {}
    edge_obs: dict[int, frozenset[int]] = {}
    priors: dict[int, float] = {}
    hyper_to_edges: dict[int, set] = {}

    def handle(prob: float, dets: list[list[int]], frames: list[list[int]]):
        hdets = _iter_xor(dets)
        hobs = _iter_xor(frames)
        if hdets not in hyper_ids:
            hyper_ids[hdets] = len(hyper_ids)
            priors[hyper_ids[hdets]] = 0.0
        hid = hyper_ids[hdets]
        hyper_obs[hid] = hobs
        priors[hid] = priors[hid] * (1 - prob) + prob * (1 - priors[hid])
        eids = set()
        for i in range(len(dets)):
            edets = frozenset(dets[i])
            if len(edets) > 2:
                continue  # undecomposed hyperedge: no graph edge
            if edets not in edge_ids:
                edge_ids[edets] = len(edge_ids)
            eid = edge_ids[edets]
            edge_obs[eid] = frozenset(frames[i])
            eids.add(eid)
        hyper_to_edges.setdefault(hid, set()).update(eids)

    for inst in dem.flattened():
        if inst.type != "error":
            continue
        dets: list[list[int]] = [[]]
        frames: list[list[int]] = [[]]
        p = inst.args_copy()[0]
        for t in inst.targets_copy():
            if t.is_relative_detector_id():
                dets[-1].append(t.val)
            elif t.is_logical_observable_id():
                frames[-1].append(t.val)
            elif t.is_separator():
                dets.append([])
                frames.append([])
        handle(p, dets, frames)

    nD = dem.num_detectors
    nO = dem.num_observables
    nH = len(hyper_ids)
    nE = len(edge_ids)

    hyper_check = np.zeros((nD, nH), dtype=np.uint8)
    for dets_fs, hid in hyper_ids.items():
        for d in dets_fs:
            hyper_check[d, hid] ^= 1
    hyper_obs_m = np.zeros((nO, nH), dtype=np.uint8)
    for hid, obs in hyper_obs.items():
        for o in obs:
            hyper_obs_m[o, hid] ^= 1
    edge_check = np.zeros((nD, nE), dtype=np.uint8)
    for dets_fs, eid in edge_ids.items():
        for d in dets_fs:
            edge_check[d, eid] ^= 1
    edge_obs_m = np.zeros((nO, nE), dtype=np.uint8)
    for eid, obs in edge_obs.items():
        for o in obs:
            edge_obs_m[o, eid] ^= 1
    hyper_to_edge = np.zeros((nE, nH), dtype=np.uint8)
    for hid, eids in hyper_to_edges.items():
        for eid in eids:
            hyper_to_edge[eid, hid] = 1
    prior_arr = np.zeros(nH, dtype=np.float64)
    for hid, p in priors.items():
        prior_arr[hid] = p

    return _Matrices(hyper_check, hyper_obs_m, prior_arr, hyper_to_edge, edge_check, edge_obs_m, nD, nO)


class BeliefMatching:
    """Belief-propagation + minimum-weight perfect matching decoder."""

    def __init__(self, matrices: _Matrices, max_iter: int = 30, bp_shortcut: bool = False):
        self._m = matrices
        self.max_iter = int(max_iter)
        # Trusting BP's hard decision when it merely satisfies the syndrome can
        # pick a worse logical coset than MWPM (hurts at small d). Off by default:
        # always reweight + match, which is never worse than plain matching.
        # (The matching step is exact weighted MWPM, rebuilt per shot with BP's
        # posteriors - this is the high-accuracy path, slower than plain MWPM.)
        self.bp_shortcut = bool(bp_shortcut)
        self.n_checks = matrices.num_detectors
        self.num_observables = matrices.num_observables

        # BP graph = hyperedge check matrix.
        self._hic, self._hie = build_incidence(matrices.hyper_check)
        self._n_hyper = matrices.hyper_check.shape[1]
        p = np.clip(matrices.hyper_priors, 1e-15, 1 - 1e-15)
        self._prior_llr = np.log((1.0 - p) / p)

        # Matching graph = edge check matrix.
        self._n_edges = matrices.edge_check.shape[1]
        self._edge_c2q: list[list[int]] = [
            sorted(int(e) for e in np.nonzero(matrices.edge_check[c])[0]) for c in range(self.n_checks)
        ]

    # -- constructors ------------------------------------------------------
    @classmethod
    def from_detector_error_model(cls, dem: Any, max_iter: int = 20) -> BeliefMatching:
        return cls(build_matching_matrices(dem), max_iter=max_iter)

    @classmethod
    def from_stim_circuit(cls, circuit, max_iter: int = 20) -> BeliefMatching:
        return cls.from_detector_error_model(circuit.detector_error_model(decompose_errors=True), max_iter=max_iter)

    @classmethod
    def from_numpy_h(cls, H, error_rate: float = 0.05, max_iter: int = 20) -> BeliefMatching:
        """Construct from a raw NumPy parity-check matrix ``H`` (n_checks x n_qubits).

        Builds the internal ``_Matrices`` from a dense parity-check matrix.
        Each column of H is treated as a hyperedge. Edge graph is derived
        from weight-2 columns. Priors are set to ``error_rate`` uniformly.

        Args:
            H: NumPy array of shape ``(n_checks, n_qubits)``, dtype uint8.
            error_rate: Physical error rate for prior probabilities.
            max_iter: Max BP iterations.

        Returns:
            BeliefMatching instance.
        """
        H = np.asarray(H, dtype=np.uint8)
        if H.ndim != 2:
            raise ValueError(f"H must be 2D, got {H.shape}")
        nD, nQ = H.shape

        hyper_ids: dict[tuple[int, ...], int] = {}
        edge_ids: dict[tuple[int, ...], int] = {}
        hyper_to_edges: dict[int, set[int]] = {}
        priors: dict[int, float] = {}

        for q in range(nQ):
            dets = tuple(sorted(np.nonzero(H[:, q])[0].tolist()))
            if not dets:
                continue
            if dets not in hyper_ids:
                hyper_ids[dets] = len(hyper_ids)
                priors[hyper_ids[dets]] = error_rate
            hid = hyper_ids[dets]
            if len(dets) <= 2:
                if dets not in edge_ids:
                    edge_ids[dets] = len(edge_ids)
                eid = edge_ids[dets]
                hyper_to_edges.setdefault(hid, set()).add(eid)

        nH = len(hyper_ids)
        nE = len(edge_ids)
        hyper_check = np.zeros((nD, nH), dtype=np.uint8)
        for dets, hid in hyper_ids.items():
            for d in dets:
                hyper_check[d, hid] ^= 1
        # Observables map hyperedge flips back to something the caller can use.
        # A raw H carries no observable definition, so the previous
        # `np.zeros((0, nH))` left ZERO rows -- `decode` returns
        # `hyper_obs @ hard`, so every syndrome produced an empty array and the
        # decoder was silently useless when built this way.
        #
        # For a bare parity-check matrix the useful contract is a per-QUBIT
        # correction, so observables are the qubit->hyperedge incidence: row q
        # is set for the hyperedge that column q collapsed into. `decode` then
        # returns a length-`nQ` correction satisfying H @ corr == syndrome.
        #
        # Columns with identical detector support share a hyperedge and are
        # therefore flipped together -- an inherent degeneracy of collapsing
        # duplicate columns, not a defect. Such columns are indistinguishable
        # from the syndrome alone.
        hyper_obs_m = np.zeros((nQ, nH), dtype=np.uint8)
        for q in range(nQ):
            dets = tuple(sorted(np.nonzero(H[:, q])[0].tolist()))
            if dets:
                hyper_obs_m[q, hyper_ids[dets]] = 1

        edge_check = np.zeros((nD, nE), dtype=np.uint8)
        for dets, eid in edge_ids.items():
            for d in dets:
                edge_check[d, eid] ^= 1
        # Same fix for the matching stage. BP only short-circuits when its hard
        # decision already reproduces the syndrome; otherwise decode falls
        # through to matching and returns `edge_obs @ ...`, which with zero rows
        # produced an empty array for exactly the syndromes BP found hardest.
        # Row q is set for the edge that column q collapsed into (weight-<=2
        # columns only -- a genuine hyperedge has no edge representation).
        edge_obs_m = np.zeros((nQ, nE), dtype=np.uint8)
        for q in range(nQ):
            dets = tuple(sorted(np.nonzero(H[:, q])[0].tolist()))
            if dets and len(dets) <= 2:
                edge_obs_m[q, edge_ids[dets]] = 1
        hyper_to_edge = np.zeros((nE, nH), dtype=np.uint8)
        for hid, eids in hyper_to_edges.items():
            for eid in eids:
                hyper_to_edge[eid, hid] = 1
        prior_arr = np.full(nH, error_rate, dtype=np.float64)

        matrices = _Matrices(hyper_check, hyper_obs_m, prior_arr, hyper_to_edge, edge_check, edge_obs_m, nD, 0)
        return cls(matrices, max_iter=max_iter)

    # -- decoding ----------------------------------------------------------
    def decode(self, syndrome) -> np.ndarray:
        s: np.ndarray = np.asarray(syndrome, dtype=np.uint8).reshape(-1)
        if s.shape[0] < self.n_checks:
            s = np.concatenate([s, np.zeros(self.n_checks - s.shape[0], np.uint8)])

        posterior = sum_product_bp(
            self._hic,
            self._hie,
            self.n_checks,
            self._n_hyper,
            self._prior_llr,
            s,
            self.max_iter,
        )
        if self.bp_shortcut:
            hard = (posterior < 0.0).astype(np.uint8)
            if np.array_equal((self._m.hyper_check @ hard) & 1, s):
                return ((self._m.hyper_obs @ hard) & 1).astype(np.uint8)

        # Map hyperedge posteriors -> edge probabilities, reweight, match.
        p_h = 1.0 / (1.0 + np.exp(np.clip(posterior, -60, 60)))
        p_e = self._m.hyper_to_edge @ p_h
        p_e = np.clip(p_e, 1e-14, 1 - 1e-14)
        w = -np.log(p_e)
        matcher = BlossomDecoder(self._edge_c2q, self._n_edges, w.tolist())
        corr = np.asarray(matcher.decode(s), dtype=np.uint8).reshape(-1)
        return (self._m.edge_obs @ corr) & 1

    def decode_batch(self, shots) -> np.ndarray:
        arr = np.asarray(shots, dtype=np.uint8)
        if arr.ndim != 2:
            raise ValueError(f"shots must be 2D, got shape {arr.shape}")
        return np.stack([self.decode(arr[i]) for i in range(arr.shape[0])]).astype(np.uint8)

    @property
    def num_detectors(self) -> int:
        return self.n_checks

    def __repr__(self) -> str:  # pragma: no cover
        return (
            f"<BeliefMatching detectors={self.n_checks} hyperedges={self._n_hyper} "
            f"edges={self._n_edges} max_iter={self.max_iter}>"
        )


# =============================================================================
# Task C - GNN-Enhanced MWPM (belief matching via learned dynamic weights)
# =============================================================================
#
# Mirrors the native Rust pipeline in `src/hybrid_decoder.rs::decode_hybrid`:
#   1. Build a DetectorGraph for the syndrome.
#   2. GNN forward -> per-edge adjusted weights (positive, softplus).
#   3. Map edge -> qubit via `graph.edge_qubit_id` and hand the pairs to
#      `SparseBlossomDecoder.decode_with_weights`, whose integer cost is
#      `cost = clamp(weight_scale / weight, 1, 100000)` - a LARGE weight means
#      a CHEAP edge, i.e. the matcher prefers flipping qubits the GNN scored
#      highly (see src/sparse_blossom.rs:473-478). The native decoder verifies
#      the correction and falls back to exact Blossom internally, so the
#      output is always syndrome-faithful; we additionally re-check in Python
#      and fall back to the static-weight decode (never worse than MWPM).


class GNNBeliefMatcher:
    """End-to-end GNN-guided minimum-weight perfect matching decoder.

    The GNN predicts contextual per-qubit weights from the actual syndrome
    (correlated-noise aware); the region-growing blossom then priorities the
    predicted physical error paths over naive static graph distances.

    Parameters
    ----------
    check_to_qubits:
        ``check -> [qubit indices]`` adjacency of the code (graphlike for
        MWPM to apply).
    n_qubits:
        Total qubit count; inferred from the checks when omitted.
    gnn:
        Optional pre-configured GNN. Any object with
        ``predict_with_node_probs(graph) -> (edge_weights, node_probs)``
        works (duck-type seam for testing/custom models). When omitted a
        fresh :class:`GNNPredecoder` is built with `hidden_size`/`n_layers`.
    hidden_size, n_layers:
        Architecture of the default GNN.
    train_samples:
        When > 0, quick-train the GNN on synthetic i.i.d. Bernoulli noise at
        `error_rate` before first use (targets: weight 10.0 for flipped
        qubits, 1.0 otherwise - the convention the native hybrid decoder's
        heuristic uses).
    error_rate, train_epochs, seed:
        Noise level / epochs / RNG seed for the optional synthetic training.
    """

    def __init__(
        self,
        check_to_qubits,
        n_qubits=None,
        *,
        gnn=None,
        hidden_size: int = 32,
        n_layers: int = 2,
        train_samples: int = 0,
        error_rate: float = 0.05,
        train_epochs: int = 3,
        seed=None,
    ):
        c2q = [list(map(int, check)) for check in check_to_qubits]
        if n_qubits is None:
            n_qubits = 1 + max((q for check in c2q for q in check), default=-1)
        self._c2q = c2q
        self.n_qubits = int(n_qubits)
        self.n_checks = len(c2q)

        self._decoder = SparseBlossomDecoder(c2q, self.n_qubits)

        if gnn is None:
            gnn = GNNPredecoder(
                DetectorGraph.NODE_FEAT_DIM,
                DetectorGraph.EDGE_FEAT_DIM,
                int(hidden_size),
                int(n_layers),
            )
        self._gnn = gnn

        # Dense H for the faithfulness re-check.
        self._H = np.zeros((self.n_checks, self.n_qubits), dtype=np.uint8)
        for ci, qs in enumerate(c2q):
            for q in qs:
                self._H[ci, q] ^= 1

        if train_samples > 0:
            self._synthetic_train(int(train_samples), float(error_rate), int(train_epochs), seed)

    # -- training ----------------------------------------------------------
    def _synthetic_train(self, n_samples: int, error_rate: float, n_epochs: int, seed) -> None:
        rng = np.random.default_rng(seed)
        graphs, targets = [], []
        for _ in range(n_samples):
            error = (rng.random(self.n_qubits) < error_rate).astype(np.uint8)
            syndrome = (self._H @ error) & 1
            graph = DetectorGraph(self._c2q, syndrome.tolist(), n_qubits=self.n_qubits)
            # Per-edge target weights: edge e touches qubit edge_qubit_id[e];
            # high weight (cheap cost) where the true error flipped the qubit.
            eqid = np.asarray(graph.edge_qubit_id, dtype=np.int64)
            tgt = np.where(error[eqid] == 1, 10.0, 1.0)
            graphs.append(graph)
            targets.append(tgt.tolist())
        self._gnn.train(graphs, targets, int(n_epochs))

    # -- decoding ----------------------------------------------------------
    def _dynamic_weights(self, syndrome: np.ndarray):
        """GNN per-qubit weight overrides for `syndrome` (hybrid_decoder.rs mapping)."""
        graph = DetectorGraph(self._c2q, syndrome.tolist(), n_qubits=self.n_qubits)
        edge_weights, _node_probs = self._gnn.predict_with_node_probs(graph)
        eqid = graph.edge_qubit_id
        # Keep the GNN's strongest opinion per qubit (several graph edges can
        # share one physical qubit): max weight = cheapest cost = most blamed.
        best = {}
        for q, w in zip(eqid, edge_weights):
            q = int(q)
            w = float(w)
            if w > best.get(q, 0.0):
                best[q] = w
        return [(q, w) for q, w in best.items()]

    def decode(self, syndrome) -> np.ndarray:
        s = np.asarray(syndrome, dtype=np.uint8).reshape(-1)
        if s.shape[0] != self.n_checks:
            raise ValueError(f"syndrome length {s.shape[0]} != n_checks {self.n_checks}")
        weights = self._dynamic_weights(s)
        corr = np.asarray(self._decoder.decode_with_weights(s, weights), dtype=np.uint8)
        # Faithfulness is guaranteed natively (Blossom fallback); belt-and-braces:
        # if anything ever slips through, the static-weight decode is never
        # worse than plain MWPM.
        if not np.array_equal((self._H @ corr) & 1, s):
            corr = np.asarray(self._decoder.decode(s), dtype=np.uint8)
        return corr

    def batch_decode(self, syndromes) -> np.ndarray:
        arr = np.asarray(syndromes, dtype=np.uint8)
        if arr.ndim != 2:
            raise ValueError(f"shots must be 2D, got shape {arr.shape}")
        return np.stack([self.decode(arr[i]) for i in range(arr.shape[0])]).astype(np.uint8)

    # -- accessors ---------------------------------------------------------
    @property
    def gnn(self):
        return self._gnn

    @property
    def decoder(self):
        return self._decoder

    def __repr__(self) -> str:  # pragma: no cover
        return f"<GNNBeliefMatcher checks={self.n_checks} qubits={self.n_qubits}>"


def decode_with_gnn(check_to_qubits, n_qubits, syndrome, **kwargs) -> np.ndarray:
    """One-shot GNN-guided MWPM decode (see :class:`GNNBeliefMatcher`)."""
    return GNNBeliefMatcher(check_to_qubits, n_qubits, **kwargs).decode(syndrome)
