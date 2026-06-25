"""
qector_decoder_v3.belief_matching — belief-propagation + MWPM (belief-matching).

Plain minimum-weight perfect matching on a *decomposed* (graphlike) detector
error model discards the X–Z correlations that the decomposition split apart.
**Belief-matching** (Higgott et al., 2023) recovers them and achieves a **lower
logical error rate than plain MWPM/PyMatching** at the same noise.

The architecture (matching the reference ``beliefmatching`` package, but using
QECTOR's exact weighted Blossom for the matching step):

  1. Build two views of the DEM — the **hyperedge** check matrix (one column per
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
>>> dem = stim.Circuit.generated("surface_code:rotated_memory_x", distance=5,
...        rounds=5, after_clifford_depolarization=0.005).detector_error_model(
...        decompose_errors=True)
>>> bm = BeliefMatching.from_detector_error_model(dem)
>>> predicted_observables = bm.decode_batch(detection_events)
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, FrozenSet, List

import numpy as np

from . import BlossomDecoder
from ._bp_core import build_incidence, sum_product_bp

__all__ = ["BeliefMatching", "build_matching_matrices"]


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


def _iter_xor(sets: List[List[int]]) -> FrozenSet[int]:
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

    hyper_ids: Dict[FrozenSet[int], int] = {}
    edge_ids: Dict[FrozenSet[int], int] = {}
    hyper_obs: Dict[int, FrozenSet[int]] = {}
    edge_obs: Dict[int, FrozenSet[int]] = {}
    priors: Dict[int, float] = {}
    hyper_to_edges: Dict[int, set] = {}

    def handle(prob: float, dets: List[List[int]], frames: List[List[int]]):
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
        dets: List[List[int]] = [[]]
        frames: List[List[int]] = [[]]
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
    """Belief-propagation + minimum-weight perfect matching decoder.

    Can be constructed from:
    - A :class:`_Matrices` object (returned by :func:`build_matching_matrices`).
    - A raw numpy check matrix ``H`` of shape ``(num_detectors, num_qubits)``
      with uniform priors (each column weight ``-log(p/(1-p))`` at ``p=0.1``).
      This is the convenience constructor: ``BeliefMatching(H)`` or
      ``BeliefMatching(H, p=0.05)``.  Observable matrix is assumed to be
      identity (each qubit is its own logical).
    """

    def __init__(self, matrices, max_iter: int = 30, bp_shortcut: bool = False, p: float = 0.1):
        # Accept a raw numpy H matrix as a convenience shortcut.
        if isinstance(matrices, np.ndarray):
            H = np.asarray(matrices, dtype=np.uint8)
            if H.ndim != 2:
                raise ValueError(f"H must be 2D, got shape {H.shape}")
            num_detectors, num_qubits = H.shape
            # Uniform prior p for all mechanisms
            prior_arr = np.full(num_qubits, float(p), dtype=np.float64)
            # Observable matrix: identity (each qubit is its own logical)
            obs_m = np.eye(num_qubits, dtype=np.uint8)
            # No hyperedge structure: hyper == edge
            matrices = _Matrices(
                hyper_check=H.copy(),
                hyper_obs=obs_m,
                hyper_priors=prior_arr,
                hyper_to_edge=np.eye(num_qubits, dtype=np.uint8),
                edge_check=H.copy(),
                edge_obs=obs_m,
                num_detectors=num_detectors,
                num_observables=num_qubits,
            )
        self._m = matrices
        self.max_iter = int(max_iter)
        # Trusting BP's hard decision when it merely satisfies the syndrome can
        # pick a worse logical coset than MWPM (hurts at small d). Off by default:
        # always reweight + match, which is never worse than plain matching.
        # (The matching step is exact weighted MWPM, rebuilt per shot with BP's
        # posteriors — this is the high-accuracy path, slower than plain MWPM.)
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
        self._edge_c2q: List[List[int]] = [
            sorted(int(e) for e in np.nonzero(matrices.edge_check[c])[0]) for c in range(self.n_checks)
        ]

    # -- constructors ------------------------------------------------------
    @classmethod
    def from_detector_error_model(cls, dem: Any, max_iter: int = 20) -> "BeliefMatching":
        return cls(build_matching_matrices(dem), max_iter=max_iter)

    @classmethod
    def from_stim_circuit(cls, circuit, max_iter: int = 20) -> "BeliefMatching":
        return cls.from_detector_error_model(circuit.detector_error_model(decompose_errors=True), max_iter=max_iter)

    # -- decoding ----------------------------------------------------------
    def decode(self, syndrome) -> np.ndarray:
        s = np.asarray(syndrome, dtype=np.uint8).reshape(-1)
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
