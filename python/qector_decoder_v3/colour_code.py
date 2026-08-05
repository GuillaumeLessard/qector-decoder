"""C1-04: Colour-code decoder.

Why colour codes need their own entry point
-------------------------------------------
Minimum-weight perfect matching is not a correct decoder for colour codes. A
colour-code error mechanism can light up three detectors at once, and such a
mechanism has no graphlike (>=2-detector) decomposition. Stim says so directly:
``detector_error_model(decompose_errors=True)`` **raises** on
``color_code:memory_xyz`` at d>=5. Any colour-code "matching" decoder is
therefore either decoding a different code than it claims, or silently dropping
the mechanisms it cannot represent.

The approach used here
----------------------
Decode the **undecomposed hypergraph** detector error model. Two stages are
available; the default (``method="bposd"``) sends the whole syndrome to
BP-OSD. Stage 1 is opt-in via ``method="cluster_bposd"``:

1. **Hyperedge cluster expansion** — weighted union-find growth over the
   hypergraph (detectors = nodes, mechanisms = hyperedges, growth cost =
   prior log-odds), then a verified per-component prior-greedy solve. This is a
   reference implementation of the algorithm — faithful and opt-in, but on this
   machine it is *not* faster than global BP-OSD (measured, see table below)
   and slightly less accurate on dense clusters, so it is not the default.
2. **BP-OSD backstop** — any cluster component that fails peel verification
   (on a genuine hyperedge, even parity does not imply span) and any residual
   syndrome is decoded by BP-OSD, which is defined for arbitrary GF(2) check
   matrices and needs no graphlike structure. ``method="bposd"`` skips stage 1
   and sends the whole syndrome to BP-OSD.

Both stages extract the real ``(detectors x mechanisms)`` check matrix ``H``,
the ``(observables x mechanisms)`` observable matrix ``L``, and per-mechanism
priors from the DEM; observables are predicted as ``(L @ e) & 1``.

Measured on ``color_code:memory_xyz``, p=0.003, rounds=d (1500 shots at
d=3/5, 600 at d=7; seed 1000+d). ``bposd`` is the default (accuracy-first);
``cluster_bposd`` is the opt-in fast path — within sampling slack of BP-OSD at
d=3/7 but measurably less accurate on dense clusters at d=5. A matching
decoder is never a valid choice here: it cannot represent the colour-code
hyperedges and, where it can be forced to build (d=3), it loses ~2×.

===== =============== =============== =============== ===============
d     bposd (default) cluster_bposd   belief-match   trivial
===== =============== =============== =============== ===============
3     0.0240          0.0240          0.0440         0.0427
5     0.0227          0.0413          cannot build   0.1440
7     0.0217          0.0250          cannot build   0.2467
===== =============== =============== =============== ===============

Usage
-----
>>> import stim
>>> from qector_decoder_v3.colour_code import ColourCodeDecoder
>>> circuit = stim.Circuit.generated("color_code:memory_xyz", distance=5, rounds=5, after_clifford_depolarization=0.003)
>>> dec = ColourCodeDecoder.from_stim_circuit(circuit)
>>> det, obs = circuit.compile_detector_sampler().sample(100, separate_observables=True)
>>> prediction = dec.decode_batch(det)
"""

from __future__ import annotations

import heapq
from typing import Any

import numpy as np

__all__ = ["ColourCodeDecoder", "colour_codes_from_dem"]


class _HypergraphClusterExpansion:
    """Weighted hyperedge cluster-expansion pre-decoder (union-find on a hypergraph).

    Nodes are detectors, hyperedges are DEM mechanisms (which may touch 1, 2 or
    >=3 detectors — the >=3 case is what makes a colour code unmatchable).
    Clusters rooted at defect detectors expand by claiming their **cheapest**
    frontier mechanism first (growth cost = prior log-odds ``log((1-p)/p)``),
    merging whenever a mechanism's detectors span several clusters. A cluster
    stops growing once it is *valid*: even residual syndrome parity, or it has
    covered a boundary mechanism (one touching a single detector — the DEM
    analogue of the boundary node in graphlike union-find).

    Each final cluster's covered-mechanism incidence is then peeled leaf-to-root
    on a BFS spanning tree of the bipartite (detector, mechanism) graph. Peeling
    is **verified**: on a genuine hyperedge component even parity does not
    guarantee span, so any component whose residuals do not cancel is discarded
    wholesale and its syndrome handed back to the caller (the BP-OSD backstop).
    The committed part therefore always satisfies ``H_cluster @ e == s_cluster``
    on its own detectors.
    """

    def __init__(self, H: np.ndarray, priors: np.ndarray):
        H = np.asarray(H, dtype=np.uint8)
        self._H = H
        self.n_checks, self.n_mech = (int(H.shape[0]), int(H.shape[1]))
        # Bipartite incidence lists (built once).
        self.mech_dets: list[np.ndarray] = [np.nonzero(H[:, m])[0].astype(np.int64) for m in range(self.n_mech)]
        self.det_mechs: list[np.ndarray] = [np.nonzero(H[u, :])[0].astype(np.int64) for u in range(self.n_checks)]
        # Growth cost per mechanism: prior log-odds, floored to stay positive.
        p = np.clip(np.asarray(priors, dtype=np.float64).reshape(-1), 1e-15, 1 - 1e-15)
        self.mech_cost = np.maximum(np.log((1.0 - p) / p), 1e-6)
        # Mechanisms with zero detectors never constrain the syndrome; the
        # cluster stage ignores them (BP-OSD prices them by prior on the
        # residual path).
        self.growable = np.array([d.size > 0 for d in self.mech_dets], dtype=bool)

    def decode(self, syndrome: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Return ``(correction, residual)`` with ``H @ correction ^ residual ==
        syndrome``: components that fail verification are rolled back into
        ``residual`` for the BP-OSD backstop.
        """
        s = np.asarray(syndrome, dtype=np.uint8).reshape(-1) & 1
        defects = np.nonzero(s)[0]
        if defects.size == 0:
            return np.zeros(self.n_mech, np.uint8), s

        # ---- growth: cheapest-mechanism-first union-find over detectors ----
        parent = np.arange(self.n_checks, dtype=np.int64)
        parity = s.astype(np.uint8).copy()  # XOR of syndrome bits per root
        has_boundary = np.zeros(self.n_checks, dtype=bool)
        clustered = np.zeros(self.n_checks, dtype=bool)  # detector joined a cluster
        clustered[defects] = True
        covered = np.zeros(self.n_mech, dtype=bool)
        covered_by = np.full(self.n_mech, -1, dtype=np.int64)  # m -> claiming root

        def find(u: int) -> int:
            r = u
            while parent[r] != r:
                r = int(parent[r])
            while parent[u] != r:
                parent[u], u = r, int(parent[u])
            return r

        def valid(r: int) -> bool:
            return (not parity[r]) or has_boundary[r]

        heap: list[tuple[float, int]] = []
        for u in defects:
            for m in self.det_mechs[u]:
                if self.growable[m] and not covered[m]:
                    heapq.heappush(heap, (float(self.mech_cost[m]), int(m)))

        while heap:
            if all(valid(find(int(u))) for u in defects):
                break
            _, m = heapq.heappop(heap)
            if covered[m] or not self.growable[m]:
                continue
            dets = self.mech_dets[m]
            roots = sorted({find(int(u)) for u in dets if clustered[u]})
            if not roots:
                continue  # not on any live cluster frontier
            if all(valid(r) for r in roots):
                continue  # only valid clusters touch it: not needed for growth
            # Merge all touched clusters + any unclustered detectors of m.
            root = roots[0]
            for other in roots[1:]:
                parent[other] = root
                parity[root] ^= parity[other]
                has_boundary[root] |= has_boundary[other]
            for u in dets:
                u = int(u)
                if not clustered[u]:
                    clustered[u] = True
                    parent[u] = root
                    parity[root] ^= s[u]
                    for m2 in self.det_mechs[u]:
                        if self.growable[m2] and not covered[m2]:
                            heapq.heappush(heap, (float(self.mech_cost[m2]), int(m2)))
            covered[m] = True
            covered_by[m] = root
            if dets.size == 1:
                has_boundary[root] = True

        return self._solve_components(s, covered, covered_by, find)

    def _solve_components(
        self, s: np.ndarray, covered: np.ndarray, covered_by: np.ndarray, find
    ) -> tuple[np.ndarray, np.ndarray]:
        """Per-component prior-greedy min-weight solve, with verification.

        Each grown cluster's covered mechanism set is solved independently with
        the GF(2) ordered-statistics solve ordered by *prior* reliability
        (ascending log-odds), i.e. the cheapest mechanisms form the solved basis
        and the expensive ones stay at their (zero) hard decision — the
        prior-greedy min-weight explanation of the component. This replaced an
        earlier plain spanning-tree peel: the tree solution is unique but not
        weight-optimal on cyclic components, which cost measurable LER at d=7
        (0.053 vs 0.022 BP-OSD-only, 300-shot lane) — the per-component greedy
        solve closes most of that gap. (Ordering by a short 5-iteration BP
        posterior instead of priors was *measured worse* at d=5 — 0.049 vs
        0.041 prior-greedy vs 0.023 BP-OSD-only — because so few iterations give
        a misleading ordering, so the prior-greedy ordering is kept.)

        Verification: a component whose solve does not reproduce its syndrome
        exactly (possible only on genuine hyperedge components, where even
        parity does not imply span) is rolled back wholesale — its mechanisms
        stay 0 and its syndrome remains in ``residual`` for the BP-OSD
        backstop. The committed part always satisfies
        ``H_cluster @ e == s_cluster`` on its own detectors.
        """
        from .bposd import _gf2_osd_solve

        x = np.zeros(self.n_mech, np.uint8)
        residual = s.copy()

        # Group covered mechanisms by their cluster's current root.
        comp_mechs: dict[int, list[int]] = {}
        for m in np.nonzero(covered)[0]:
            comp_mechs.setdefault(find(int(covered_by[m])), []).append(int(m))

        gamma = self.mech_cost  # prior log-odds log((1-p)/p), floored positive
        for mechs in comp_mechs.values():
            mechs_arr = np.asarray(mechs, dtype=np.int64)
            dets = np.unique(np.concatenate([self.mech_dets[m] for m in mechs]))
            sub = self._H[np.ix_(dets, mechs_arr)]
            s_C = s[dets] & 1
            # Ascending prior reliability: the least reliable (cheapest, highest
            # p) columns form the solved basis; expensive mechanisms stay 0.
            order = np.argsort(gamma[mechs_arr], kind="stable")
            hard = np.zeros(mechs_arr.size, np.uint8)
            x_C, _ = _gf2_osd_solve(sub, s_C, order, hard)
            if not np.array_equal((sub @ x_C) & 1, s_C):
                continue  # not spanned: roll the component back to BP-OSD
            for local_i, m in enumerate(mechs):
                if x_C[local_i]:
                    x[m] = 1
                    residual[self.mech_dets[m]] ^= 1

        return x, residual & 1


class ColourCodeDecoder:
    """Colour-code decoder: BP-OSD over the undecomposed hypergraph DEM.

    Parameters
    ----------
    dem:
        A ``stim.DetectorErrorModel``, or its string form. It must **not** be
        decomposed — see the module docstring. Passing a decomposed DEM still
        works but throws away the very mechanisms that make the code a colour
        code, so :meth:`from_stim_circuit` builds the DEM itself.
    max_iter:
        BP iterations before OSD post-processing.
    osd_order:
        OSD order; ``0`` is OSD-0 (fast, the usual choice).
    method:
        ``"bposd"`` (default) sends the whole syndrome to BP-OSD — the
        accuracy-first path, and the pre-cluster behaviour. ``"cluster_bposd"``
        runs the **hyperedge cluster-expansion** stage first (weighted union-find
        growth over the undecomposed hypergraph DEM + verified per-component
        prior-greedy solve), then hands only the unresolved residual to BP-OSD.
        This is a **reference implementation** of the algorithm: measured on this
        machine (pure-Python growth loop) it is currently *slower* than global
        BP-OSD, not faster (d=3 5711 vs 7677 shots/s; d=5 422 vs 889), and a
        little less accurate on dense clusters (d=5 LER 0.0433 vs 0.0267; d=7
        0.0250 vs 0.0217; d=3 0.0250 vs 0.0217). Both are syndrome-faithful.
        Keep ``cluster_bposd`` as an opt-in experiment; it is not the default
        because it is neither faster nor more accurate at these distances.
    """

    def __init__(self, dem: Any, max_iter: int = 30, osd_order: int = 0, method: str = "bposd"):
        import stim

        from .belief_matching import build_matching_matrices
        from .bposd import BpOsdDecoder

        if isinstance(dem, str):
            dem = stim.DetectorErrorModel(dem)

        if method not in ("cluster_bposd", "bposd"):
            raise ValueError(f"method must be 'cluster_bposd' or 'bposd', got {method!r}")
        self.method = method

        m = build_matching_matrices(dem)
        self._H = np.asarray(m.hyper_check, dtype=np.uint8)
        self._L = np.asarray(m.hyper_obs, dtype=np.uint8)
        self._priors = np.asarray(m.hyper_priors, dtype=np.float64)
        self.n_checks = int(self._H.shape[0])
        self._n_mechanisms = int(self._H.shape[1])
        self._n_obs = int(self._L.shape[0])

        if self._n_mechanisms == 0:
            raise ValueError("DEM has no error mechanisms — nothing to decode")

        self._cluster = _HypergraphClusterExpansion(self._H, self._priors)
        self._bposd = BpOsdDecoder(
            self._H,
            priors=self._priors,
            max_iter=max_iter,
            osd_order=osd_order,
        )

    # -- constructors ------------------------------------------------------
    @classmethod
    def from_detector_error_model(cls, dem: Any, max_iter: int = 30, osd_order: int = 0) -> ColourCodeDecoder:
        """Build from an existing (undecomposed) DEM."""
        return cls(dem, max_iter=max_iter, osd_order=osd_order)

    @classmethod
    def from_stim_circuit(cls, circuit: Any, max_iter: int = 30, osd_order: int = 0) -> ColourCodeDecoder:
        """Build from a Stim circuit, deriving the DEM without decomposition.

        This is the recommended entry point: it cannot be handed a decomposed
        DEM by accident.
        """
        dem = circuit.detector_error_model(decompose_errors=False)
        return cls(dem, max_iter=max_iter, osd_order=osd_order)

    # -- decoding ----------------------------------------------------------
    def _predict(self, mechanisms: np.ndarray) -> np.ndarray:
        """Map a mechanism-support estimate to logical observable flips."""
        return ((mechanisms @ self._L.T) & 1).astype(np.uint8)

    def _pad_syndrome(self, syndrome: Any) -> np.ndarray:
        s = np.asarray(syndrome, dtype=np.uint8).reshape(-1)
        if s.shape[0] < self.n_checks:
            s = np.concatenate([s, np.zeros(self.n_checks - s.shape[0], np.uint8)])
        elif s.shape[0] > self.n_checks:
            raise ValueError(f"syndrome length {s.shape[0]} exceeds detector count {self.n_checks}")
        return s

    def _mechanism_decode(self, s: np.ndarray) -> np.ndarray:
        """Syndrome -> mechanism-support estimate, honouring ``self.method``."""
        if self.method == "bposd":
            return np.asarray(self._bposd.decode(s), dtype=np.uint8).reshape(-1)
        # cluster_bposd: verified cluster peel, then BP-OSD on the residual only.
        e_cluster, residual = self._cluster.decode(s)
        if not residual.any():
            return e_cluster
        e_bposd = np.asarray(self._bposd.decode(residual), dtype=np.uint8).reshape(-1)
        return (e_cluster ^ e_bposd).astype(np.uint8)

    def decode(self, syndrome) -> np.ndarray:
        """Decode one detector vector to an observable-flip prediction."""
        s = self._pad_syndrome(syndrome)
        e = self._mechanism_decode(s)
        return self._predict(e[None, :])[0]

    def decode_batch(self, syndromes) -> np.ndarray:
        """Decode a ``[batch, detectors]`` stack to ``[batch, observables]``.

        With ``method="bposd"`` this delegates to :meth:`BpOsdDecoder.batch_decode`
        (GPU batched-BP when a device is available). With ``"cluster_bposd"`` the
        per-shot cluster stage runs first and only the unresolved residual shots
        are batched through BP-OSD, keeping the GPU path for the hard shots.
        """
        arr = np.asarray(syndromes, dtype=np.uint8)
        if arr.ndim != 2:
            raise ValueError(f"syndromes must be 2D, got shape {arr.shape}")
        if arr.shape[1] < self.n_checks:
            pad = np.zeros((arr.shape[0], self.n_checks - arr.shape[1]), dtype=np.uint8)
            arr = np.concatenate([arr, pad], axis=1)
        elif arr.shape[1] > self.n_checks:
            raise ValueError(f"syndrome width {arr.shape[1]} exceeds detector count {self.n_checks}")
        if self.method == "bposd":
            e = np.asarray(self._bposd.batch_decode(arr), dtype=np.uint8)
            return self._predict(e)
        e = np.zeros((arr.shape[0], self._n_mechanisms), dtype=np.uint8)
        residual_rows: list[int] = []
        residual_syndromes: list[np.ndarray] = []
        for i in range(arr.shape[0]):
            e_c, res = self._cluster.decode(arr[i])
            e[i] = e_c
            if res.any():
                residual_rows.append(i)
                residual_syndromes.append(res)
        if residual_rows:
            res_stack = np.stack(residual_syndromes).astype(np.uint8)
            e_b = np.asarray(self._bposd.batch_decode(res_stack), dtype=np.uint8)
            for row, i in enumerate(residual_rows):
                e[i] ^= e_b[row]
        return self._predict(e)

    def decode_correction(self, syndrome) -> np.ndarray:
        """Explicit API: decode syndrome into physical mechanism correction vector."""
        return self._mechanism_decode(self._pad_syndrome(syndrome))

    def decode_observables(self, syndrome) -> np.ndarray:
        """Explicit API: decode syndrome into logical observable prediction vector."""
        return self.decode(syndrome)

    # -- accessors ---------------------------------------------------------
    @property
    def num_detectors(self) -> int:
        return self.n_checks

    @property
    def num_observables(self) -> int:
        return self._n_obs

    @property
    def num_mechanisms(self) -> int:
        return self._n_mechanisms

    def __repr__(self) -> str:  # pragma: no cover
        return (
            f"<ColourCodeDecoder detectors={self.n_checks} mechanisms={self._n_mechanisms} observables={self._n_obs}>"
        )


def colour_codes_from_dem(dem: Any, distance: int | None = None, max_iter: int = 30) -> ColourCodeDecoder:
    """Build a :class:`ColourCodeDecoder` from a DEM.

    ``distance`` is accepted and ignored — the DEM already determines the
    decoding problem completely. It is retained so existing call sites keep
    working.
    """
    return ColourCodeDecoder(dem, max_iter=max_iter)
