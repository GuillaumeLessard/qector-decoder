"""
qector_decoder_v3.bposd - belief propagation + ordered-statistics decoding.

A self-contained Python BP-OSD for general (LDPC / non-graphlike) CSS codes,
built on the shared vectorised min-sum BP. When BP alone explains the syndrome it
returns immediately; otherwise an **exact GF(2) ordered-statistics** post-process
produces a guaranteed syndrome-consistent correction using BP's reliabilities:
OSD-0 when ``osd_order=0``, and a formal **combination-sweep CS-OSD(λ, w)**
(Panteleev & Kalachev, arXiv:1904.02703; same sweep structure as the reference
``ldpc`` package's ``osd_cs``) when ``osd_order >= 1``. BP supports optional LLR
message **damping** (``damping`` parameter) against loopy-graph oscillation.

This is the path for codes matching cannot handle - bivariate-bicycle,
hypergraph-product, bicycle. It is cross-validated against the reference ``ldpc``
package's BP-OSD in the test suite.

Example
-------
>>> from qector_decoder_v3 import codes
>>> from qector_decoder_v3.bposd import BpOsdDecoder
>>> cx, cz = codes.bivariate_bicycle_code(6, 6, [("x", 3), ("y", 1), ("y", 2)], [("y", 3), ("x", 1), ("x", 2)])
>>> dec = BpOsdDecoder(cx.parity_check_matrix(), error_rate=0.05, osd_order=0)
>>> correction = dec.decode(syndrome)
"""

from __future__ import annotations

from typing import Any, cast

import numpy as np

from . import BPOSDDecoder as GraphBPOSDDecoder
from . import gpu_backend as _gb
from ._bp_core import _check_damping, build_incidence, min_sum_bp, sum_product_bp
from .codes import _to_dense_binary

__all__ = ["BpOsdDecoder", "GraphBPOSDDecoder", "MatrixBPOSDDecoder"]


class BpOsdDecoder:
    """BP + ordered-statistics decoder for general GF(2) check matrices.

    Takes a **dense GF(2) parity-check matrix** ``H`` (shape ``(n_checks,
    n_qubits)``). For the adjacency-list variant see
    :class:`~qector_decoder_v3.GraphBPOSDDecoder` (native Rust, check-to-qubits
    list). The near-identical spelling of the two names is a known trap; prefer
    :class:`MatrixBPOSDDecoder` (this class) going forward.
    """

    def __init__(
        self,
        H: Any,
        error_rate: float = 0.05,
        priors: Any = None,
        max_iter: int = 30,
        ms_scale: float = 1.0,
        osd_order: int = 0,
        bp_method: str = "sum_product",
        use_gpu: Any = None,
        max_latency_ms: float | None = None,
        damping: float = 0.0,
        osd_lambda: Any = None,
    ):
        self.H = _to_dense_binary(H)
        if self.H.ndim != 2:
            raise ValueError(f"H must be 2D, got {self.H.shape}")
        self.n_checks, self.n_qubits = (int(self.H.shape[0]), int(self.H.shape[1]))
        self.ic, self.ie = build_incidence(self.H)
        p: np.ndarray
        if priors is None:
            p = np.full(self.n_qubits, float(error_rate), dtype=np.float64)
        else:
            p = np.asarray(priors, dtype=np.float64).reshape(-1)
        p = np.clip(p, 1e-15, 1 - 1e-15)
        self.priors = p
        self.prior_llr = np.log((1.0 - p) / p)
        self.max_iter = int(max_iter)
        self.ms_scale = float(ms_scale)
        self.osd_order = int(osd_order)
        if bp_method not in ("sum_product", "min_sum", "relay"):
            raise ValueError("bp_method must be 'sum_product', 'min_sum', or 'relay'")
        self.bp_method = bp_method
        self.damping = _check_damping(damping)
        if osd_lambda is None:
            self.osd_lambda: int | None = None
        else:
            self.osd_lambda = max(1, int(osd_lambda))
        self.max_latency_ms = float(max_latency_ms) if max_latency_ms is not None else None
        # GPU batched-BP policy. ``None`` -> auto (use the GPU iff one is usable);
        # ``True``/``False`` force the choice. The single-shot ``decode`` path is
        # never affected; only ``batch_decode`` consults this.
        self.use_gpu = use_gpu
        self._batched = None  # lazily built BatchedBpDecoder (device-resident)

    # -- public ------------------------------------------------------------
    def decode(self, syndrome) -> np.ndarray:
        s: np.ndarray = np.asarray(syndrome, dtype=np.uint8).reshape(-1)
        if s.shape[0] < self.n_checks:
            s = np.concatenate([s, np.zeros(self.n_checks - s.shape[0], np.uint8)])

        if self.max_latency_ms is not None:
            import time as _time

            t_start = _time.perf_counter()
            max_seconds = self.max_latency_ms / 1000.0

        actual_iter = min(self.max_iter, 50)
        if self.bp_method == "relay":
            from . import BPOSDDecoder as _RustBPOSD

            rust_dec = _RustBPOSD(
                self.check_to_qubits(), self.n_qubits, error_rate=float(self.priors.mean()), bp_method="relay"
            )
            return rust_dec.decode(s)
        elif self.bp_method == "sum_product":
            posterior = sum_product_bp(
                self.ic,
                self.ie,
                self.n_checks,
                self.n_qubits,
                self.prior_llr,
                s,
                actual_iter,
                damping=self.damping,
            )
        else:
            posterior = min_sum_bp(
                self.ic,
                self.ie,
                self.n_checks,
                self.n_qubits,
                self.prior_llr,
                s,
                actual_iter,
                self.ms_scale,
                damping=self.damping,
            )

        if self.max_latency_ms is not None and (_time.perf_counter() - t_start) > max_seconds:
            from . import UnionFindDecoder

            checks = [sorted(int(c) for c in np.nonzero(self.H[r])[0]) for r in range(self.n_checks)]
            uf = UnionFindDecoder(checks, self.n_qubits)
            return np.asarray(uf.decode(s), dtype=np.uint8).reshape(-1)

        hard = (posterior < 0.0).astype(np.uint8)
        if np.array_equal((self.H @ hard) & 1, s):
            return hard
        return self._osd(s, posterior)

    def decode_correction(self, syndrome) -> np.ndarray:
        """Explicit API: decode syndrome into physical correction vector of length n_qubits."""
        return self.decode(syndrome)

    def decode_observables(self, syndrome, observables_matrix: Any = None) -> np.ndarray:
        """Explicit API: decode syndrome into logical observable prediction vector."""
        corr = self.decode(syndrome)
        if observables_matrix is not None:
            L = np.asarray(observables_matrix, dtype=np.uint8)
            return (L @ corr.T) % 2
        return corr

    def batch_decode(self, syndromes) -> np.ndarray:
        """Decode a stack of syndromes ``[batch, n_checks]`` to ``[batch, n_qubits]``.

        When a CUDA device is usable (and ``use_gpu`` is not ``False``), the BP stage
        for the whole batch is run **once** on the GPU via
        :class:`qector_decoder_v3.bp_cupy.BatchedBpDecoder`. Shots that BP already
        explains (``H @ c == s``) are returned directly (the OSD-0 fast path - no
        ordered-statistics work); only the residual non-converged shots are sent
        through the exact GF(2) OSD post-process, seeded with that shot's GPU BP
        reliabilities. The result is bit-for-bit a valid BP-OSD output and matches
        the CPU path's correctness contract (every row satisfies its syndrome).

        Falls back to the per-shot CPU path on any GPU error, and is used
        automatically when no device is present, so behaviour is unchanged on
        CPU-only machines.
        """
        arr = np.asarray(syndromes, dtype=np.uint8)
        if arr.ndim != 2:
            raise ValueError(f"syndromes must be 2D, got {arr.shape}")
        if self._gpu_enabled():
            try:
                return self._gpu_batch_decode(arr)
            except (ImportError, OSError, AttributeError, RuntimeError):  # pragma: no cover - defensive GPU fallback
                _gb.note_fallback()
        return np.stack([self.decode(arr[i]) for i in range(arr.shape[0])]).astype(np.uint8)

    # -- GPU batch path ----------------------------------------------------
    def _gpu_enabled(self) -> bool:
        """Return ``True`` iff the GPU batched-BP path should be used right now."""
        if self.use_gpu is False:
            return False
        return bool(_gb.gpu_available() and _gb.get_prefer_gpu())

    def _get_batched(self):
        """Return the cached device-resident :class:`BatchedBpDecoder` (built once)."""
        if self._batched is None:
            from .bp_cupy import BatchedBpDecoder

            self._batched = BatchedBpDecoder(
                self.H,
                priors=self.priors,
                max_iter=self.max_iter,
                alpha=self.ms_scale,
                bp_method=self.bp_method,
                prefer_gpu=True,
            )
        return self._batched

    def _gpu_batch_decode(self, arr: np.ndarray) -> np.ndarray:
        """Batched GPU BP + selective OSD; every returned row satisfies its syndrome."""
        if arr.shape[1] < self.n_checks:
            pad = np.zeros((arr.shape[0], self.n_checks - arr.shape[1]), dtype=np.uint8)
            arr = np.concatenate([arr, pad], axis=1)
        bd = self._get_batched()
        corr, converged, llr = bd.decode_batch(arr, return_llr=True, early_stop=True)
        out = corr.astype(np.uint8, copy=True)
        # OSD-0 fast path: shots BP already explains skip OSD entirely; only the
        # residual non-converged shots get the exact GF(2) ordered-statistics solve,
        # seeded with the GPU BP posterior so OSD inherits BP's reliabilities.
        for i in np.nonzero(~converged)[0]:
            s_i = arr[i].reshape(-1).astype(np.uint8)
            out[i] = self._osd(s_i, llr[i])
        return cast(np.ndarray, out.astype(np.uint8))

    @property
    def n_qubits_(self) -> int:
        return self.n_qubits

    # -- OSD ---------------------------------------------------------------
    def _osd(self, s: np.ndarray, posterior: np.ndarray) -> np.ndarray:
        hard = (posterior < 0.0).astype(np.uint8)
        rel = np.abs(posterior)
        # Least-reliable columns first: the basis (information set) is built from
        # the bits BP is least sure about, so the syndrome's residual errors land
        # there; the most-reliable bits are "free" and kept at BP's hard decision.
        order = np.argsort(rel)
        x0, pivot_cols = _gf2_osd_solve(self.H, s, order, hard)
        if self.osd_order <= 0:
            return x0
        return self._cs_osd_sweep(s, order, hard, rel, x0, pivot_cols)

    def _cs_osd_sweep(
        self,
        s: np.ndarray,
        order: np.ndarray,
        hard: np.ndarray,
        rel: np.ndarray,
        x0: np.ndarray,
        pivot_cols: np.ndarray,
    ) -> np.ndarray:
        # Formal combination_sweep OSD-E (CS-OSD) implementation
        n = self.n_qubits
        is_pivot = np.zeros(n, dtype=bool)
        is_pivot[np.asarray(pivot_cols, dtype=np.int64)] = True
        free = np.nonzero(~is_pivot)[0]
        if free.size == 0:
            return x0
        lam = self.osd_lambda if self.osd_lambda is not None else 24
        lam = min(int(lam), free.size)
        sweep = free[np.argsort(rel[free], kind="stable")[:lam]]
        depth = min(self.osd_order, 8)

        target = self.prior_llr
        best = x0
        best_key = (float(target @ x0), int(x0.sum()))

        # Formal combination_sweep and OSD-E logic without itertools brute force
        def combination_sweep(elements, k):
            if k == 0:
                yield []
            else:
                stack = [(0, [])]
                while stack:
                    idx, current = stack.pop()
                    if len(current) == k:
                        yield current
                    elif len(current) + len(elements) - idx >= k:
                        stack.append((idx + 1, current))
                        stack.append((idx + 1, current + [elements[idx]]))

        for r in range(1, depth + 1):
            improved = False
            for combo in combination_sweep(sweep.tolist(), r):
                flipped = hard.copy()
                flipped[combo] ^= 1
                x, _ = _gf2_osd_solve(self.H, s, order, flipped)
                if not np.array_equal((self.H @ x) & 1, s):
                    continue
                key = (float(target @ x), int(x.sum()))
                if key < best_key:
                    best_key, best, improved = key, x, True
            if not improved:
                break
        return best


# ---------------------------------------------------------------------------
# GF(2) ordered-statistics solve
# ---------------------------------------------------------------------------
def _gf2_osd_solve(H: np.ndarray, s: np.ndarray, order: np.ndarray, hard: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """OSD-0 solve. ``order`` lists columns least-reliable first; the first
    rank(H) independent of them form the basis, the rest (free) take their BP hard
    decision, and the basis is solved so ``H x == s (mod 2)``.

    Returns (x, pivot_columns_in_original_indexing).
    """
    r, ncol = H.shape
    M = H[:, order].copy().astype(np.uint8) % 2
    pivots_perm: list[int] = []
    rr = 0
    for c in range(ncol):
        nz = np.nonzero(M[rr:, c])[0]
        if nz.size == 0:
            continue
        p = rr + int(nz[0])
        if p != rr:
            M[[rr, p]] = M[[p, rr]]
        for i in np.nonzero(M[:, c])[0]:
            if i != rr:
                M[i] ^= M[rr]
        pivots_perm.append(c)
        rr += 1
        if rr == r:
            break

    pivset = set(pivots_perm)
    x_perm = np.zeros(ncol, dtype=np.uint8)
    for c in range(ncol):
        if c not in pivset:
            x_perm[c] = hard[order[c]]  # free bits = BP hard decision

    # residual syndrome from the free assignment
    x = np.zeros(ncol, dtype=np.uint8)
    x[order] = x_perm
    s_eff = (s ^ ((H @ x) & 1)) & 1

    # solve the basis (pivot) columns for s_eff
    M2 = H[:, order].copy().astype(np.uint8) % 2
    rhs = s_eff.copy().astype(np.uint8) % 2
    rr = 0
    pivot_rows: list[tuple] = []
    for c in pivots_perm:
        nz = np.nonzero(M2[rr:, c])[0]
        if nz.size == 0:
            continue
        p = rr + int(nz[0])
        if p != rr:
            M2[[rr, p]] = M2[[p, rr]]
            rhs[[rr, p]] = rhs[[p, rr]]
        for i in np.nonzero(M2[:, c])[0]:
            if i != rr:
                M2[i] ^= M2[rr]
                rhs[i] ^= rhs[rr]
        pivot_rows.append((c, rr))
        rr += 1
    for c, row in pivot_rows:
        x_perm[c] ^= rhs[row]

    x = np.zeros(ncol, dtype=np.uint8)
    x[order] = x_perm
    pivot_cols = order[pivots_perm]
    return x, pivot_cols


class MatrixBPOSDDecoder(BpOsdDecoder):
    """Clear-name alias for :class:`BpOsdDecoder` (dense GF(2) check matrix).

    Prefer this name over ``BpOsdDecoder``: it distinguishes the dense-matrix
    constructor from the native :class:`~qector_decoder_v3.GraphBPOSDDecoder`
    (check-to-qubits adjacency list), which differ wildly (matrix vs graph
    wiring) while spelling nearly identically. ``BpOsdDecoder`` remains
    importable unchanged.
    """
