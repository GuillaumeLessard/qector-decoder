"""
qector_decoder_v3.bp_cupy - batched, GPU-resident belief propagation.

This module is the batched counterpart of the single-shot, edge-vectorised BP in
:mod:`qector_decoder_v3._bp_core`. Where ``_bp_core`` runs one syndrome at a time
over the Tanner-graph edge lists, :class:`BatchedBpDecoder` runs a whole *stack* of
syndromes ``[batch, n_checks]`` in lock-step, carrying the batch as a trailing
array axis so every variable→check and check→variable update is a single
vectorised operation over ``[n_edges, batch]`` message arrays.

Design goals
------------
* **One code path, two backends.** All arithmetic goes through the array module
  returned by :func:`qector_decoder_v3.gpu_backend.xp` (CuPy on a usable CUDA
  device, NumPy otherwise). CuPy 14.x implements ``add.at`` / ``minimum.at`` with
  exactly NumPy's axis-0 scatter semantics, so the *same* source runs on either
  backend with no per-backend branching in the hot loop.
* **GPU-resident.** The Tanner incidence (``ic``/``ie``), the prior LLRs and all
  message buffers are built **once** on the device in :meth:`__init__`. Per call,
  only the syndrome stack is moved host→device and only the corrections (and
  optionally the posterior LLRs / convergence mask) are moved device→host - the
  minimal possible traffic across the PCIe boundary.
* **Faithful semantics.** The recurrences are a line-for-line batched port of
  :func:`_bp_core.min_sum_bp` (normalised min-sum, ``alpha`` == ``ms_scale``) and
  :func:`_bp_core.sum_product_bp` (box-plus / tanh sum-product). With CuPy absent
  the NumPy path is **bit-identical** to ``_bp_core`` for a single shot, because
  ``np.add.at`` accumulates per column in the same edge order.

The validity bar for any returned correction ``c`` is the usual GF(2) syndrome
equation ``H @ c == s (mod 2)``; the returned convergence mask reports exactly the
shots for which BP alone already meets it (the rest are handed to OSD by
:mod:`qector_decoder_v3.bposd`).

Example
-------
>>> from qector_decoder_v3 import codes
>>> from qector_decoder_v3.bp_cupy import batched_bp_decode
>>> cx, _ = codes.bivariate_bicycle_code(6, 6, [("x", 3), ("y", 1), ("y", 2)], [("y", 3), ("x", 1), ("x", 2)])
>>> H = cx.parity_check_matrix()
>>> import numpy as np
>>> errs = (np.random.default_rng(0).random((128, 72)) < 0.05).astype(np.uint8)
>>> syns = (errs @ H.T) & 1
>>> corr, converged = batched_bp_decode(H, syns, error_rate=0.05, max_iter=30)
>>> bool(np.all((H @ corr[converged].T).T % 2 == syns[converged]))
True
"""

from __future__ import annotations

from typing import Any

import numpy as np

from . import gpu_backend as gb
from ._bp_core import build_incidence
from .codes import _to_dense_binary

__all__ = ["BatchedBpDecoder", "batched_bp_decode"]

# Clamp used for min-sum messages, matching ``_bp_core.min_sum_bp``.
_LLR_CLAMP = 1.0e6
# Numerical floor for the tanh/arctanh box-plus update in sum-product.
_SP_EPS = 1.0e-12


class BatchedBpDecoder:
    """Batched belief-propagation decoder over a fixed GF(2) check matrix ``H``.

    The decoder is built once for a given ``H`` (and prior model) and then decodes
    arbitrarily many syndrome stacks. All device-resident state - the Tanner
    incidence, the prior log-likelihood ratios and the reusable scratch buffers -
    is allocated in :meth:`__init__`, so steady-state decoding moves only the
    syndromes in and the corrections out.

    Parameters
    ----------
    H:
        Parity-check matrix, ``[n_checks, n_qubits]``. Dense array-like or any
        object exposing ``toarray`` (e.g. ``scipy.sparse``); reduced mod 2.
    error_rate:
        Uniform per-qubit prior error probability, used when ``priors`` is ``None``.
    priors:
        Optional per-qubit prior error probabilities (length ``n_qubits``). Overrides
        ``error_rate`` when given.
    max_iter:
        Maximum number of BP message-passing iterations.
    alpha:
        Normalised min-sum scale factor (the ``ms_scale`` of ``_bp_core``). Ignored
        by the sum-product update, which has no scale parameter.
    bp_method:
        ``"min_sum"`` (normalised min-sum) or ``"sum_product"`` (box-plus). The
        default matches the most common GPU use; :class:`bposd.BpOsdDecoder` passes
        its own configured method through.
    prefer_gpu:
        Request the GPU backend. The GPU is used only when this is true **and** CuPy
        is importable **and** a CUDA device is usable **and** the global preference
        (:func:`gpu_backend.get_prefer_gpu`) is on; otherwise the NumPy path is used
        transparently.

    Attributes
    ----------
    on_gpu:
        ``True`` iff this instance resolved to the CuPy backend at construction.
    """

    def __init__(
        self,
        H: Any,
        *,
        error_rate: float = 0.05,
        priors: Any = None,
        max_iter: int = 30,
        alpha: float = 1.0,
        bp_method: str = "min_sum",
        prefer_gpu: bool = True,
    ) -> None:
        self.H = _to_dense_binary(H)
        if self.H.ndim != 2:
            raise ValueError(f"H must be 2D, got shape {self.H.shape}")
        self.n_checks, self.n_qubits = (int(self.H.shape[0]), int(self.H.shape[1]))
        if bp_method not in ("sum_product", "min_sum"):
            raise ValueError("bp_method must be 'sum_product' or 'min_sum'")
        self.bp_method = bp_method
        self.max_iter = int(max_iter)
        self.alpha = float(alpha)

        # Host-side prior model (kept for introspection / reproducibility).
        p: np.ndarray
        if priors is None:
            p = np.full(self.n_qubits, float(error_rate), dtype=np.float64)
        else:
            p = np.asarray(priors, dtype=np.float64).reshape(-1)
            if p.shape[0] != self.n_qubits:
                raise ValueError(f"priors length {p.shape[0]} != n_qubits {self.n_qubits}")
        p = np.clip(p, 1e-15, 1.0 - 1e-15)
        self.priors = p
        prior_llr_host = np.log((1.0 - p) / p)

        ic_host, ie_host = build_incidence(self.H)
        self.n_edges = int(ic_host.shape[0])

        # Resolve the backend ONCE. ``has_cupy()`` is consulted explicitly so a test
        # monkeypatching it to False forces the NumPy path even on a GPU box.
        self.on_gpu = bool(prefer_gpu and gb.get_prefer_gpu() and gb.has_cupy() and gb.gpu_available())
        if self.on_gpu:
            try:
                import cupy as _cp  # type: ignore[import-not-found]

                self._xp = _cp
            except RuntimeError:  # pragma: no cover - import guarded
                self.on_gpu = False
                self._xp = np
        else:
            self._xp = np

        xp = self._xp
        # Device-resident, batch-independent state (built once).
        self._ic = xp.asarray(ic_host.astype(np.int64))
        self._ie = xp.asarray(ie_host.astype(np.int64))
        self._prior_llr = xp.asarray(prior_llr_host.astype(np.float64))
        # Per-edge prior LLR, shaped for broadcasting against [n_edges, batch].
        self._prior_edge = self._prior_llr[self._ie][:, None]
        self._inc_idx = xp.arange(self.n_edges, dtype=np.int64)

    # -- public ------------------------------------------------------------
    def decode_batch(
        self,
        syndromes: Any,
        *,
        return_llr: bool = False,
        early_stop: bool = True,
    ) -> tuple[np.ndarray, np.ndarray] | tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Decode a stack of syndromes with batched BP.

        Parameters
        ----------
        syndromes:
            ``[batch, n_checks]`` (or a single ``[n_checks]`` row) of 0/1 syndrome
            bits. Rows shorter than ``n_checks`` are zero-padded.
        return_llr:
            Also return the posterior per-qubit LLRs ``[batch, n_qubits]`` (BP
            reliabilities, suitable for seeding OSD).
        early_stop:
            Stop iterating once **every** shot in the batch satisfies its syndrome.
            Disable for a fixed-iteration schedule that is bit-identical to
            ``_bp_core`` on the NumPy backend (used by the parity tests).

        Returns
        -------
        corrections:
            ``[batch, n_qubits]`` uint8 hard decisions (BP marginals thresholded at
            LLR < 0). Converged rows satisfy ``H @ c == s (mod 2)``.
        converged:
            ``[batch]`` bool mask; ``True`` where the correction meets the syndrome.
        posterior:
            Only when ``return_llr``: ``[batch, n_qubits]`` float64 posterior LLRs.
        """
        s = np.asarray(syndromes, dtype=np.uint8)
        if s.ndim == 1:
            s = s[None, :]
        if s.ndim != 2:
            raise ValueError(f"syndromes must be 1D or 2D, got shape {s.shape}")
        if s.shape[1] < self.n_checks:
            pad = np.zeros((s.shape[0], self.n_checks - s.shape[1]), dtype=np.uint8)
            s = np.concatenate([s, pad], axis=1)
        elif s.shape[1] > self.n_checks:
            raise ValueError(f"syndrome width {s.shape[1]} exceeds n_checks {self.n_checks}")
        batch = int(s.shape[0])

        xp = self._xp
        # Host -> device: the syndrome stack, transposed to [n_checks, batch] so the
        # batch is the trailing (scatter-friendly) axis.
        synd_i = xp.asarray(np.ascontiguousarray(s.T.astype(np.int8)))
        if self.on_gpu:
            gb.TELEMETRY["h2d"] += 1
            gb.note_gpu_call()

        if self.bp_method == "min_sum":
            posterior = self._run_min_sum(synd_i, batch, early_stop)
        else:
            posterior = self._run_sum_product(synd_i, batch, early_stop)

        hard = (posterior < 0.0).astype(xp.uint8)  # [n_qubits, batch]
        pred = self._predict_syndrome(hard, batch)  # [n_checks, batch]
        converged = xp.all(pred == synd_i, axis=0)  # [batch]

        corrections = gb.to_host(hard.T).astype(np.uint8)
        converged_host = gb.to_host(converged).astype(bool)
        if return_llr:
            posterior_host = gb.to_host(posterior.T).astype(np.float64)
            return corrections, converged_host, posterior_host
        return corrections, converged_host

    # -- internal kernels --------------------------------------------------
    def _predict_syndrome(self, hard: Any, batch: int) -> Any:
        """Return ``(H @ hard) mod 2`` as ``[n_checks, batch]`` via edge scatter."""
        xp = self._xp
        pred = xp.zeros((self.n_checks, batch), dtype=xp.int64)
        xp.add.at(pred, self._ic, hard[self._ie].astype(xp.int64))
        return pred & 1

    def _all_converged(self, c2v: Any, synd_i: Any, batch: int) -> bool:
        """Cheap convergence probe: does the current posterior satisfy every shot?"""
        xp = self._xp
        s_acc = xp.zeros((self.n_qubits, batch), dtype=xp.float64)
        xp.add.at(s_acc, self._ie, c2v)
        hard = ((self._prior_llr[:, None] + s_acc) < 0.0).astype(xp.uint8)
        pred = self._predict_syndrome(hard, batch)
        return bool(xp.all(pred == synd_i))

    def _run_min_sum(self, synd_i: Any, batch: int, early_stop: bool) -> Any:
        """Batched normalised min-sum BP; returns posterior LLR ``[n_qubits, batch]``.

        Line-for-line batched port of :func:`_bp_core.min_sum_bp` with the batch as
        the trailing axis. ``self.alpha`` is the ``ms_scale`` normalisation factor.
        """
        xp = self._xp
        ic, ie = self._ic, self._ie
        nC, nQ, M = self.n_checks, self.n_qubits, self.n_edges
        inc_idx = self._inc_idx[:, None]  # [M, 1]
        prior_edge = self._prior_edge  # [M, 1]

        # syndrome sign per edge: -1 where the incident check fired, else +1.
        synd_sign = xp.where(synd_i[ic] == 1, -1.0, 1.0)  # [M, batch]

        c2v = xp.zeros((M, batch), dtype=xp.float64)
        S_e = xp.zeros((nQ, batch), dtype=xp.float64)

        for it in range(self.max_iter):
            # variable -> check (leave-one-out sum of incoming check messages)
            S_e.fill(0.0)
            xp.add.at(S_e, ie, c2v)
            if early_stop and it > 0 and self._all_converged(c2v, synd_i, batch):
                break
            v2c = prior_edge + S_e[ie] - c2v  # [M, batch]

            av = xp.abs(v2c)
            sgn = xp.where(v2c < 0.0, -1.0, 1.0)

            # leave-one-out minima per check (two-smallest trick)
            min1 = xp.full((nC, batch), xp.inf, dtype=xp.float64)
            xp.minimum.at(min1, ic, av)
            is_min = av >= min1[ic]
            tmp = xp.where(is_min, inc_idx, M)
            amin = xp.full((nC, batch), M, dtype=xp.int64)
            xp.minimum.at(amin, ic, tmp)
            is_argmin = inc_idx == amin[ic]
            av_masked = xp.where(is_argmin, xp.inf, av)
            min2 = xp.full((nC, batch), xp.inf, dtype=xp.float64)
            xp.minimum.at(min2, ic, av_masked)
            loo = xp.where(is_argmin, min2[ic], min1[ic])

            # leave-one-out sign (parity of the other incident signs)
            negcount = xp.zeros((nC, batch), dtype=xp.int64)
            xp.add.at(negcount, ic, (sgn < 0.0).astype(xp.int64))
            total_sign = xp.where(negcount % 2 == 0, 1.0, -1.0)
            loo_sign = total_sign[ic] * sgn

            c2v = self.alpha * loo_sign * synd_sign * xp.minimum(loo, _LLR_CLAMP)
            xp.clip(c2v, -_LLR_CLAMP, _LLR_CLAMP, out=c2v)

        S_e.fill(0.0)
        xp.add.at(S_e, ie, c2v)
        return self._prior_llr[:, None] + S_e

    def _run_sum_product(self, synd_i: Any, batch: int, early_stop: bool) -> Any:
        """Batched sum-product (box-plus) BP; returns posterior LLR ``[n_qubits, batch]``.

        Line-for-line batched port of :func:`_bp_core.sum_product_bp`, using the
        log-magnitude + sign form of the check update for numerical stability.
        """
        xp = self._xp
        ic, ie = self._ic, self._ie
        nC, nQ, M = self.n_checks, self.n_qubits, self.n_edges
        prior_edge = self._prior_edge  # [M, 1]
        eps = _SP_EPS

        synd_sign = xp.where(synd_i[ic] == 1, -1.0, 1.0)  # [M, batch]
        c2v = xp.zeros((M, batch), dtype=xp.float64)
        S_e = xp.zeros((nQ, batch), dtype=xp.float64)

        for it in range(self.max_iter):
            S_e.fill(0.0)
            xp.add.at(S_e, ie, c2v)
            if early_stop and it > 0 and self._all_converged(c2v, synd_i, batch):
                break
            v2c = prior_edge + S_e[ie] - c2v

            t = xp.tanh(xp.clip(0.5 * v2c, -30.0, 30.0))
            t = xp.clip(t, -1.0 + eps, 1.0 - eps)
            sgn = xp.where(t < 0.0, -1.0, 1.0)
            logabs = xp.log(xp.abs(t))

            sum_log = xp.zeros((nC, batch), dtype=xp.float64)
            xp.add.at(sum_log, ic, logabs)
            negcount = xp.zeros((nC, batch), dtype=xp.int64)
            xp.add.at(negcount, ic, (sgn < 0.0).astype(xp.int64))
            total_sign = xp.where(negcount % 2 == 0, 1.0, -1.0)

            loo_log = sum_log[ic] - logabs
            loo_sign = total_sign[ic] * sgn
            loo = loo_sign * xp.exp(xp.clip(loo_log, -60.0, 0.0))
            loo = xp.clip(loo, -1.0 + eps, 1.0 - eps)
            c2v = synd_sign * 2.0 * xp.arctanh(loo)
            xp.clip(c2v, -_LLR_CLAMP, _LLR_CLAMP, out=c2v)

        S_e.fill(0.0)
        xp.add.at(S_e, ie, c2v)
        return self._prior_llr[:, None] + S_e


def batched_bp_decode(
    H: Any,
    syndromes: Any,
    *,
    error_rate: float = 0.05,
    priors: Any = None,
    max_iter: int = 30,
    alpha: float = 1.0,
    bp_method: str = "min_sum",
    prefer_gpu: bool = True,
    early_stop: bool = True,
    return_llr: bool = False,
) -> tuple[np.ndarray, np.ndarray] | tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Decode a stack of syndromes with batched belief propagation.

    Convenience wrapper that builds a one-shot :class:`BatchedBpDecoder` and decodes
    a single syndrome stack. For repeated decoding of the same code, construct a
    :class:`BatchedBpDecoder` once (its device-resident state is then reused) and
    call :meth:`BatchedBpDecoder.decode_batch`.

    Parameters
    ----------
    H:
        Parity-check matrix ``[n_checks, n_qubits]``.
    syndromes:
        ``[batch, n_checks]`` (or single ``[n_checks]``) syndrome bits.
    error_rate, priors, max_iter, alpha, bp_method, prefer_gpu:
        See :class:`BatchedBpDecoder`.
    early_stop:
        Stop once all shots satisfy their syndrome (see
        :meth:`BatchedBpDecoder.decode_batch`).
    return_llr:
        Also return posterior per-qubit LLRs.

    Returns
    -------
    See :meth:`BatchedBpDecoder.decode_batch`.
    """
    dec = BatchedBpDecoder(
        H,
        error_rate=error_rate,
        priors=priors,
        max_iter=max_iter,
        alpha=alpha,
        bp_method=bp_method,
        prefer_gpu=prefer_gpu,
    )
    return dec.decode_batch(syndromes, return_llr=return_llr, early_stop=early_stop)
