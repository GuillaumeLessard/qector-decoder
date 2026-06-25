"""Shared vectorised normalised min-sum belief propagation over a Tanner graph.

Used by both :mod:`qector_decoder_v3.belief_matching` (graphlike DEMs) and
:mod:`qector_decoder_v3.bposd` (general CSS LDPC). Operates on the sparse
check-matrix incidence; works for any ``H`` (no graphlike assumption).
"""

from __future__ import annotations

import numpy as np


def build_incidence(H: np.ndarray):
    """Return (ic, ie) — check and edge index per nonzero of H."""
    ic, ie = np.nonzero(H)
    return ic.astype(np.int64), ie.astype(np.int64)


def min_sum_bp(
    ic: np.ndarray,
    ie: np.ndarray,
    n_checks: int,
    n_edges: int,
    prior_llr: np.ndarray,
    syndrome: np.ndarray,
    max_iter: int,
    ms_scale: float = 1.0,
) -> np.ndarray:
    """Run normalised min-sum BP; return the posterior LLR per edge (length n_edges).

    Negative posterior LLR ⇒ that bit is likely 1 (an error).
    """
    M = ic.shape[0]
    inc_idx = np.arange(M)
    synd_sign = np.where(syndrome[ic] == 1, -1.0, 1.0)
    c2v = np.zeros(M, dtype=np.float64)
    S_e = np.zeros(n_edges, dtype=np.float64)

    for _ in range(max_iter):
        # variable -> check
        S_e.fill(0.0)
        np.add.at(S_e, ie, c2v)
        v2c = prior_llr[ie] + S_e[ie] - c2v

        av = np.abs(v2c)
        sgn = np.where(v2c < 0.0, -1.0, 1.0)

        # leave-one-out minima per check
        min1 = np.full(n_checks, np.inf)
        np.minimum.at(min1, ic, av)
        is_min = av >= min1[ic]
        tmp = np.where(is_min, inc_idx, M)
        amin = np.full(n_checks, M, dtype=np.int64)
        np.minimum.at(amin, ic, tmp)
        is_argmin = inc_idx == amin[ic]
        av_masked = np.where(is_argmin, np.inf, av)
        min2 = np.full(n_checks, np.inf)
        np.minimum.at(min2, ic, av_masked)
        loo = np.where(is_argmin, min2[ic], min1[ic])

        # leave-one-out sign
        negcount = np.zeros(n_checks, dtype=np.int64)
        np.add.at(negcount, ic, (sgn < 0).astype(np.int64))
        total_sign = np.where(negcount % 2 == 0, 1.0, -1.0)
        loo_sign = total_sign[ic] * sgn

        c2v = ms_scale * loo_sign * synd_sign * np.minimum(loo, 1e6)
        np.clip(c2v, -1e6, 1e6, out=c2v)

    S_e.fill(0.0)
    np.add.at(S_e, ie, c2v)
    return prior_llr + S_e


def sum_product_bp(
    ic: np.ndarray,
    ie: np.ndarray,
    n_checks: int,
    n_edges: int,
    prior_llr: np.ndarray,
    syndrome: np.ndarray,
    max_iter: int,
) -> np.ndarray:
    """Vectorised sum-product (belief-propagation) in the LLR domain.

    Uses the log-magnitude + sign form of the box-plus check update so the
    leave-one-out product is numerically stable. Returns posterior LLR per edge.
    """
    M = ic.shape[0]
    synd_sign = np.where(syndrome[ic] == 1, -1.0, 1.0)
    c2v = np.zeros(M, dtype=np.float64)
    S_e = np.zeros(n_edges, dtype=np.float64)
    # eps guards: tanh saturates to ±1 at large LLR; clip before log prevents
    # divide-by-zero / -inf from log(0). All such cases are numerically ±inf
    # LLR (certainty), handled correctly by the subsequent clip.
    eps = 1e-12

    with np.errstate(divide="ignore", invalid="ignore"):
        for _ in range(max_iter):
            S_e.fill(0.0)
            np.add.at(S_e, ie, c2v)
            v2c = prior_llr[ie] + S_e[ie] - c2v

            t = np.tanh(np.clip(0.5 * v2c, -30.0, 30.0))
            t = np.clip(t, -1.0 + eps, 1.0 - eps)
            sgn = np.where(t < 0.0, -1.0, 1.0)
            logabs = np.log(np.abs(t))  # safe: t clipped away from 0

            sum_log = np.zeros(n_checks, dtype=np.float64)
            np.add.at(sum_log, ic, logabs)
            negcount = np.zeros(n_checks, dtype=np.int64)
            np.add.at(negcount, ic, (sgn < 0).astype(np.int64))
            total_sign = np.where(negcount % 2 == 0, 1.0, -1.0)

            loo_log = sum_log[ic] - logabs
            loo_sign = total_sign[ic] * sgn
            loo = loo_sign * np.exp(np.clip(loo_log, -60.0, 0.0))
            loo = np.clip(loo, -1.0 + eps, 1.0 - eps)
            c2v = synd_sign * 2.0 * np.arctanh(loo)
            np.clip(c2v, -1e6, 1e6, out=c2v)

    S_e.fill(0.0)
    np.add.at(S_e, ie, c2v)
    return prior_llr + S_e
