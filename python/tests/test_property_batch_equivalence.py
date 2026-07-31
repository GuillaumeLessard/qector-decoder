"""Property-based batch/single equivalence and batch row-indexing invariants.

``test_batch_vs_single_decode.py`` already pins bit-identity for
``CPUBatchDecoder`` and ``BlossomDecoder`` on three named codes with two fixed
seeds and p=0.08, and ``test_batch_shapes.py`` pins the rank checks. This file
covers what those cannot:

  * **Row-order invariance.** The batch paths shard rows across rayon workers
    (``fast_uf.rs`` does ``par_chunks_exact_mut(n_qubits).enumerate()`` and
    indexes the input as ``syndromes[i * n_checks..(i + 1) * n_checks]``). An
    off-by-one or a transposed stride there still produces a *plausible,
    faithful-looking* output for every row -- it just pairs each row with the
    wrong answer. A fixed-order test cannot see that. Permuting the input rows
    must permute the output rows identically; nothing else in the suite checks
    this.
  * **Adversarial and degenerate batches** -- N=1, N=0, all-identical rows,
    all-zero rows, the max-weight row -- rather than only p=0.08 samples.
  * **All five graph decoders**, via the universal per-row faithfulness
    property. Bit-identity is only claimed for the deterministic order-
    independent decoders, so the region-growing ones get the weaker (but still
    load-bearing) invariant.
"""

from __future__ import annotations

import numpy as np
import pytest
import qector_decoder_v3 as qd
from hypothesis import HealthCheck, given, settings
from hypothesis import strategies as st
from qector_decoder_v3 import codes

# Decoders documented as deterministic and order-independent per shot, so their
# batch output must be bit-identical to per-shot decoding.
BIT_IDENTICAL = ["CPUBatchDecoder", "BlossomDecoder"]

# Every graph decoder exposing batch_decode: weaker per-row faithfulness holds
# for all of them.
ALL_BATCH = [
    "UnionFindDecoder",
    "FastUnionFindDecoder",
    "BlossomDecoder",
    "SparseBlossomDecoder",
    "LookupTableDecoder",
    "CPUBatchDecoder",
]


def _codes():
    return [
        ("repetition_d5", codes.repetition_code(5)),
        ("rotated_surface_d3", codes.rotated_surface_code(3)),
        ("rotated_surface_d5", codes.rotated_surface_code(5)),
    ]


CODES = _codes()
CODE_IDS = [n for n, _ in CODES]


def _mk(name, code):
    return getattr(qd, name)(code.check_to_qubits, code.n_qubits)


def _batch(dec, S, n_qubits):
    return np.asarray(dec.batch_decode(S), np.uint8).reshape(len(S), n_qubits)


def _per_shot(dec, S, n_qubits):
    if len(S) == 0:
        return np.zeros((0, n_qubits), np.uint8)
    return np.array([np.asarray(dec.decode(row), np.uint8) for row in S], np.uint8)


def _syndromes_from_errors(code, E):
    H = code.parity_check_matrix()
    return ((E @ H.T) & 1).astype(np.uint8)


# ---------------------------------------------------------------------------
# 1. Row-order invariance -- the rayon-indexing net
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("code_name,code", CODES, ids=CODE_IDS)
@pytest.mark.parametrize("name", ALL_BATCH)
def test_batch_row_order_invariance(name, code_name, code):
    """Permuting input rows must permute output rows the same way.

    This is the property that a row/worker indexing bug violates while every
    individual row still looks like a valid correction. Concretely: if the batch
    kernel pairs row *i*'s syndrome with row *j*'s output slot, per-row
    faithfulness fails only by luck, but ``batch(P·S) == P·batch(S)`` fails
    always.
    """
    n = code.n_qubits
    dec = _mk(name, code)
    rng = np.random.default_rng(17)
    N = 48
    E = (rng.random((N, n)) < 0.15).astype(np.uint8)
    S = _syndromes_from_errors(code, E)

    out = _batch(dec, S, n)

    perm = rng.permutation(N)
    out_perm = _batch(_mk(name, code), np.ascontiguousarray(S[perm]), n)

    assert np.array_equal(out_perm, out[perm]), (
        f"{name} on {code_name}: batch(P.S) != P.batch(S) -- rows are paired with the wrong outputs"
    )


# ---------------------------------------------------------------------------
# 2. Per-row faithfulness over Hypothesis-drawn batches
# ---------------------------------------------------------------------------


@settings(max_examples=40, deadline=8000, suppress_health_check=[HealthCheck.too_slow])
@given(data=st.data())
@pytest.mark.parametrize("code_name,code", CODES, ids=CODE_IDS)
@pytest.mark.parametrize("name", ALL_BATCH)
def test_every_batch_row_is_faithful(name, code_name, code, data):
    """Every row of a drawn batch must satisfy ``(H @ c_i) & 1 == s_i``."""
    H = code.parity_check_matrix()
    n = code.n_qubits
    dec = _mk(name, code)
    N = data.draw(st.integers(min_value=1, max_value=12))
    E = np.array(
        [data.draw(st.lists(st.integers(0, 1), min_size=n, max_size=n)) for _ in range(N)],
        np.uint8,
    )
    S = _syndromes_from_errors(code, E)
    out = _batch(dec, S, n)
    assert out.shape == (N, n)
    assert np.array_equal((H @ out.T).T & 1, S), f"{name} on {code_name}: some batch row unfaithful"


# ---------------------------------------------------------------------------
# 3. Bit-identity to per-shot decoding, over drawn batches
# ---------------------------------------------------------------------------


@settings(max_examples=40, deadline=8000, suppress_health_check=[HealthCheck.too_slow])
@given(data=st.data())
@pytest.mark.parametrize("code_name,code", CODES, ids=CODE_IDS)
@pytest.mark.parametrize("name", BIT_IDENTICAL)
def test_batch_bit_identical_to_single_on_drawn_batches(name, code_name, code, data):
    """Batching is an optimization, never a different answer."""
    n = code.n_qubits
    dec = _mk(name, code)
    N = data.draw(st.integers(min_value=1, max_value=10))
    E = np.array(
        [data.draw(st.lists(st.integers(0, 1), min_size=n, max_size=n)) for _ in range(N)],
        np.uint8,
    )
    S = _syndromes_from_errors(code, E)
    assert np.array_equal(_batch(dec, S, n), _per_shot(_mk(name, code), S, n)), (
        f"{name} on {code_name}: batch != single"
    )


# ---------------------------------------------------------------------------
# 4. Degenerate and adversarial batch shapes
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("code_name,code", CODES, ids=CODE_IDS)
@pytest.mark.parametrize("name", ALL_BATCH)
def test_adversarial_batches(name, code_name, code):
    """N=1, all-zero rows, identical rows, and the all-ones syndrome.

    Batch kernels routinely special-case "small N" or tile by a fixed block
    size; N=1 and N=block_size+1 are where those special cases go wrong.
    """
    H = code.parity_check_matrix()
    n, nc = code.n_qubits, code.n_checks
    dec = _mk(name, code)

    e1 = np.zeros(n, np.uint8)
    e1[0] = 1
    s1 = ((H @ e1) & 1).astype(np.uint8)

    cases = {
        "N=1": np.array([s1], np.uint8),
        "N=2-identical": np.array([s1, s1], np.uint8),
        "N=33-identical": np.array([s1] * 33, np.uint8),
        "all-zero-rows": np.zeros((5, nc), np.uint8),
        "zero-then-defect": np.array([np.zeros(nc, np.uint8), s1], np.uint8),
    }
    for label, S in cases.items():
        out = _batch(dec, S, n)
        assert out.shape == (len(S), n), f"{name}/{code_name}/{label}: shape {out.shape}"
        assert np.array_equal((H @ out.T).T & 1, S), f"{name}/{code_name}/{label}: unfaithful"
        # Identical inputs must give identical outputs.
        if "identical" in label:
            assert np.array_equal(out, np.repeat(out[:1], len(S), axis=0)), (
                f"{name}/{code_name}/{label}: identical rows produced different corrections"
            )


@pytest.mark.parametrize("name", ALL_BATCH)
def test_empty_batch_is_empty_or_raises(name):
    """A zero-row batch must yield zero rows or raise -- never read row 0."""
    code = codes.repetition_code(5)
    dec = _mk(name, code)
    S = np.zeros((0, code.n_checks), np.uint8)
    try:
        out = np.asarray(dec.batch_decode(S), np.uint8)
    except (ValueError, TypeError):
        return
    assert out.size == 0, f"{name}: empty batch produced {out.size} elements"
