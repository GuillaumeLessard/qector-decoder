"""Property-based GF(2) and statelessness invariants.

The crate works over **GF(2)**: parity checks XOR, so a qubit appearing twice in
a check *cancels* rather than counting twice (todo7 GF2-01, where the solver used
``|=`` while every verifier used ``^=``). Every parity computation in this file
therefore uses ``^=`` / ``& 1``, matching the verifier convention.

What is asserted here that is asserted nowhere else:

  * **Statelessness / idempotence.** ``fast_uf.rs`` reuses a *thread-local*
    ``UfScratch`` across calls (``TL_SCRATCH``), and ``decode_into`` reuses a
    caller-owned ``DecodeBuffers``. If any scratch field is read before being
    reset, the answer for syndrome *B* depends on whichever syndrome was decoded
    before it. That is invisible to every fixed-sequence test in the suite,
    because they all decode in one fixed order. Here the *same* syndrome is
    decoded repeatedly, interleaved with others, and after a ``batch_decode``,
    and the answers must be bit-identical every time.
  * **Correction alphabet.** Corrections must contain only 0/1. A value of 2
    (or 255) is the fingerprint of a count-instead-of-parity or ``|=``-instead-of-
    ``^=`` bug -- exactly the GF2-01 defect class -- and would still satisfy a
    faithfulness check written with ``% 2`` on ints.
  * **Faithfulness across *all five* graph decoders on real codes.**
    ``test_property_faithfulness.py`` restricts its random-graph property to the
    two exact decoders (correctly -- the approximate ones are not faithful on
    arbitrary degree-<=2 hypergraphs) and covers the rest on four fixed codes
    with fixed seeds. This file runs all five over Hypothesis-drawn error
    patterns, including the high-weight tail.
"""

from __future__ import annotations

import numpy as np
import pytest
import qector_decoder_v3 as qd
from hypothesis import HealthCheck, given, settings
from hypothesis import strategies as st
from qector_decoder_v3 import codes

GRAPH_DECODERS = [
    "UnionFindDecoder",
    "FastUnionFindDecoder",
    "BlossomDecoder",
    "SparseBlossomDecoder",
    "LookupTableDecoder",
]


def _codes():
    """Small enough that a 100-example property run stays under a few seconds."""
    return [
        ("repetition_d5", codes.repetition_code(5)),
        ("repetition_d9", codes.repetition_code(9)),
        ("ring_d6", codes.ring_code(6)),
        ("rotated_surface_d3", codes.rotated_surface_code(3)),
        ("rotated_surface_d5", codes.rotated_surface_code(5)),
    ]


CODES = _codes()
CODE_IDS = [n for n, _ in CODES]


def _mk(name, code):
    return getattr(qd, name)(code.check_to_qubits, code.n_qubits)


# ---------------------------------------------------------------------------
# 1. Faithfulness over Hypothesis-drawn error patterns
# ---------------------------------------------------------------------------


@settings(max_examples=100, deadline=4000, suppress_health_check=[HealthCheck.too_slow])
@given(data=st.data())
@pytest.mark.parametrize("code_name,code", CODES, ids=CODE_IDS)
@pytest.mark.parametrize("name", GRAPH_DECODERS)
def test_faithfulness_on_drawn_errors(name, code_name, code, data):
    """``(H @ decode(H @ e)) & 1 == (H @ e) & 1`` for Hypothesis-drawn ``e``.

    Drawing the error (not the syndrome) keeps every syndrome *reachable*, which
    is the precondition the approximate decoders are entitled to assume.
    """
    H = code.parity_check_matrix()
    n = code.n_qubits
    dec = _mk(name, code)
    e = np.array(data.draw(st.lists(st.integers(0, 1), min_size=n, max_size=n)), np.uint8)
    s = ((H @ e) & 1).astype(np.uint8)
    corr = np.asarray(dec.decode(s), np.uint8)
    assert corr.shape == (n,)
    assert np.array_equal((H @ corr) & 1, s), f"{name} on {code_name}: H@c != s for e={e.tolist()}"


# ---------------------------------------------------------------------------
# 2. Correction alphabet: GF(2) means {0, 1}
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("code_name,code", CODES, ids=CODE_IDS)
@pytest.mark.parametrize("name", GRAPH_DECODERS)
def test_correction_is_binary(name, code_name, code):
    """No correction entry may exceed 1.

    A count rather than a parity (``|=``/``+=`` instead of ``^=``) shows up here
    and nowhere else: ``(H @ c) & 1`` can still match the syndrome while ``c``
    itself carries a 2.
    """
    H = code.parity_check_matrix()
    n = code.n_qubits
    dec = _mk(name, code)
    rng = np.random.default_rng(11)
    for _ in range(40):
        e = (rng.random(n) < 0.3).astype(np.uint8)
        s = ((H @ e) & 1).astype(np.uint8)
        corr = np.asarray(dec.decode(s), np.uint8)
        bad = np.unique(corr[corr > 1])
        assert bad.size == 0, f"{name} on {code_name}: non-binary correction entries {bad.tolist()}"


# ---------------------------------------------------------------------------
# 3. All-zero syndrome
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("code_name,code", CODES, ids=CODE_IDS)
@pytest.mark.parametrize("name", GRAPH_DECODERS)
def test_all_zero_syndrome_decodes_to_zero_syndrome(name, code_name, code):
    """The trivial syndrome must not invent defects.

    A weight-0 syndrome is the single most common input in a low-error-rate
    run, so a decoder that returns a non-stabilizer correction here is broken in
    the common case, not the rare one.
    """
    H = code.parity_check_matrix()
    dec = _mk(name, code)
    z = np.zeros(code.n_checks, np.uint8)
    corr = np.asarray(dec.decode(z), np.uint8)
    assert corr.shape == (code.n_qubits,)
    assert np.array_equal((H @ corr) & 1, z), f"{name} on {code_name}: zero syndrome produced defects"


# ---------------------------------------------------------------------------
# 4. Statelessness -- the thread-local-scratch net
# ---------------------------------------------------------------------------


@settings(max_examples=30, deadline=6000, suppress_health_check=[HealthCheck.too_slow])
@given(data=st.data())
@pytest.mark.parametrize("name", GRAPH_DECODERS)
def test_decode_is_idempotent_and_order_independent(name, data):
    """The answer for a syndrome must not depend on what was decoded before it.

    ``fast_uf.rs`` keeps a thread-local ``UfScratch`` alive between calls; a
    field that is read before being reset makes ``decode(b)`` depend on the
    preceding ``decode(a)``. This draws two syndromes and checks that decoding
    ``a`` alone, and decoding ``a`` after ``b``, agree bit for bit.
    """
    code = codes.rotated_surface_code(5)
    H = code.parity_check_matrix()
    n = code.n_qubits
    dec = _mk(name, code)

    def draw_syn():
        e = np.array(data.draw(st.lists(st.integers(0, 1), min_size=n, max_size=n)), np.uint8)
        return ((H @ e) & 1).astype(np.uint8)

    a, b = draw_syn(), draw_syn()

    baseline_a = np.asarray(dec.decode(a), np.uint8)
    baseline_b = np.asarray(dec.decode(b), np.uint8)

    # Repeat: same input, same output.
    assert np.array_equal(np.asarray(dec.decode(a), np.uint8), baseline_a), f"{name}: decode(a) not idempotent"

    # Interleave: b in between must not perturb a.
    dec.decode(b)
    assert np.array_equal(np.asarray(dec.decode(a), np.uint8), baseline_a), f"{name}: decode(a) perturbed by decode(b)"

    # A fresh instance must agree with the used one.
    fresh = _mk(name, code)
    assert np.array_equal(np.asarray(fresh.decode(b), np.uint8), baseline_b), (
        f"{name}: warm instance disagrees with fresh"
    )


@pytest.mark.parametrize("name", GRAPH_DECODERS)
def test_batch_decode_does_not_pollute_single_decode(name):
    """A ``batch_decode`` must leave no state that changes later ``decode`` calls.

    The batch path uses different scratch (and, for the rayon paths, different
    threads) than the single path; sharing a buffer between them is a plausible
    optimization and a silent-corruption bug.
    """
    code = codes.rotated_surface_code(5)
    H = code.parity_check_matrix()
    n = code.n_qubits
    dec = _mk(name, code)
    rng = np.random.default_rng(3)

    probe_e = (rng.random(n) < 0.2).astype(np.uint8)
    probe = ((H @ probe_e) & 1).astype(np.uint8)
    before = np.asarray(dec.decode(probe), np.uint8)

    S = np.array(
        [((H @ (rng.random(n) < 0.25).astype(np.uint8)) & 1).astype(np.uint8) for _ in range(64)],
        np.uint8,
    )
    try:
        dec.batch_decode(S)
    except (ValueError, TypeError, AttributeError):
        pytest.skip(f"{name} has no usable batch_decode")

    after = np.asarray(dec.decode(probe), np.uint8)
    assert np.array_equal(before, after), f"{name}: decode(probe) changed after batch_decode"


# ---------------------------------------------------------------------------
# 5. Weight-1 errors: every column of H must be individually decodable
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("code_name,code", CODES, ids=CODE_IDS)
@pytest.mark.parametrize("name", GRAPH_DECODERS)
def test_every_single_qubit_error_is_faithful(name, code_name, code):
    """Exhaustive over weight-1 errors -- one per qubit, no sampling.

    A single miswired column of H (or one boundary edge omitted from the
    matching graph) affects exactly one qubit and is easy for a random sampler
    at p=0.1 to miss on a large code. This cannot miss it.
    """
    H = code.parity_check_matrix()
    n = code.n_qubits
    dec = _mk(name, code)
    for q in range(n):
        e = np.zeros(n, np.uint8)
        e[q] = 1
        s = ((H @ e) & 1).astype(np.uint8)
        corr = np.asarray(dec.decode(s), np.uint8)
        assert np.array_equal((H @ corr) & 1, s), f"{name} on {code_name}: weight-1 error on qubit {q} unfaithful"
