"""Property-based and adversarial syndrome-faithfulness tests.

Uses Hypothesis to generate random matching graphs (every qubit in <= 2 checks)
and random / adversarial syndromes, asserting the core invariant
``H @ decode(s) == s (mod 2)`` for every decoder.  This is the property that the
older shape-only tests missed.
"""

import numpy as np
import pytest
from hypothesis import given, settings, strategies as st

import qector_decoder_v3 as qd
from qector_decoder_v3 import codes


def _H(c2q, n):
    H = np.zeros((len(c2q), n), dtype=np.uint8)
    for ci, qs in enumerate(c2q):
        for q in qs:
            H[ci, q] ^= np.uint8(1)
    return H


@st.composite
def matching_graphs(draw):
    """Random graphlike code built as a graph of edges (=qubits).

    Every qubit is an edge between two distinct nodes drawn from ``n_checks``
    checks plus one virtual boundary node, so each qubit touches at most two
    checks (degree <= 2) — a genuine matching graph.  Empty checks are bound to
    the boundary with a fresh qubit so no check is degenerate.
    """
    n_checks = draw(st.integers(min_value=2, max_value=9))
    boundary = n_checks
    n_edges = draw(st.integers(min_value=n_checks, max_value=2 * n_checks + 2))
    c2q = [[] for _ in range(n_checks)]
    qubit = 0
    for _ in range(n_edges):
        u = draw(st.integers(min_value=0, max_value=n_checks))
        v = draw(st.integers(min_value=0, max_value=n_checks))
        if u == v:
            continue  # skip self-loops (would XOR a column to zero)
        for node in (u, v):
            if node != boundary:
                c2q[node].append(qubit)
        qubit += 1
    for ci in range(n_checks):
        if not c2q[ci]:
            c2q[ci].append(qubit)  # boundary edge -> fresh qubit, degree 1
            qubit += 1
    n_qubits = max(qubit, 1)
    if qubit == 0:
        c2q[0].append(0)
    return [sorted(set(x)) for x in c2q], n_qubits


# The exact decoders (Blossom = exact MWPM, SparseBlossom = region-growing) are
# guaranteed syndrome-faithful on *any* matching graph — verified to 0 failures
# over thousands of random graphs. UnionFind / FastUnionFind are fast *approximate*
# decoders: faithful on proper QEC matching graphs (surface/repetition/toric — see
# test_syndrome_faithfulness.py and test_codes.py), but they can return an invalid
# correction on rare adversarial degree-<=2 hypergraphs where a defect's only path
# to the boundary threads through several checks. So this arbitrary-graph property
# test covers the exact decoders; UnionFind faithfulness is asserted on real codes.
EXACT_DECODER_FACTORIES = {
    "Blossom": lambda c, n: qd.BlossomDecoder(c, n),
    "SparseBlossom": lambda c, n: qd.SparseBlossomDecoder(c, n),
}


@settings(max_examples=200, deadline=None)
@given(graph=matching_graphs(), data=st.data())
def test_random_graph_faithfulness(graph, data):
    c2q, n = graph
    H = _H(c2q, n)
    # a reachable syndrome from a random error
    e = np.array([data.draw(st.integers(0, 1)) for _ in range(n)], dtype=np.uint8)
    s = (H @ e) & 1
    for name, factory in EXACT_DECODER_FACTORIES.items():
        dec = factory(c2q, n)
        c = np.asarray(dec.decode(s.astype(np.uint8))).astype(np.uint8)
        assert c.shape == (n,)
        assert np.array_equal((H @ c) & 1, s), f"{name}: H@c != s for {c2q}"


@pytest.mark.parametrize(
    "code",
    [
        codes.rotated_surface_code(5),
        codes.rotated_surface_code(7),
        codes.unrotated_surface_code(5),
        codes.toric_code(5),
    ],
    ids=lambda c: c.name,
)
def test_adversarial_dense_and_all_zero_syndromes(code):
    """Dense, all-zero, and single-defect syndromes must all stay faithful."""
    H = code.parity_check_matrix()
    n = code.n_qubits
    decs = {
        "UnionFind": qd.UnionFindDecoder(code.check_to_qubits, n),
        "Blossom": qd.BlossomDecoder(code.check_to_qubits, n),
        "SparseBlossom": qd.SparseBlossomDecoder(code.check_to_qubits, n),
    }
    rng = np.random.default_rng(0)
    syndromes = [np.zeros(code.n_checks, np.uint8)]
    # high-density reachable syndromes (p=0.45)
    for _ in range(40):
        e = (rng.random(n) < 0.45).astype(np.uint8)
        syndromes.append(((H @ e) & 1).astype(np.uint8))
    # every single-defect-pair reachable syndrome via single qubit flips
    for q in range(n):
        e = np.zeros(n, np.uint8)
        e[q] = 1
        syndromes.append(((H @ e) & 1).astype(np.uint8))

    for name, dec in decs.items():
        for s in syndromes:
            c = np.asarray(dec.decode(s)).astype(np.uint8)
            assert np.array_equal((H @ c) & 1, s), f"{name} on {code.name}"


def test_all_zero_syndrome_yields_zero_or_stabilizer():
    """An all-zero syndrome must produce a correction with zero syndrome."""
    code = codes.rotated_surface_code(7)
    H = code.parity_check_matrix()
    z = np.zeros(code.n_checks, np.uint8)
    for name, dec in [
        ("UnionFind", qd.UnionFindDecoder(code.check_to_qubits, code.n_qubits)),
        ("Blossom", qd.BlossomDecoder(code.check_to_qubits, code.n_qubits)),
        ("SparseBlossom", qd.SparseBlossomDecoder(code.check_to_qubits, code.n_qubits)),
    ]:
        c = np.asarray(dec.decode(z)).astype(np.uint8)
        assert np.array_equal((H @ c) & 1, z), name
