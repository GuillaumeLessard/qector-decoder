"""Property-based robustness: malformed input must raise, never abort.

Why this file exists
--------------------
The Rust core is built with ``panic = "abort"`` in the release profile. A
``panic!``/``unwrap``/``expect``/``assert`` reached from any Python entry point
therefore does **not** surface as a Python exception -- it takes down the whole
interpreter, and with it any MCP server, gRPC server or Jupyter kernel hosting
it. Six such panic-to-abort sites were found and fixed in the Rust core (gRPC
and CUDA mutex poisoning, swallowed CUDA errors, a ``Bernoulli::new`` unwrap, a
cascade-decoder ``expect``, and ``cuda_workspace::pointers()``). None of them
were caught by an existing test.

This file is the net for the seventh. The contract asserted throughout is:

    For *any* input, a public decoder entry point either
      (a) returns a syndrome-faithful correction of the documented shape, or
      (b) raises ``ValueError`` or ``TypeError``,
    and in no case terminates the process.

``test_malformed_input_never_aborts_the_process`` enforces the "no abort" half
directly, in a **subprocess**: an abort inside the pytest process would kill the
run itself, so the negative case has to be observed from outside. Every other
test here would be silently unable to fail on an abort.

Complements rather than duplicates:
  * ``test_invalid_inputs.py``  -- fixed, hand-written invalid cases.
  * ``test_property_faithfulness.py`` -- the positive faithfulness property.
"""

from __future__ import annotations

import subprocess
import sys

import numpy as np
import pytest
import qector_decoder_v3 as qd
from hypothesis import HealthCheck, given, settings
from hypothesis import strategies as st

# The typed exceptions a caller is entitled to see. Anything else -- and in
# particular process death -- is a defect.
CLEAN = (ValueError, TypeError)

# Matching-graph decoders sharing the (check_to_qubits, n_qubits) constructor.
GRAPH_DECODERS = [
    "UnionFindDecoder",
    "FastUnionFindDecoder",
    "BlossomDecoder",
    "SparseBlossomDecoder",
    "LookupTableDecoder",
]


def _H(c2q, n):
    """GF(2) parity-check matrix.

    Note the ``^=``: the crate works over GF(2), so a qubit listed twice in the
    same check **cancels** (todo7 GF2-01). Building this with ``|=`` would make
    the verifier disagree with the solver on exactly those inputs.
    """
    H = np.zeros((len(c2q), n), dtype=np.uint8)
    for ci, qs in enumerate(c2q):
        for q in qs:
            H[ci, q] ^= np.uint8(1)
    return H


# ---------------------------------------------------------------------------
# 1. Constructors: arbitrary check structures
# ---------------------------------------------------------------------------

# Deliberately unconstrained: most draws are invalid (empty checks, duplicate
# qubits, out-of-range indices, hyperedges, n_qubits too small). The property is
# not "these are accepted" but "the accept/reject decision is always clean".
arbitrary_checks = st.lists(
    st.lists(st.integers(min_value=-3, max_value=12), min_size=0, max_size=5),
    min_size=0,
    max_size=6,
)


@settings(max_examples=150, deadline=2000, suppress_health_check=[HealthCheck.too_slow])
@given(c2q=arbitrary_checks, n_qubits=st.integers(min_value=-2, max_value=10))
def test_constructor_rejects_cleanly_or_builds_usable_decoder(c2q, n_qubits):
    """Every constructor outcome is either a clean raise or a working decoder.

    A decoder that constructs must then decode its own all-zero syndrome
    without raising -- "constructed" may not mean "half-built and lethal on
    first use".
    """
    for name in GRAPH_DECODERS:
        cls = getattr(qd, name)
        try:
            dec = cls(c2q, n_qubits)
        except CLEAN:
            continue  # (b) clean rejection
        # (a) it built -- so it must be usable.
        syn = np.zeros(len(c2q), np.uint8)
        corr = np.asarray(dec.decode(syn), np.uint8)
        assert corr.shape == (n_qubits,), f"{name}: built with n_qubits={n_qubits} but returned {corr.shape}"
        H = _H(c2q, n_qubits)
        assert np.array_equal((H @ corr) & 1, syn), f"{name}: unfaithful on all-zero syndrome"


# ---------------------------------------------------------------------------
# 2. Wrong-length syndromes
# ---------------------------------------------------------------------------


@settings(max_examples=60, deadline=2000)
@given(
    n_checks=st.integers(min_value=1, max_value=6),
    bad_len=st.integers(min_value=0, max_value=40),
)
@pytest.mark.parametrize("name", GRAPH_DECODERS)
def test_wrong_length_syndrome_raises(name, n_checks, bad_len):
    """A syndrome whose length != n_checks must raise, at every length."""
    c2q = [[i, i + 1] for i in range(n_checks)]
    n_qubits = n_checks + 1
    dec = getattr(qd, name)(c2q, n_qubits)
    syn = np.zeros(bad_len, np.uint8)
    if bad_len == n_checks:
        np.asarray(dec.decode(syn))  # correct length: must succeed
        return
    with pytest.raises(CLEAN):
        dec.decode(syn)


# ---------------------------------------------------------------------------
# 3. Degenerate codes
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("name", GRAPH_DECODERS)
@pytest.mark.parametrize(
    "c2q,n_qubits,label",
    [
        ([[0]], 1, "1-check-1-qubit"),
        ([[0, 1]], 2, "1-check-2-qubits"),
        ([[0, 1], [1, 2]], 3, "2x3-chain"),
        ([[0], [1]], 2, "2-disjoint-weight-1"),
        ([[0, 1], [0, 1]], 2, "2-identical-checks"),
    ],
)
def test_degenerate_codes_decode_faithfully(name, c2q, n_qubits, label):
    """The smallest legal codes must work, not just the textbook-sized ones.

    Off-by-one and "assume at least two checks" bugs hide here.
    """
    dec = getattr(qd, name)(c2q, n_qubits)
    H = _H(c2q, n_qubits)
    # Exhaustive over every syndrome reachable from an error on this tiny code.
    for mask in range(1 << n_qubits):
        e = np.array([(mask >> i) & 1 for i in range(n_qubits)], np.uint8)
        s = ((H @ e) & 1).astype(np.uint8)
        corr = np.asarray(dec.decode(s), np.uint8)
        assert corr.shape == (n_qubits,), f"{name}/{label}: shape {corr.shape}"
        assert np.array_equal((H @ corr) & 1, s), f"{name}/{label}: H@c != s for e={e}"


# ---------------------------------------------------------------------------
# 4. Wrong dtype / wrong ndim / non-contiguous
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("name", GRAPH_DECODERS)
@pytest.mark.parametrize(
    "dtype", [np.float32, np.float64, np.int8, np.int16, np.int32, np.int64, np.uint16, np.uint32, np.bool_]
)
def test_non_uint8_syndrome_dtype_raises(name, dtype):
    """Only uint8 is the documented syndrome dtype; the rest must be refused.

    Silently reinterpreting e.g. an int64 buffer as bytes would read 8x the
    intended memory.
    """
    dec = getattr(qd, name)([[0, 1], [1, 2]], 3)
    with pytest.raises(CLEAN):
        dec.decode(np.zeros(2, dtype))


@pytest.mark.parametrize("name", GRAPH_DECODERS)
@pytest.mark.parametrize("shape", [(2, 2), (1, 2), (2, 2, 2), ()])
def test_wrong_ndim_syndrome_raises(name, shape):
    """``decode`` takes a 1-D syndrome; any other rank must raise."""
    dec = getattr(qd, name)([[0, 1], [1, 2]], 3)
    with pytest.raises(CLEAN):
        dec.decode(np.zeros(shape, np.uint8))


@pytest.mark.parametrize("name", GRAPH_DECODERS)
def test_noncontiguous_syndrome_is_faithful_or_raises(name):
    """A strided view must never be read as if it were contiguous.

    Reading a stride-3 view as contiguous silently decodes the *wrong bits* --
    a wrong answer, not a crash, which is the worse failure mode. So the
    assertion is on faithfulness, not merely on survival.
    """
    c2q = [[i, i + 1] for i in range(6)]
    n_qubits = 7
    dec = getattr(qd, name)(c2q, n_qubits)
    H = _H(c2q, n_qubits)

    s = np.array([1, 0, 1, 1, 0, 1], np.uint8)
    buf = np.zeros((6, 3), np.uint8)
    buf[:, 1] = s
    view = buf[:, 1]
    assert not view.flags["C_CONTIGUOUS"]

    try:
        corr = np.asarray(dec.decode(view), np.uint8)
    except CLEAN:
        return  # clean rejection is an acceptable contract
    assert np.array_equal((H @ corr) & 1, s), f"{name}: strided view decoded against the wrong bits"


# ---------------------------------------------------------------------------
# 5. Non-array inputs
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("name", GRAPH_DECODERS)
@pytest.mark.parametrize(
    "bad",
    [None, "ab", 5, 3.5, {"a": 1}, object(), [[1], [0]], np.zeros(2, np.uint8).tobytes()],
    ids=["none", "str", "int", "float", "dict", "object", "nested-list", "bytes"],
)
def test_non_array_syndrome_raises_typed(name, bad):
    """Junk of any Python type must produce ValueError/TypeError, not junk output."""
    dec = getattr(qd, name)([[0, 1], [1, 2]], 3)
    try:
        out = np.asarray(dec.decode(bad), np.uint8)
    except CLEAN:
        return
    # Accepting it is only tolerable if the result is still a well-formed
    # correction of the right length (e.g. a coerced 2-element sequence).
    assert out.shape == (3,), f"{name}: accepted {bad!r} and returned shape {out.shape}"


# ---------------------------------------------------------------------------
# 6. batch_decode shapes
# ---------------------------------------------------------------------------


@settings(max_examples=40, deadline=2000)
@given(rows=st.integers(min_value=0, max_value=4), cols=st.integers(min_value=0, max_value=8))
def test_batch_decode_wrong_width_raises(rows, cols):
    """A batch whose column count != n_checks must raise, not read past the row."""
    c2q = [[0, 1], [1, 2], [2, 3]]
    n_checks, n_qubits = 3, 4
    dec = qd.BlossomDecoder(c2q, n_qubits)
    S = np.zeros((rows, cols), np.uint8)
    if cols == n_checks:
        out = np.asarray(dec.batch_decode(S), np.uint8).reshape(rows, n_qubits)
        assert out.shape == (rows, n_qubits)
        return
    with pytest.raises(CLEAN):
        dec.batch_decode(S)


# ---------------------------------------------------------------------------
# 7. The abort sentinel -- runs out of process on purpose
# ---------------------------------------------------------------------------

# Everything above runs *inside* pytest, where an abort kills the runner before
# any assertion can report it. This runs the same class of abuse in a child and
# inspects its exit status, which is the only way to observe "the process died"
# as a test failure rather than as a vanished test session.
_ABUSE_SCRIPT = r"""
import numpy as np
import qector_decoder_v3 as qd

CLEAN = (ValueError, TypeError)
NAMES = ["UnionFindDecoder", "FastUnionFindDecoder", "BlossomDecoder",
         "SparseBlossomDecoder", "LookupTableDecoder"]

BAD_CHECKS = [
    [],                       # no checks at all
    [[]],                     # an empty check
    [[0, 0]],                 # duplicate qubit in one check (GF(2): cancels)
    [[-1]],                   # negative index
    [[99]],                   # out of range
    [[0, 1], [0, 2], [0, 3]], # hyperedge: qubit 0 in three checks
    [[0]] * 64,               # many identical checks
    [list(range(40))],        # one very wide check
]
BAD_SYNDROMES = [
    np.zeros(0, np.uint8),
    np.zeros(1, np.uint8),
    np.zeros(999, np.uint8),
    np.full(3, 255, np.uint8),          # values outside {0,1}
    np.zeros((2, 2), np.uint8),         # wrong rank
    np.zeros(3, np.float64),            # wrong dtype
    np.zeros((3, 3), np.uint8)[:, 0],   # non-contiguous
    None, "xyz", 7, [1, 0, 1],
]

for name in NAMES:
    cls = getattr(qd, name)
    for checks in BAD_CHECKS:
        for nq in (0, 1, 4, 100):
            try:
                dec = cls(checks, nq)
            except CLEAN:
                continue
            except Exception as exc:            # noqa: BLE001
                print("UNTYPED-CTOR", name, type(exc).__name__, exc)
                raise SystemExit(3)
            for syn in BAD_SYNDROMES:
                try:
                    dec.decode(syn)
                except CLEAN:
                    pass
                except Exception as exc:        # noqa: BLE001
                    print("UNTYPED-DECODE", name, type(exc).__name__, exc)
                    raise SystemExit(4)
            for bad_batch in (np.zeros((2, 2), np.uint8), np.zeros(4, np.uint8),
                              np.zeros((0, 3), np.uint8)):
                try:
                    dec.batch_decode(bad_batch)
                except CLEAN:
                    pass
                except Exception as exc:        # noqa: BLE001
                    print("UNTYPED-BATCH", name, type(exc).__name__, exc)
                    raise SystemExit(5)
print("SURVIVED")
"""


def test_malformed_input_never_aborts_the_process(tmp_path):
    """The core must not ``abort()`` on malformed input.

    Run from ``tmp_path`` so the child resolves the *installed* package (with
    its compiled extension) and not the repo source tree -- same reasoning as
    ``conftest.py``.

    A negative return code on POSIX, or a status like 0xC0000409 on Windows, is
    the signature of an abort rather than of a Python-level error.
    """
    proc = subprocess.run(
        [sys.executable, "-c", _ABUSE_SCRIPT],
        capture_output=True,
        text=True,
        cwd=str(tmp_path),
        timeout=600,
        env={**__import__("os").environ, "QECTOR_SILENT": "1"},
    )
    assert proc.returncode == 0, (
        f"child exited {proc.returncode} (abort/untyped-exception).\n"
        f"--- stdout ---\n{proc.stdout}\n--- stderr ---\n{proc.stderr}"
    )
    assert "SURVIVED" in proc.stdout, f"child did not finish the abuse sweep:\n{proc.stdout}\n{proc.stderr}"
