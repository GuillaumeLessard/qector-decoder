"""Pure-Python fallbacks must be byte-exact equivalents of the native utils.

Release wheels compile the Rust core from the RUST_SRC_B64_* secrets bundle,
which can lag the repo's Python layer. The v0.6.9 tag build died at import
because `py_generate_parity_check_matrix` existed in the repo's lib.rs but not
in the bundle's older core -- the same class of failure that made every v0.6.7
wheel unimportable. `__init__` now resolves these six utilities via
`_native_or(name, fallback)`.

Two contracts locked here:

1. On a build where the native implementations exist (this machine), each
   fallback produces *identical* output -- so a wheel that falls back loses
   nothing.
2. `_native_or` actually returns the fallback when the symbol is absent, so
   import survives an older core.
"""

from __future__ import annotations

import numpy as np
import pytest
import qector_decoder_v3 as qd

_HAS_NATIVE = all(
    hasattr(qd._native_module, n)
    for n in (
        "py_check_to_edges",
        "py_generate_surface_code_checks",
        "py_generate_toy_code_checks",
        "py_generate_ring_code_checks",
        "py_generate_repetition_code_checks",
        "py_generate_parity_check_matrix",
    )
)

needs_native = pytest.mark.skipif(
    not _HAS_NATIVE, reason="native utils absent in this build; equivalence needs both sides"
)


def _norm_checks(pair):
    """Normalise (checks, n_qubits) across native (tuples/uints) and Python (lists/ints)."""
    checks, nq = pair
    return [[int(q) for q in c] for c in checks], int(nq)


@needs_native
@pytest.mark.parametrize("d", [2, 3, 5, 7])
def test_surface_checks_equivalent(d):
    assert _norm_checks(qd._py_generate_surface_code_checks(d)) == _norm_checks(
        qd._native_module.py_generate_surface_code_checks(d)
    )


@needs_native
def test_surface_rejects_distance_below_two():
    with pytest.raises(ValueError):
        qd._py_generate_surface_code_checks(1)
    with pytest.raises(ValueError):
        qd._native_module.py_generate_surface_code_checks(1)


@needs_native
@pytest.mark.parametrize("d", [1, 2, 3, 5, 8])
def test_ring_checks_equivalent(d):
    assert _norm_checks(qd._py_generate_ring_code_checks(d)) == _norm_checks(
        qd._native_module.py_generate_ring_code_checks(d)
    )


@needs_native
@pytest.mark.parametrize("d", [1, 2, 3, 5])
def test_toy_checks_equivalent(d):
    assert _norm_checks(qd._py_generate_toy_code_checks(d)) == _norm_checks(
        qd._native_module.py_generate_toy_code_checks(d)
    )


@needs_native
@pytest.mark.parametrize("d", [0, 1, 2, 5, 9])
def test_repetition_checks_equivalent(d):
    assert _norm_checks(qd._py_generate_repetition_code_checks(d)) == _norm_checks(
        qd._native_module.py_generate_repetition_code_checks(d)
    )


EDGE_CASES = [
    [[0, 1], [1, 2], [2, 3]],
    [[0, 1, 3, 4], [1, 2, 4, 5]],  # weight-4 checks -> consecutive pairs
    [[7]],  # single-qubit check contributes no edges
    [[0, 5]],  # sparse indices
]


@needs_native
@pytest.mark.parametrize("c2q", EDGE_CASES)
def test_check_to_edges_equivalent(c2q):
    py = [(int(a), int(b)) for a, b in qd._py_check_to_edges(c2q)]
    native = [(int(a), int(b)) for a, b in qd._native_module.py_check_to_edges(c2q)]
    assert py == native


PARITY_CASES = [
    ([[0, 1], [1, 2]], 3),
    ([[0, 1], [1, 2]], None),  # n_qubits inferred as max+1
    ([[0, 3]], None),
    ([[0, 5]], 3),  # out-of-range qubit dropped, not an error
    ([[0, 1, 3, 4], [1, 2, 4, 5]], 9),
]


@needs_native
@pytest.mark.parametrize(("c2q", "nq"), PARITY_CASES)
def test_parity_check_matrix_equivalent(c2q, nq):
    py = qd._py_generate_parity_check_matrix(c2q, nq)
    native = np.asarray(qd._native_module.py_generate_parity_check_matrix(c2q, nq))
    assert py.dtype == np.uint8
    assert py.shape == native.shape
    assert np.array_equal(py, native)


def test_native_or_falls_back_when_symbol_missing(monkeypatch):
    """The exact CI-wheel condition, exercised through the real resolver.

    `_native_or` reads from the module-level `_native_module`, so swap that for
    a core with no utils registered and confirm the fallback is what comes back.
    """

    class _OlderCore:
        pass  # the bundle's core: none of the six utils exist

    monkeypatch.setattr(qd, "_native_module", _OlderCore())

    sentinel = object()

    def fallback():
        return sentinel

    resolved = qd._native_or("py_generate_parity_check_matrix", fallback)
    assert resolved is fallback, "_native_or must return the fallback for a missing symbol"
    assert resolved() is sentinel


@needs_native
def test_native_or_prefers_the_compiled_symbol():
    """When the core does provide it, the native implementation must win."""

    def fallback():  # pragma: no cover - must never be selected here
        raise AssertionError("fallback selected despite a native symbol being present")

    resolved = qd._native_or("py_generate_parity_check_matrix", fallback)
    assert resolved is qd._native_module.py_generate_parity_check_matrix


def test_fallbacks_faithful_against_decoder():
    """A fallback-generated code must decode correctly end to end."""
    checks, nq = qd._py_generate_ring_code_checks(3)
    dec = qd.UnionFindDecoder(checks, n_qubits=nq)
    h = qd._py_generate_parity_check_matrix(checks, nq)
    syndrome = np.zeros(len(checks), dtype=np.uint8)
    syndrome[0] = 1
    syndrome[1] = 1
    corr = dec.decode(syndrome)
    assert np.array_equal((h @ corr) & 1, syndrome)


def test_distance_estimate_handles_stim_dem_and_repetition_code():
    """A DEM's error-mechanism count must never be mistaken for code distance."""
    stim = pytest.importorskip("stim")
    from qector_decoder_v3 import dem

    circuit = stim.Circuit.generated(
        "surface_code:rotated_memory_x",
        distance=5,
        rounds=5,
        after_clifford_depolarization=0.003,
    )
    model = dem.from_stim(circuit.detector_error_model(decompose_errors=True))
    checks = model.check_to_qubits()

    # The d=5 DEM has thousands of fault mechanisms; its distance is still 5.
    assert qd._py_estimate_distance(checks, model.num_errors) == 5
    assert qd.estimate_distance(checks, model.num_errors) == 5

    repetition_checks = [[i, i + 1] for i in range(24)]
    assert qd._py_estimate_distance(repetition_checks, 25) == 25
    assert qd.estimate_distance(repetition_checks, 25) == 25


@needs_native
@pytest.mark.parametrize(
    ("checks", "n_qubits"),
    [
        ([[0, 1], [1, 2], [2, 3]], 4),
        ([[0, 1, 2], [1, 3], [0, 2, 3]], 4),
        ([[0, 1], [1, 2], [2, 3], [3, 4]], 5),
    ],
)
def test_distance_estimator_native_fallback_equivalence(checks, n_qubits):
    assert qd._py_estimate_distance(checks, n_qubits) == qd._native_module.py_estimate_distance(checks, n_qubits)
