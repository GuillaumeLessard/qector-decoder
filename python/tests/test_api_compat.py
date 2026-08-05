"""API compatibility contract (todo8 items 0.2 / 8.2).

``test_public_symbols.py`` already enforces ``expected_symbols.txt``: which
names exist, that none has degraded to a ``_guard()`` stub, and that the surface
cannot grow unreviewed. That is a *presence* contract. It says nothing about how
any of those names may be **called**.

This file is the *shape* contract. It is what stands between the SemVer policy
and a silent break:

* renaming or reordering a constructor parameter,
* promoting an optional parameter to required,
* dropping a keyword argument a caller passes by name,
* a registered console script whose target module or attribute does not exist
  in the built wheel,
* a documented import spelling (``from qector_decoder_v3.pymatching import
  Matching``) that stops resolving,
* a decoder method whose spelling drifts (``batch_decode`` vs ``decode_batch``).

Every one of those is a major-version event that ``import qector_decoder_v3``
and an attribute check both sail straight past.

**On failure, check the wheel first.** ``conftest.py`` deliberately resolves
``qector_decoder_v3`` to the *installed* package, so an edit under
``python/qector_decoder_v3/`` is invisible here until
``maturin build --release && pip install --force-reinstall --no-deps <wheel>``.
A symbol that is present in the working tree and missing here is a stale wheel,
not a regression. See ``docs/TROUBLESHOOTING.md``.

The signature table below is a **baseline recorded from the working tree on
2026-07-31**, not an aspiration. Regenerating it to make a red test go green
defeats the entire point: change it only together with a CHANGELOG entry and a
SemVer decision.
"""

from __future__ import annotations

import importlib
import inspect
import os
from pathlib import Path

import numpy as np
import pytest

os.environ.setdefault("QECTOR_SILENT", "1")

import qector_decoder_v3 as qd

REPO_ROOT = Path(__file__).resolve().parents[2]
PYPROJECT = REPO_ROOT / "pyproject.toml"

STALE_WHEEL_HINT = (
    "If this name exists under python/qector_decoder_v3/ but not here, the "
    "installed wheel predates the working tree: rebuild with "
    "`maturin build --release && pip install --force-reinstall --no-deps <wheel>`."
)


# ---------------------------------------------------------------------------
# Normalised signature spec
# ---------------------------------------------------------------------------
def signature_spec(fn) -> str:
    """Render a callable's parameters as an annotation-independent string.

    ``"check_to_qubits, n_qubits=, edge_weights="`` means: one required
    positional-or-keyword parameter followed by two optional ones, in that
    order. ``*name`` marks keyword-only, ``**name`` marks var-keyword.

    Annotations and *default values* are deliberately excluded. Retyping a
    parameter or retuning a default is a behaviour change tracked in the
    CHANGELOG; renaming, reordering, or making one required is an API break,
    and that is what this string pins.
    """
    parts: list[str] = []
    for name, p in inspect.signature(fn).parameters.items():
        if name in ("self", "cls"):
            continue
        if p.kind is inspect.Parameter.VAR_POSITIONAL:
            parts.append("*" + name)
        elif p.kind is inspect.Parameter.VAR_KEYWORD:
            parts.append("**" + name)
        else:
            prefix = "*" if p.kind is inspect.Parameter.KEYWORD_ONLY else ""
            parts.append(prefix + name + ("=" if p.default is not inspect.Parameter.empty else ""))
    return ", ".join(parts)


# Constructors of the documented public decoder classes.
#
# Recorded 2026-07-31 against the working tree. `BlossomDecoder(c2q, n_qubits)`
# stays valid because `edge_weights` is optional on both sides of the PyO3
# boundary (`Option<Vec<f64>>` in Rust, defaulted in the .pyi) -- adding it was
# additive, not breaking, and this table is what keeps it that way.
CONSTRUCTOR_SPECS: dict[str, str] = {
    "BlossomDecoder": "check_to_qubits, n_qubits=, edge_weights=",
    "SparseBlossomDecoder": "check_to_qubits, n_qubits=, edge_weights=",
    "UnionFindDecoder": "check_to_qubits, n_qubits=, edge_weights=",
    "FastUnionFindDecoder": "check_to_qubits, n_qubits=, edge_weights=",
    "BPOSDDecoder": "check_to_qubits, n_qubits=, error_rate=, bp_method=, osd_order=",
    "CPUBatchDecoder": "check_to_qubits, n_qubits=",
    "BatchDecoder": "check_to_qubits, n_qubits=",
    "LookupTableDecoder": "check_to_qubits, n_qubits=",
    "HybridCascadeDecoder": ("check_to_qubits, n_qubits=, edge_weights=, max_accept_weight=, escalation=, error_rate="),
    "HybridDecoder": (
        "check_to_qubits, n_qubits=, check_positions=, check_types=, base_weights=, gnn_hidden_size=, gnn_n_layers="
    ),
    "StreamingDecoder": "check_to_qubits, n_qubits=, history_size=, check_types=, p_data=, p_meas=",
    "SlidingWindowDecoder": ("check_to_qubits, n_qubits=, window_size=, decay_factor=, check_types=, p_data=, p_meas="),
    # C1-03 / C1-02, new in 0.7.0. `check_types` is the one *required* second
    # positional in the whole family; a caller who omits it gets a TypeError,
    # so it must never silently move.
    "TwoStageDecoder": "check_to_qubits, check_types, n_qubits=, x_decoder=, z_decoder=",
    "AmbiguityClusterDecoder": ("check_to_qubits, n_qubits=, error_rate=, ambig_threshold=, max_cluster_size="),
    "BeliefMatching": "matrices, max_iter=, bp_shortcut=",
    # v1.0 additive: `damping` (LLR message damping) and `osd_lambda` (CS-OSD
    # sweep-set size) are optional keyword args appended at the end; every
    # pre-1.0 call site keeps working unchanged.
    "BpOsdDecoder": (
        "H, error_rate=, priors=, max_iter=, ms_scale=, osd_order=, bp_method=, use_gpu=, max_latency_ms=, "
        "damping=, osd_lambda="
    ),
    "PredecodedDecoder": "check_to_qubits, n_qubits=, backend=",
    "AutoDecoder": "check_to_qubits, n_qubits=, config=",
    "DecoderPool": "check_to_qubits=, n_qubits=, decoder_type=, n_workers=, num_threads=",
    "BackendConfig": "rayon_threshold=, gpu_threshold=, allow_gpu=, prefer=, force=, enable_auto_debug=",
    "DecodeResult": (
        "correction, syndrome, n_qubits, n_checks, weight=, logical_flips=, decode_seconds=, "
        "backend=, fallback=, fallback_reason=, syndrome_valid=, metadata="
    ),
    "AutoRouter": "*priority=, *hardware=, *error_rate=, *code_family=, *use_native_auto=",
    "StreamingSession": "code_or_checks, n_qubits=, *window_size=, *decoder=, *logicals=, *prefer_gpu=",
    "BatchedBpDecoder": "H, *error_rate=, *priors=, *max_iter=, *alpha=, *bp_method=, *prefer_gpu=",
    "GNNBeliefMatcher": (
        "check_to_qubits, n_qubits=, *gnn=, *hidden_size=, *n_layers=, *train_samples=, "
        "*error_rate=, *train_epochs=, *seed="
    ),
}

# Module-level public callables.
FUNCTION_SPECS: dict[str, str] = {
    "from_circuit": "circuit, decoder_type=, **kwargs",
    "get_decoder": "checks_tuple, n_qubits, decoder_type=",
    "get_decoder_pool": "checks_tuple, n_qubits, decoder_type=, n_workers=",
    "clear_decoder_cache": "",
    "decode_with_diagnostics": "code, syndrome, kind=, decoder=, logicals=",
    "decode_with_gnn": "check_to_qubits, n_qubits, syndrome, **kwargs",
    "set_license_key": "key",
    "get_license_info": "",
    "record_shots": "count",
    "get_accumulated_shots": "",
    "verify_license_token": "token, customer_email=",
    "enforce_distance_cap": "distance",
    "enforce_unlocked": "",
    "check_to_edges": "check_to_qubits",
    "estimate_distance": "check_to_qubits, n_qubits=",
    "recommend_decoder": "code_family=, distance=, n_qubits=, batch_size=, priority=, hardware=",
    "recommend": "code_family=, distance=, n_qubits=, batch_size=, priority=, hardware=, *graphlike=",
    "detect_hardware": "",
    "sliding_window_decode": (
        "syndrome_rounds, code=, *check_to_qubits=, *n_qubits=, *window_size=, *decoder=, *logicals=, *prefer_gpu="
    ),
    "batched_bp_decode": (
        "H, syndromes, *error_rate=, *priors=, *max_iter=, *alpha=, *bp_method=, *prefer_gpu=, "
        "*early_stop=, *return_llr="
    ),
    "get_backend": "",
    "gpu_available": "",
    "has_cuda_rust": "",
    "has_cupy": "",
    "generate_surface_code_checks": "distance",
    "generate_repetition_code_checks": "distance",
    "generate_ring_code_checks": "distance",
    "generate_toy_code_checks": "distance",
    "generate_biconnected_qldpc_checks": "n_qubits, degree",
    "generate_space_time_surface_code_checks": "distance, rounds",
    "generate_triangular_color_code_4_8_8_checks": "distance",
    "compute_detector_differences": "history",
    "sparse_blossom_radix_neighbors": "decoder, defects, k=",
    "start_metrics_server": "addr=",
    "changelog": "",
}

# Instance methods on the documented decode surface.
METHOD_SPECS: dict[str, str] = {
    "BlossomDecoder.decode": "syndrome",
    "BlossomDecoder.batch_decode": "syndromes",
    "BlossomDecoder.set_edge_weights": "weights",
    "SparseBlossomDecoder.decode": "syndrome",
    "SparseBlossomDecoder.batch_decode": "syndromes",
    "UnionFindDecoder.decode": "syndrome",
    "UnionFindDecoder.batch_decode": "syndromes",
    "FastUnionFindDecoder.decode": "syndrome",
    "FastUnionFindDecoder.batch_decode": "syndromes",
    "BPOSDDecoder.decode": "syndrome",
    "BPOSDDecoder.batch_decode": "syndromes",
    "CPUBatchDecoder.batch_decode": "syndromes",
    "BatchDecoder.batch_decode": "syndromes",
    "TwoStageDecoder.decode": "syndrome",
    "TwoStageDecoder.batch_decode": "syndromes",
    "AmbiguityClusterDecoder.decode": "syndrome",
    "AmbiguityClusterDecoder.batch_decode": "syndromes",
}


@pytest.mark.parametrize("name", sorted(CONSTRUCTOR_SPECS))
def test_public_constructor_arity_is_unchanged(name):
    cls = getattr(qd, name, None)
    assert cls is not None, f"public class {name} is missing from the installed package. {STALE_WHEEL_HINT}"
    try:
        actual = signature_spec(cls.__init__)
    except (TypeError, ValueError) as exc:  # pragma: no cover - would mean the wrapper vanished
        pytest.fail(
            f"{name}.__init__ is no longer introspectable ({exc}). The pure-Python wrapper in "
            "__init__.py has been replaced by the raw PyO3 class, which silently drops every "
            "default this table records."
        )
    assert actual == CONSTRUCTOR_SPECS[name], (
        f"{name}.__init__ parameters changed\n"
        f"  recorded: ({CONSTRUCTOR_SPECS[name]})\n"
        f"  actual:   ({actual})\n"
        "Renaming, reordering, or promoting a parameter to required breaks every caller. "
        "If this is intentional, update this table together with a CHANGELOG entry and a "
        "SemVer decision (see docs/SEMVER_POLICY.md, docs/MIGRATION.md)."
    )


@pytest.mark.parametrize("name", sorted(FUNCTION_SPECS))
def test_public_function_arity_is_unchanged(name):
    fn = getattr(qd, name, None)
    assert fn is not None, f"public function {name} is missing from the installed package. {STALE_WHEEL_HINT}"
    assert callable(fn), f"{name} resolved to {type(fn).__name__}, not a callable"
    actual = signature_spec(fn)
    assert actual == FUNCTION_SPECS[name], (
        f"{name}() parameters changed\n"
        f"  recorded: ({FUNCTION_SPECS[name]})\n"
        f"  actual:   ({actual})\n"
        "Update this table only alongside a CHANGELOG entry and a SemVer decision."
    )


@pytest.mark.parametrize("dotted", sorted(METHOD_SPECS))
def test_public_method_arity_is_unchanged(dotted):
    cls_name, meth_name = dotted.split(".")
    cls = getattr(qd, cls_name, None)
    assert cls is not None, f"public class {cls_name} is missing. {STALE_WHEEL_HINT}"
    meth = getattr(cls, meth_name, None)
    assert meth is not None, (
        f"{dotted} is missing. Method-name drift is invisible to a presence check: "
        "the class still imports, the call site raises AttributeError at runtime."
    )
    actual = signature_spec(meth)
    assert actual == METHOD_SPECS[dotted], (
        f"{dotted}() parameters changed\n  recorded: ({METHOD_SPECS[dotted]})\n  actual:   ({actual})"
    )


# ---------------------------------------------------------------------------
# Method-name spelling
# ---------------------------------------------------------------------------
BATCH_CAPABLE = [
    "BlossomDecoder",
    "SparseBlossomDecoder",
    "UnionFindDecoder",
    "FastUnionFindDecoder",
    "BPOSDDecoder",
    "CPUBatchDecoder",
    "BatchDecoder",
]


@pytest.mark.parametrize("name", BATCH_CAPABLE)
def test_batch_method_is_spelled_batch_decode(name):
    """The batch entry point is ``batch_decode``, never ``decode_batch``.

    Both spellings read naturally, which is exactly why the wrong one keeps
    getting written: ``python/qector_decoder_v3/cli.py:70`` calls
    ``dec.decode_batch(syndromes)``, so ``qector decode`` raises AttributeError
    on any multi-shot input file. Pinning the spelling here means the next such
    call site is caught by the suite rather than by a user.
    """
    cls = getattr(qd, name, None)
    assert cls is not None, f"{name} is missing. {STALE_WHEEL_HINT}"
    assert hasattr(cls, "batch_decode"), f"{name} lost its batch_decode method"
    assert not hasattr(cls, "decode_batch"), (
        f"{name} grew a decode_batch alias. Two spellings for one operation is how the "
        "cli.py:70 bug survived review; keep exactly one."
    )


# ---------------------------------------------------------------------------
# Packaging entry points
# ---------------------------------------------------------------------------
def _load_pyproject() -> dict:
    if not PYPROJECT.is_file():
        pytest.skip(f"no pyproject.toml at {PYPROJECT} (running outside a source checkout)")
    try:
        import tomllib
    except ModuleNotFoundError:  # pragma: no cover - Python < 3.11
        tomllib = pytest.importorskip("tomli")
    return tomllib.loads(PYPROJECT.read_text(encoding="utf-8"))


def _declared_targets() -> list[tuple[str, str]]:
    """``[(label, "module:attr" or "module"), ...]`` from every entry-point table."""
    project = _load_pyproject().get("project", {})
    out: list[tuple[str, str]] = []
    for script, target in project.get("scripts", {}).items():
        out.append((f"scripts/{script}", target))
    for group, table in project.get("entry-points", {}).items():
        for ep, target in table.items():
            out.append((f"{group}/{ep}", target))
    return out


def test_pyproject_declares_the_expected_entry_point_groups():
    """A dropped entry-point table is a silent feature removal.

    ``qector-doctor`` and the sinter/qiskit plugin registrations are the only
    way those integrations are discoverable; nothing imports them, so nothing
    else notices when they disappear from the packaging metadata.
    """
    project = _load_pyproject().get("project", {})
    scripts = project.get("scripts", {})
    for expected in ("qector", "qector-bench-quick", "qector-doctor"):
        assert expected in scripts, f"console script {expected!r} is no longer declared in pyproject.toml"
    groups = project.get("entry-points", {})
    assert "sinter_decoder" in groups, "the sinter_decoder entry-point group was removed"
    assert "qiskit.qec" in groups, "the qiskit.qec entry-point group was removed"


@pytest.mark.parametrize("label,target", _declared_targets(), ids=lambda v: v if isinstance(v, str) else "")
def test_declared_entry_points_resolve(label, target):
    """Every declared entry point must import and expose its attribute.

    An entry point is resolved by the *installed distribution*, not by the
    source tree, so this is exactly where a file that exists in the repo but
    was left out of the wheel shows up. Nothing else in the suite imports these
    modules: a broken console script otherwise reaches users intact.
    """
    module_name, _, attr = target.partition(":")
    try:
        module = importlib.import_module(module_name)
    except ImportError as exc:
        pytest.fail(
            f"entry point {label} = {target!r} does not import: {type(exc).__name__}: {exc}. {STALE_WHEEL_HINT}"
        )
    if attr:
        obj = getattr(module, attr, None)
        assert obj is not None, f"entry point {label} = {target!r}: module has no attribute {attr!r}"
        assert callable(obj), f"entry point {label} = {target!r}: {attr!r} is not callable"


# ---------------------------------------------------------------------------
# Documented import spellings
# ---------------------------------------------------------------------------
IMPORT_SPELLINGS = [
    # (module, attribute-or-None, why it is documented)
    ("qector_decoder_v3.pymatching", "Matching", "PyMatching drop-in shim (API-02)"),
    ("qector_decoder_v3.pymatching_compat", "Matching", "the shim's real implementation"),
    ("qector_decoder_v3.colour_code", "ColourCodeDecoder", "C1-04 colour-code entry point"),
    ("qector_decoder_v3.colour_code", "colour_codes_from_dem", "C1-04 helper, in __all__"),
    ("qector_decoder_v3.cli", "main", "the `qector` console script (C3-01)"),
    ("qector_decoder_v3.doctor", "main", "the `qector-doctor` console script (todo8 0.5)"),
    ("qector_decoder_v3.doctor", "run_checks", "programmatic diagnostic, used by support bundles"),
    ("qector_decoder_v3.doctor", "Check", "the per-check record type run_checks returns"),
    ("qector_decoder_v3.bench_quick", "main", "the `qector-bench-quick` console script"),
    ("qector_decoder_v3.sinter_compat", "QectorSinterDecoder", "target of 5 sinter entry points"),
    ("qector_decoder_v3.qiskit_plugin", None, "target of the qiskit.qec entry point"),
]


@pytest.mark.parametrize(
    "module_name,attr,why",
    IMPORT_SPELLINGS,
    ids=[f"{m}.{a}" if a else m for m, a, _ in IMPORT_SPELLINGS],
)
def test_documented_import_spellings_resolve(module_name, attr, why):
    """`import x.y` and `x.y` as an attribute are different contracts.

    API-02 was exactly this: ``qd.pymatching.Matching`` worked because
    ``__init__.py`` bound the name, while ``from qector_decoder_v3.pymatching
    import Matching`` -- the spelling a PyMatching user actually types -- failed,
    because no such submodule existed until ``pymatching.py`` was added.
    """
    try:
        module = importlib.import_module(module_name)
    except ImportError as exc:
        pytest.fail(f"documented import {module_name!r} ({why}) failed: {exc}. {STALE_WHEEL_HINT}")
    if attr is not None:
        assert hasattr(module, attr), f"{module_name}.{attr} ({why}) is gone"


# ---------------------------------------------------------------------------
# from_circuit: the one-call entry point
# ---------------------------------------------------------------------------
DOCUMENTED_DECODER_TYPES = [
    ("blossom", "BlossomDecoder"),
    ("sparse_blossom", "SparseBlossomDecoder"),
    ("bposd", "BPOSDDecoder"),
    ("belief_match", "BeliefMatching"),
    ("colour_code", "ColourCodeDecoder"),
    ("auto", "AutoDecoder"),
    ("ambig_cluster", "AmbiguityClusterDecoder"),
]


@pytest.fixture(scope="module")
def small_circuit():
    stim = pytest.importorskip("stim", reason="from_circuit is documented to require stim")
    return stim.Circuit.generated(
        "surface_code:rotated_memory_z", distance=3, rounds=3, after_clifford_depolarization=0.005
    )


@pytest.mark.parametrize("decoder_type,expected_cls", DOCUMENTED_DECODER_TYPES)
def test_from_circuit_builds_every_documented_decoder_type(small_circuit, decoder_type, expected_cls):
    """Each ``decoder_type`` named in the ``from_circuit`` docstring must build.

    The dispatch is a chain of string comparisons with alias normalisation
    (``lower()``, ``_`` and ``-`` stripped), so a typo in a branch is not a
    syntax error -- it falls through to ``ValueError: Unknown decoder_type``
    for one type only, which no import-level test can see.
    """
    dec = qd.from_circuit(small_circuit, decoder_type=decoder_type)
    assert type(dec).__name__ == expected_cls, (
        f"from_circuit(decoder_type={decoder_type!r}) returned {type(dec).__name__}, expected {expected_cls}"
    )


def test_from_circuit_two_stage_requires_check_types(small_circuit):
    """``two_stage`` is the one type that cannot be inferred from the circuit.

    It must say so instead of raising a TypeError from deep inside the
    constructor -- the message names the argument and its encoding.
    """
    with pytest.raises(ValueError, match="check_types"):
        qd.from_circuit(small_circuit, decoder_type="two_stage")


def test_from_circuit_rejects_unknown_decoder_type(small_circuit):
    with pytest.raises(ValueError, match="Unknown decoder_type"):
        qd.from_circuit(small_circuit, decoder_type="definitely-not-a-decoder")


# ---------------------------------------------------------------------------
# Return-type contract for the 0.7.0 decoder families
# ---------------------------------------------------------------------------
def _repetition_chain(n=8):
    c2q = [[i, i + 1] for i in range(n - 1)]
    H = np.zeros((n - 1, n), dtype=np.uint8)
    for ci, qs in enumerate(c2q):
        for q in qs:
            H[ci, q] ^= 1
    syn = np.zeros(n - 1, dtype=np.uint8)
    syn[2] = 1
    syn[5] = 1
    return c2q, H, syn


@pytest.mark.parametrize("name", ["TwoStageDecoder", "AmbiguityClusterDecoder"])
def test_new_decoder_families_return_a_uint8_ndarray(name):
    """0.7.0 fixed both of these returning ``bytes``; they must not regress.

    PyO3 converts a bare ``Vec<u8>`` to Python ``bytes``, which indexes, has a
    length, and compares equal to nothing a caller expects. Seven other decoder
    families return a ``uint8`` ndarray, so a ``bytes`` return made these two
    non-substitutable for the decoder they replace -- while every shape and
    length assertion still passed.
    """
    c2q, H, syn = _repetition_chain()
    n = H.shape[1]
    if name == "TwoStageDecoder":
        types = [i % 2 == 0 for i in range(len(c2q))]
        dec = qd.TwoStageDecoder(c2q, types, n)
    else:
        dec = qd.AmbiguityClusterDecoder(c2q, n)
    corr = dec.decode(syn)
    assert isinstance(corr, np.ndarray), f"{name}.decode returned {type(corr).__name__}, expected np.ndarray"
    assert corr.dtype == np.uint8, f"{name}.decode returned dtype {corr.dtype}, expected uint8"
    assert corr.shape == (n,), f"{name}.decode returned shape {corr.shape}, expected ({n},)"


@pytest.mark.parametrize("name", ["TwoStageDecoder", "AmbiguityClusterDecoder"])
def test_new_decoder_families_expose_batch_decode(name):
    """CHANGELOG [Unreleased] advertises ``batch_decode`` on both families.

    The native pyclasses do implement it (``src/two_stage_decoder.rs:362``,
    ``src/ambig_cluster.rs:802``), but the pure-Python wrappers in
    ``__init__.py`` forward only ``decode``, so the method is unreachable from
    Python. Either the wrappers gain a two-line forwarder or the CHANGELOG
    entry is wrong; this test is what forces one of the two.
    """
    c2q, H, syn = _repetition_chain()
    n = H.shape[1]
    if name == "TwoStageDecoder":
        types = [i % 2 == 0 for i in range(len(c2q))]
        dec = qd.TwoStageDecoder(c2q, types, n)
    else:
        dec = qd.AmbiguityClusterDecoder(c2q, n)
    assert hasattr(dec, "batch_decode"), (
        f"{name} has no batch_decode. The native class implements it; the Python wrapper in "
        "__init__.py does not forward it."
    )
    out = dec.batch_decode(np.stack([syn, syn]))
    assert isinstance(out, np.ndarray) and out.shape == (2, n)


# ---------------------------------------------------------------------------
# Drop-in shims
# ---------------------------------------------------------------------------
MATCHING_SURFACE = {
    "decode": "syndrome",
    "decode_batch": "shots",
    "decode_to_edges_array": "syndrome",
    "add_edge": "node1, node2, fault_ids=, weight=, **_",
    "add_boundary_edge": "node, fault_ids=, weight=, **_",
    "from_check_matrix": "H, weights=, faults_matrix=, **_",
    "from_detector_error_model": "dem",
    "edges": "",
}


@pytest.mark.parametrize("meth", sorted(MATCHING_SURFACE))
def test_matching_shim_keeps_the_pymatching_call_shape(meth):
    """The shim's value is that PyMatching code runs unmodified.

    Its parameter *names* are therefore load-bearing -- PyMatching users pass
    ``weight=`` and ``fault_ids=`` by keyword. ``decode_batch`` is PyMatching's
    spelling and is correct here, unlike on QECTOR's own decoders.
    """
    from qector_decoder_v3.pymatching import Matching

    fn = getattr(Matching, meth, None)
    assert fn is not None, f"Matching.{meth} is gone; PyMatching code that calls it will not port"
    actual = signature_spec(fn)
    assert actual == MATCHING_SURFACE[meth], (
        f"Matching.{meth}() parameters changed\n"
        f"  recorded: ({MATCHING_SURFACE[meth]})\n"
        f"  actual:   ({actual})\n"
        "PyMatching callers pass these by keyword; renaming one breaks the drop-in claim."
    )


def test_matching_shim_and_top_level_name_are_the_same_class():
    """Three documented spellings, one implementation.

    ``qd.Matching``, ``qd.pymatching.Matching`` and
    ``qector_decoder_v3.pymatching_compat.Matching`` must not drift into two
    classes with subtly different behaviour.
    """
    from qector_decoder_v3.pymatching import Matching as shim
    from qector_decoder_v3.pymatching_compat import Matching as impl

    assert shim is impl
    assert qd.Matching is impl


COLOUR_CODE_SURFACE = {
    # v1.0 additive: `method` (cluster_bposd / bposd) is an optional keyword arg
    # appended at the end; every pre-1.0 call site keeps working unchanged.
    "__init__": "dem, max_iter=, osd_order=, method=",
    "from_stim_circuit": "circuit, max_iter=, osd_order=",
    "decode": "syndrome",
    "decode_batch": "syndromes",
}


@pytest.mark.parametrize("meth", sorted(COLOUR_CODE_SURFACE))
def test_colour_code_decoder_surface(meth):
    """C1-04's entry point is ``from_stim_circuit`` on an *undecomposed* DEM.

    Matching cannot decode colour codes at all -- Stim raises on
    ``decompose_errors=True`` for ``color_code:memory_xyz`` at d>=5 -- so this
    constructor pair is the only supported path, and its shape is documented in
    the module docstring and in ``from_circuit(decoder_type="colour_code")``.
    """
    from qector_decoder_v3.colour_code import ColourCodeDecoder

    fn = getattr(ColourCodeDecoder, meth, None)
    assert fn is not None, f"ColourCodeDecoder.{meth} is gone"
    actual = signature_spec(fn)
    assert actual == COLOUR_CODE_SURFACE[meth], (
        f"ColourCodeDecoder.{meth}() parameters changed\n"
        f"  recorded: ({COLOUR_CODE_SURFACE[meth]})\n"
        f"  actual:   ({actual})"
    )


# ---------------------------------------------------------------------------
# qector-doctor
# ---------------------------------------------------------------------------
def test_doctor_run_checks_returns_structured_results():
    """``--json`` output is a support-bundle format; its keys are a contract.

    GPU probes are skipped: the OpenCL probe spawns a child process and can
    take seconds, which does not belong in an API-shape test.
    """
    doctor = importlib.import_module("qector_decoder_v3.doctor")
    checks = doctor.run_checks(repo=str(REPO_ROOT), skip_gpu=True)
    assert checks, "run_checks returned nothing"
    for c in checks:
        d = c.as_dict()
        assert set(d) == {"check", "status", "detail", "remedy"}, (
            f"doctor JSON keys changed to {sorted(d)}; support tooling parses these"
        )
        assert d["status"] in (doctor.PASS, doctor.WARN, doctor.FAIL)
    names = {c.name for c in checks}
    for required in ("package", "native-core", "working-tree", "decode", "licence-tier"):
        assert required in names, f"doctor lost its {required!r} check ({sorted(names)})"


def test_doctor_main_accepts_its_documented_flags():
    """Exit status is the contract CI and support scripts consume.

    0 unless a check FAILs; ``--strict`` additionally fails on WARN. A machine
    with no GPU on the Community tier is a healthy machine, not a failure.
    """
    doctor = importlib.import_module("qector_decoder_v3.doctor")
    assert signature_spec(doctor.main) == "argv="
    rc = doctor.main(["--json", "--skip-gpu", "--repo", str(REPO_ROOT)])
    assert rc in (0, 1), f"qector-doctor returned exit status {rc}, expected 0 or 1"
