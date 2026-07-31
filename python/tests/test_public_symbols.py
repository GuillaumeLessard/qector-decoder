"""Public API surface contract (todo8 item 0.2 / phase 8.2).

Two releases (0.6.5 and 0.6.7) were yanked because a public symbol went missing
from a published wheel. Nothing caught it, and the reason is structural:
``__init__.py`` wraps every native lookup in ``_guard()``, which substitutes an
``_Unavailable`` stub when the symbol is absent. The stub imports cleanly, has the
right ``__name__``, and only raises when someone *constructs* it — so
``import qector_decoder_v3`` succeeds, a smoke test that merely imports succeeds,
and the breakage reaches users.

These tests close that hole:

* every name in ``expected_symbols.txt`` is present on the package;
* no required name has silently degraded to a ``_guard()`` stub;
* the package exposes nothing public that is *not* in the file, so growing the
  API surface is a deliberate, reviewable act;
* ``__all__`` is self-consistent (no entry that ``from ... import *`` would fail
  on, no duplicates);
* the headline decoders actually construct and decode, because "the attribute
  exists" is not the property users depend on.
"""

from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import pytest

os.environ.setdefault("QECTOR_SILENT", "1")

import qector_decoder_v3 as qd

EXPECTED_FILE = Path(__file__).with_name("expected_symbols.txt")

# Build-dependent backends. They must still be *present* as attributes (the
# package always defines them, falling back to a stub or a probe returning
# False), but a CPU-only wheel is allowed to have them non-functional.
OPTIONAL_SYMBOLS = frozenset(
    {
        "CUDABatchDecoder",
        "OpenCLBatchDecoder",
        "cuda_is_available",
        "opencl_is_available",
        "run_grpc_server",
    }
)


def _load_sections() -> tuple[set[str], set[str]]:
    """Return ``(declared_api, reachable_only)`` from the contract file.

    Everything before the ``[reachable]`` marker is the supported API and must
    also appear in ``__all__``; everything after is present-but-not-promoted.
    """
    api: set[str] = set()
    reachable: set[str] = set()
    target = api
    for raw in EXPECTED_FILE.read_text(encoding="utf-8").splitlines():
        line = raw.split("#", 1)[0].strip()
        if not line:
            continue
        if line == "[reachable]":
            target = reachable
            continue
        target.add(line)
    return api, reachable


def _load_expected() -> set[str]:
    """Every name that must be present, of either kind."""
    api, reachable = _load_sections()
    return api | reachable


def _is_guard_stub(obj: object) -> bool:
    """True iff ``obj`` is the ``_Unavailable`` placeholder ``_guard()`` returns."""
    return isinstance(obj, type) and repr(obj).startswith("<unavailable ")


def test_expected_symbols_file_is_readable_and_nonempty():
    expected = _load_expected()
    assert len(expected) > 50, f"expected_symbols.txt looks truncated ({len(expected)} names)"


def test_every_expected_symbol_is_present():
    missing = sorted(name for name in _load_expected() if not hasattr(qd, name))
    assert not missing, (
        "public symbols missing from the installed package: "
        f"{missing}. This is the 0.6.5 / 0.6.7 yank failure mode — either the "
        "build dropped them, or they were removed without updating "
        "expected_symbols.txt."
    )


def test_no_required_symbol_degraded_to_a_stub():
    """A `_guard()` stub imports fine and raises only on construction.

    That is precisely why this check exists: presence is not availability.
    """
    stubs = sorted(
        name for name in _load_expected() if name not in OPTIONAL_SYMBOLS and _is_guard_stub(getattr(qd, name, None))
    )
    assert not stubs, (
        f"these required symbols resolved to _guard() stubs: {stubs}. The wheel "
        "was built without the native symbols they need; they would raise "
        "RuntimeError on first use."
    )


def test_public_surface_has_not_grown_silently():
    """Anything public and not in the contract file fails.

    Catches the reverse error: shipping API that was never reviewed, then being
    unable to change it without a breaking release.
    """
    expected = _load_expected()
    actual_public = {name for name in dir(qd) if not name.startswith("_") and name not in {"annotations", "math", "np"}}
    undeclared = sorted(actual_public - expected)
    assert not undeclared, (
        f"public names not declared in expected_symbols.txt: {undeclared}. "
        "Add them there in the same commit that exports them (SemVer: this is a "
        "minor-version surface change), or make them private."
    )


def test_all_is_consistent():
    dupes = sorted({n for n in qd.__all__ if qd.__all__.count(n) > 1})
    assert not dupes, f"__all__ contains duplicates: {dupes}"
    broken = sorted(n for n in qd.__all__ if not hasattr(qd, n))
    assert not broken, f"__all__ names that do not resolve: {broken}"


def test_all_matches_the_declared_contract():
    api, _reachable = _load_sections()
    missing_from_all = sorted(api - set(qd.__all__))
    assert not missing_from_all, (
        f"declared public symbols absent from __all__: {missing_from_all} — "
        "`from qector_decoder_v3 import *` would not provide them."
    )


def test_reachable_only_names_stay_out_of_all():
    """The `[reachable]` section is a compatibility shim, not supported API.

    Promoting one into ``__all__`` should be a conscious decision, so it has to
    move sections in the contract file rather than drift in.
    """
    _api, reachable = _load_sections()
    promoted = sorted(reachable & set(qd.__all__))
    assert not promoted, (
        f"names listed as reachable-only appear in __all__: {promoted}. Move them "
        "into the supported section of expected_symbols.txt if that is intended."
    )


def test_no_stdlib_or_third_party_module_leaked_into_the_namespace():
    """Regression: `qector_decoder_v3.F` used to be `torch.nn.functional`.

    ``__init__.py`` states the intent explicitly ("imported locally ... to avoid
    polluting the public module namespace"), but three imports escaped it:
    ``torch.nn.functional as F``, ``logging`` and ``typing.Optional``. A caller
    doing ``from qector_decoder_v3 import *`` got torch's functional namespace.
    """
    import types

    leaked = []
    for name in dir(qd):
        if name.startswith("_"):
            continue
        obj = getattr(qd, name, None)
        if isinstance(obj, types.ModuleType):
            mod = getattr(obj, "__name__", "")
            if not mod.startswith("qector_decoder_v3"):
                leaked.append(f"{name} -> {mod}")
    assert not leaked, (
        f"foreign modules exposed as public attributes: {leaked}. Import them under a leading-underscore alias."
    )


@pytest.mark.parametrize(
    "name",
    [
        "UnionFindDecoder",
        "FastUnionFindDecoder",
        "BlossomDecoder",
        "SparseBlossomDecoder",
        "BPOSDDecoder",
    ],
)
def test_headline_decoders_construct_and_decode(name):
    """Presence is not the contract users rely on — decoding is.

    A repetition-code chain: every qubit sits in at most two checks, so it is
    valid input for every decoder here, including the Union-Find family.
    """
    cls = getattr(qd, name)
    assert not _is_guard_stub(cls), f"{name} is a _guard() stub"
    n = 8
    c2q = [[i, i + 1] for i in range(n - 1)]
    dec = cls(c2q, n) if name != "BPOSDDecoder" else cls(c2q, n, 0.01)
    syn = np.zeros(n - 1, dtype=np.uint8)
    syn[2] = 1
    syn[5] = 1
    corr = np.asarray(dec.decode(syn), dtype=np.uint8)
    assert corr.shape == (n,), f"{name} returned shape {corr.shape}, expected ({n},)"
    H = np.zeros((n - 1, n), dtype=np.uint8)
    for ci, qs in enumerate(c2q):
        for q in qs:
            H[ci, q] ^= 1
    assert np.array_equal((H @ corr) & 1, syn), (
        f"{name} returned a correction that does not reproduce the syndrome: {corr}"
    )


def test_version_is_exposed_and_parseable():
    v = qd.__version__
    parts = v.split(".")
    assert len(parts) >= 3, f"__version__ {v!r} is not dotted-triple"
    assert all(p.isdigit() for p in parts[:3]), f"__version__ {v!r} has non-numeric components"
