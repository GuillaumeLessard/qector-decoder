"""A5-02: optional dependencies must degrade, not explode.

Every module that guards an optional import must remain importable and usable
when that dependency is absent. The historical failure mode was `except
RuntimeError` around an `import` statement: `ModuleNotFoundError` is an
`ImportError`, *not* a `RuntimeError`, so the guard caught nothing and the
absence of an optional extra became a hard import failure.

These tests simulate absence with a meta-path finder that raises
`ModuleNotFoundError` for the named modules, then re-import the module under
test from scratch.
"""

from __future__ import annotations

import builtins
import importlib
import sys

import pytest

# (module under test, optional dependencies to hide)
OPTIONAL_IMPORT_MATRIX = [
    ("qector_decoder_v3.sinter_compat", ["sinter"]),
    ("qector_decoder_v3.benchmarking", ["psutil"]),
    ("qector_decoder_v3.bench_quick", ["cpuinfo"]),
    ("qector_decoder_v3.gpu_backend", ["cupy"]),
    ("qector_decoder_v3.bp_cupy", ["cupy"]),
    ("qector_decoder_v3.qiskit_plugin", ["qiskit"]),
    ("qector_decoder_v3.stim_compat", ["stim"]),
]


class _BlockedFinder:
    """Meta-path finder that makes the named top-level modules unimportable."""

    def __init__(self, blocked: list[str]):
        self.blocked = set(blocked)

    def find_module(self, fullname, path=None):  # pragma: no cover - legacy API
        return None

    def find_spec(self, fullname, path=None, target=None):
        root = fullname.split(".")[0]
        if root in self.blocked:
            raise ModuleNotFoundError(f"No module named {root!r}", name=root)


def _reimport_without(module_name: str, blocked: list[str]):
    """Import ``module_name`` fresh with ``blocked`` unavailable."""
    finder = _BlockedFinder(blocked)
    saved = {k: v for k, v in sys.modules.items() if k == module_name or k.split(".")[0] in blocked}
    for k in saved:
        del sys.modules[k]
    sys.meta_path.insert(0, finder)
    try:
        return importlib.import_module(module_name)
    finally:
        sys.meta_path.remove(finder)
        sys.modules.update(saved)
        # Restoring sys.modules is not enough. Importing `pkg.sub` also rebinds
        # the fresh module onto its PARENT as `pkg.sub`, and nothing above undoes
        # that - so the throwaway module outlives this helper as an attribute of
        # the real package while sys.modules holds the original. The two then
        # disagree, permanently, for the rest of the session:
        #
        #   sys.modules["qector_decoder_v3.gpu_backend"]  -> original
        #   qector_decoder_v3.gpu_backend                 -> throwaway
        #
        # That broke test_routing.py's live-detection cases, and did so only in a
        # full-suite run and only on a machine without a GPU, which is why it
        # looked like a CI-only mystery. `routing` holds `from . import
        # gpu_backend as _gb` (the original), while monkeypatch - by either
        # spelling, since pytest's resolve() walks parents with getattr - reached
        # the throwaway. The patch applied to a module nothing under test used.
        for name, module in saved.items():
            parent_name, _, child = name.rpartition(".")
            parent = sys.modules.get(parent_name) if parent_name else None
            if parent is not None and getattr(parent, child, None) is not module:
                setattr(parent, child, module)


@pytest.mark.parametrize(("module_name", "blocked"), OPTIONAL_IMPORT_MATRIX)
def test_module_imports_without_optional_dependency(module_name, blocked):
    """The module must import cleanly with its optional dependency missing."""
    mod = _reimport_without(module_name, blocked)
    assert mod is not None
    assert mod.__name__ == module_name


def test_sinter_compat_reports_absence_instead_of_raising():
    """A5-01 regression: this exact module raised ModuleNotFoundError at import."""
    mod = _reimport_without("qector_decoder_v3.sinter_compat", ["sinter"])
    assert mod._HAS_SINTER is False
    # And the public factory must raise a *actionable* ImportError, not crash.
    with pytest.raises(ImportError, match="sinter"):
        mod.qector_sinter_decoders()


def test_benchmarking_captures_environment_without_optional_tools(monkeypatch):
    """capture_environment must survive missing rustc/cargo/git and missing psutil."""
    import qector_decoder_v3.benchmarking as bench

    real_run = bench.subprocess.run

    def _no_binaries(cmd, *a, **kw):
        raise FileNotFoundError(f"no such binary: {cmd[0]}")

    monkeypatch.setattr(bench.subprocess, "run", _no_binaries)
    try:
        env = bench.capture_environment()
    finally:
        monkeypatch.setattr(bench.subprocess, "run", real_run)

    assert env["rust_version"] is None
    assert env["cargo_version"] is None
    assert env["git_commit"] == "unknown"
    assert env["python_version"]
    assert "numpy_version" in env


def test_pkg_version_returns_none_for_absent_package():
    """_pkg_version must return None, not raise, for a package that isn't installed."""
    from qector_decoder_v3.benchmarking import _pkg_version

    assert _pkg_version("definitely_not_a_real_package_xyz") is None


def test_no_bare_except_runtimeerror_around_imports():
    """Static guard: `except RuntimeError` must not reappear in the package.

    `ImportError` is not a subclass of `RuntimeError`, so this pattern silently
    fails to guard anything. Comments mentioning it are fine; code is not.

    Deliberately reads the **repo source**, not the installed copy: the source
    tree is what is under version control and what review sees. (The installed
    package can be a stale copy — `maturin develop` copies rather than links.)
    """
    import pathlib

    pkg = pathlib.Path(__file__).resolve().parent.parent / "qector_decoder_v3"
    if not pkg.is_dir():  # pragma: no cover - running against an installed wheel only
        pytest.skip("repo source tree not available")
    offenders = []
    for path in sorted(pkg.glob("*.py")):
        for lineno, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
            stripped = line.strip()
            if stripped.startswith("#"):
                continue
            if stripped.startswith("except RuntimeError"):
                offenders.append(f"{path.name}:{lineno}")
    assert not offenders, "bare `except RuntimeError` found at: " + ", ".join(offenders)
