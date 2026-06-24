"""Version consistency across the packaging surfaces.

The single source-of-truth version must be identical in three places:

* ``pyproject.toml`` ``[project] version`` (the wheel/sdist metadata),
* ``Cargo.toml`` ``[package] version`` (the Rust crate that compiles the
  extension), and
* ``qector_decoder_v3.__version__`` (what users actually see at runtime).

A drift between any of these ships a wheel whose advertised version does not
match the compiled core, so this test fails loudly if they disagree.
"""

import os
import tomllib

import qector_decoder_v3


def _repo_root():
    """Walk up from this test file to the directory holding pyproject.toml."""
    here = os.path.dirname(os.path.abspath(__file__))
    cur = here
    while True:
        if os.path.isfile(os.path.join(cur, "pyproject.toml")):
            return cur
        parent = os.path.dirname(cur)
        if parent == cur:
            raise AssertionError(f"pyproject.toml not found walking up from {here}")
        cur = parent


def test_version_consistency():
    root = _repo_root()

    with open(os.path.join(root, "pyproject.toml"), "rb") as fh:
        pyproject = tomllib.load(fh)
    with open(os.path.join(root, "Cargo.toml"), "rb") as fh:
        cargo = tomllib.load(fh)

    py_version = pyproject["project"]["version"]
    cargo_version = cargo["package"]["version"]
    runtime_version = qector_decoder_v3.__version__

    # Non-empty, dotted-looking versions.
    assert py_version and "." in py_version, py_version
    assert cargo_version and "." in cargo_version, cargo_version
    assert runtime_version and "." in runtime_version, runtime_version

    assert py_version == cargo_version, f"pyproject version {py_version!r} != Cargo version {cargo_version!r}"
    assert py_version == runtime_version, f"pyproject version {py_version!r} != runtime __version__ {runtime_version!r}"
    assert cargo_version == runtime_version, (
        f"Cargo version {cargo_version!r} != runtime __version__ {runtime_version!r}"
    )
