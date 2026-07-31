"""License is present and declared.

A distributable package must ship its license file and declare the license in
its metadata. This test asserts:

* a non-empty ``LICENSE`` file exists at the repo root, and
* ``pyproject.toml`` declares a license either through the ``[project] license``
  key or a ``License ::`` trove classifier.
"""

import os

import tomllib


def _repo_root():
    here = os.path.dirname(os.path.abspath(__file__))
    cur = here
    while True:
        if os.path.isfile(os.path.join(cur, "pyproject.toml")):
            return cur
        parent = os.path.dirname(cur)
        if parent == cur:
            raise AssertionError(f"pyproject.toml not found walking up from {here}")
        cur = parent


def test_license_file_exists_and_non_empty():
    root = _repo_root()
    license_path = os.path.join(root, "LICENSE")
    assert os.path.isfile(license_path), f"LICENSE missing at {license_path}"
    with open(license_path, "r", encoding="utf-8") as fh:
        text = fh.read()
    assert len(text) > 100, f"LICENSE is suspiciously short ({len(text)} chars)"
    text_lower = text.lower()

    # Assert the *terms* the project depends on, not one revision's wording.
    # v0.7.0 replaced the bespoke "QECTOR SOURCE-AVAILABLE LICENSE Version 1.0"
    # (which said "All Rights Reserved") with the standard PolyForm
    # Noncommercial License 1.0.0 plus a commercial-use rider. Pinning the old
    # phrase made a deliberate, reviewed licensing change look like a failure,
    # while still not checking that any of it survived.
    assert "qector" in text_lower
    assert "copyright" in text_lower, "LICENSE must carry a copyright notice"
    assert "guillaume lessard" in text_lower, "LICENSE must name the copyright holder"

    # Source-available, not open source: either the bespoke heading or PolyForm.
    assert "source-available" in text_lower or "polyform" in text_lower, (
        "LICENSE must identify a source-available / noncommercial license"
    )

    # Commercial use is gated behind a paid license.
    assert "commercial" in text_lower and "license" in text_lower

    # The Rust core is carved out of the distributed grant.
    assert "rust core" in text_lower, (
        "LICENSE must state that the proprietary Rust core is not covered by this grant"
    )


def test_pyproject_declares_license():
    root = _repo_root()
    with open(os.path.join(root, "pyproject.toml"), "rb") as fh:
        pyproject = tomllib.load(fh)

    project = pyproject["project"]
    license_field = project.get("license")
    # license points at the LICENSE file (PEP 621 table form)
    assert isinstance(license_field, dict) and (license_field.get("file") == "LICENSE" or "text" in license_field), (
        f"pyproject [project] license not declared via file/text: {license_field!r}"
    )
    classifiers = project.get("classifiers", [])
    assert any(c == "License :: Other/Proprietary License" for c in classifiers), (
        "expected the proprietary trove classifier"
    )
