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
    text = open(license_path, "r", encoding="utf-8").read()
    assert len(text) > 100, f"LICENSE is suspiciously short ({len(text)} chars)"
    # The QECTOR Decoder Source-Available License: proprietary, with a free
    # non-commercial grant and a paid commercial requirement.
    assert "QECTOR Decoder Source-Available License" in text
    assert "All rights reserved" in text
    assert "Commercial Use Requires a Paid License" in text


def test_pyproject_declares_license():
    root = _repo_root()
    with open(os.path.join(root, "pyproject.toml"), "rb") as fh:
        pyproject = tomllib.load(fh)

    project = pyproject["project"]
    license_field = project.get("license")
    # license points at the LICENSE file (PEP 621 table form)
    assert isinstance(license_field, dict) and (
        license_field.get("file") == "LICENSE" or "text" in license_field
    ), f"pyproject [project] license not declared via file/text: {license_field!r}"
    classifiers = project.get("classifiers", [])
    assert any(c == "License :: Other/Proprietary License" for c in classifiers), (
        "expected the proprietary trove classifier"
    )
