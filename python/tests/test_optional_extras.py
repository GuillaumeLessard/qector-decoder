"""Optional-dependency extras are declared correctly in pyproject.toml.

``pip install qector-decoder-v3[stim]`` (and friends) must resolve, which means
the extras groups have to exist and list the right packages. This test parses
``pyproject.toml`` and asserts on what is *actually* declared (the groups read
from the file at import time), not on an invented expectation.
"""

import os
import tomllib

import pytest


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


@pytest.fixture(scope="module")
def extras():
    with open(os.path.join(_repo_root(), "pyproject.toml"), "rb") as fh:
        pyproject = tomllib.load(fh)
    opt = pyproject["project"].get("optional-dependencies")
    assert opt is not None, "[project.optional-dependencies] is missing"
    assert isinstance(opt, dict) and opt, "optional-dependencies must be a non-empty table"
    return opt


def test_expected_extras_groups_present(extras):
    # These four are the ecosystem/GPU groups the docs and INSTALL.md reference.
    expected = {"stim", "bench", "cuda", "opencl"}
    present = expected & set(extras)
    assert present, f"none of {sorted(expected)} found in {sorted(extras)}"
    # All four genuinely exist in this project; assert the full set is declared.
    assert expected.issubset(set(extras)), (
        f"missing extras groups: {sorted(expected - set(extras))}"
    )


def test_stim_group_lists_stim_and_pymatching(extras):
    assert "stim" in extras, sorted(extras)
    deps = extras["stim"]
    names = {_dist_name(spec) for spec in deps}
    assert "stim" in names, f"stim extra does not list stim: {deps}"
    assert "pymatching" in names, f"stim extra does not list pymatching: {deps}"


def _dist_name(requirement: str) -> str:
    """Extract the distribution name from a requirement string like 'stim>=1.12'."""
    name = requirement.strip()
    for sep in ("[", ">", "<", "=", "!", "~", " ", ";"):
        idx = name.find(sep)
        if idx != -1:
            name = name[:idx]
    return name.strip().lower()
