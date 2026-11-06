"""Regression tests for verifying that all example scripts run successfully."""

import os
import subprocess
import sys
from pathlib import Path

import pytest


def test_run_examples():
    # Resolve the repository root directory
    # __file__ is in python/tests/test_examples.py -> parent.parent.parent = repo root
    try:
        repo_root = Path(__file__).resolve().parent.parent.parent
    except RuntimeError:
        # Fallback to CWD if path resolution fails
        repo_root = Path.cwd()
        for _ in range(4):
            if (repo_root / "examples").is_dir():
                break
            repo_root = repo_root.parent

    examples_dir = repo_root / "examples"
    if not examples_dir.is_dir():
        pytest.skip(f"examples directory not found at {examples_dir}")

    # List of examples to run
    examples = [
        "example_basic.py",
        "example_batch.py",
        "example_streaming.py",
        "example_blossom.py",
        "example_codes_and_diagnostics.py",
        "example_stim_dem.py",
        "example_pymatching_and_backend.py",
        "example_advanced_decoders.py",
    ]

    # Put python/ on PYTHONPATH only for an in-tree build (`maturin develop`
    # leaves the compiled extension there). Otherwise the source tree, which has
    # no native module, would shadow the installed wheel -- the same failure
    # conftest.py documents for sys.path.
    env = os.environ.copy()
    pkg_dir = repo_root / "python" / "qector_decoder_v3"
    if any(pkg_dir.glob("qector_decoder_v3*.so")) or any(pkg_dir.glob("qector_decoder_v3*.pyd")):
        env["PYTHONPATH"] = str(pkg_dir.parent) + os.pathsep + env.get("PYTHONPATH", "")

    for example in examples:
        example_path = examples_dir / example
        assert example_path.exists(), f"Example script {example} does not exist at {example_path}"

        res = subprocess.run(
            [sys.executable, str(example_path)],
            capture_output=True,
            text=True,
            env=env,
            cwd=str(repo_root),
            check=False,
        )
        assert res.returncode == 0, (
            f"Example {example} failed with return code {res.returncode}.\nSTDOUT:\n{res.stdout}\nSTDERR:\n{res.stderr}"
        )
