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
    except Exception:
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

    # Set PYTHONPATH to include the python/ directory in the repo
    env = os.environ.copy()
    python_dir = repo_root / "python"
    if python_dir.is_dir():
        env["PYTHONPATH"] = str(python_dir) + os.pathsep + env.get("PYTHONPATH", "")

    for example in examples:
        example_path = examples_dir / example
        assert example_path.exists(), f"Example script {example} does not exist at {example_path}"

        res = subprocess.run(
            [sys.executable, str(example_path)], capture_output=True, text=True, env=env, cwd=str(repo_root)
        )
        assert res.returncode == 0, (
            f"Example {example} failed with return code {res.returncode}.\nSTDOUT:\n{res.stdout}\nSTDERR:\n{res.stderr}"
        )
