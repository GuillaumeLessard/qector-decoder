"""Regression tests for verifying that all example scripts run successfully."""

import subprocess
import sys
import os
from pathlib import Path


def test_run_examples():
    # Resolve the repository root directory
    repo_root = Path(__file__).resolve().parent.parent.parent
    examples_dir = repo_root / "examples"

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
    env["PYTHONPATH"] = (
        str(repo_root / "python") + os.pathsep + env.get("PYTHONPATH", "")
    )

    for example in examples:
        example_path = examples_dir / example
        assert example_path.exists(), (
            f"Example script {example} does not exist at {example_path}"
        )

        res = subprocess.run(
            [sys.executable, str(example_path)],
            capture_output=True,
            text=True,
            env=env,
            cwd=str(repo_root),
        )
        assert res.returncode == 0, (
            f"Example {example} failed with return code {res.returncode}.\nSTDOUT:\n{res.stdout}\nSTDERR:\n{res.stderr}"
        )
