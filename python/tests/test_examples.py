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

    # Only inject the source python/ directory onto PYTHONPATH when the compiled
    # extension (.pyd/.so) is already importable from the *installed* wheel.  If
    # we unconditionally prepend the uncompiled source tree, Python finds the
    # source __init__.py first and then fails to import the Rust extension (which
    # lives inside the wheel's site-packages, not the source tree), breaking every
    # subprocess that tries to import qector_decoder_v3.
    env = os.environ.copy()
    try:
        import qector_decoder_v3  # noqa: F401  – test whether installed wheel works
        # Wheel is importable: do NOT shadow it with the raw source tree.
        # Leave PYTHONPATH as-is so subprocess inherits the venv site-packages.
    except ImportError:
        # No installed wheel yet (e.g. CI before build step): fall back to source.
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
            f"Example {example} failed with return code {res.returncode}.\n"
            f"STDOUT:\n{res.stdout}\nSTDERR:\n{res.stderr}"
        )
