"""The documented reproduction CLI commands parse and import cleanly.

``docs/REPRODUCE.md`` (and the broader benchmark/audit docs) point users at a set
of scripts in ``scripts/``. Running each with ``--help`` exercises its full import
chain and argparse setup without doing any heavy work: if a script's imports are
broken or its argument parser is malformed, ``--help`` returns non-zero. This is a
cheap, robust proof that the reproduction commands are wired correctly.
"""

import os
import subprocess
import sys

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


# Scripts referenced by docs/REPRODUCE.md that expose an argparse CLI.
# (run_scaling_benchmark.py is intentionally excluded: it is not an argparse CLI
# — it runs a full benchmark immediately and ignores --help, so a `--help` smoke
# test would execute the whole sweep rather than just parsing arguments.)
_REPRODUCE_SCRIPTS = [
    "run_competitive_benchmark.py",
    "competitive_belief_matching.py",
    "threshold_estimate.py",
]

# ... plus the newer driver scripts that must also parse/import cleanly.
_NEWER_SCRIPTS = [
    "competitive_stim_ler.py",
    "belief_extended.py",
    "weight_gap_analysis.py",
    "d15_mismatch_audit.py",
    "gpu_extensive_test.py",
    "native_memory_profile.py",
    "belief_reference_compare.py",
    "gpu_memory_profile.py",
    "auto_backend_calibrate.py",
    "leak_test.py",
    "run_due_diligence_bundle.py",
]

_ALL_SCRIPTS = sorted(set(_REPRODUCE_SCRIPTS + _NEWER_SCRIPTS))


@pytest.mark.parametrize("script", _ALL_SCRIPTS)
def test_script_help_parses(script):
    root = _repo_root()
    path = os.path.join(root, "scripts", script)
    if not os.path.isfile(path):
        pytest.skip(f"script not present: {script}")

    proc = subprocess.run(
        [sys.executable, os.path.join("scripts", script), "--help"],
        cwd=root,
        capture_output=True,
        text=True,
        timeout=60,
    )
    assert proc.returncode == 0, (
        f"`{script} --help` exited {proc.returncode}\nSTDOUT:\n{proc.stdout}\nSTDERR:\n{proc.stderr}"
    )
    # argparse prints a usage line on --help.
    combined = (proc.stdout + proc.stderr).lower()
    assert "usage" in combined, (
        f"{script} --help produced no usage text:\n{proc.stdout}"
    )
