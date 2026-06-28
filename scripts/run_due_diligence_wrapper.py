#!/usr/bin/env python
"""Wrapper for run_due_diligence_bundle.py that removes the local python/ path
injection so that the installed PyPI wheel is used instead of local source.
This is needed when testing the installed wheel on a dev machine where
the local python/ directory is present.
"""
from __future__ import annotations

import os
import sys

_REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_local_py = os.path.normcase(os.path.join(_REPO, "python"))

# Remove any local python/ path already in sys.path
sys.path = [p for p in sys.path if os.path.normcase(p) != _local_py]

bundle_path = os.path.join(_REPO, "scripts", "run_due_diligence_bundle.py")
with open(bundle_path, encoding="utf-8") as f:
    src = f.read()

# Strip the local path injection line from the source before exec-ing
# This line is: sys.path.insert(0, os.path.join(_REPO, 'python'))
src = src.replace(
    "sys.path.insert(0, os.path.join(_REPO, \"python\"))",
    "pass  # path injection disabled by wrapper (use installed wheel)",
)

# Re-inject sys.argv so argparse works (strip off this wrapper script's name)
sys.argv = [bundle_path] + sys.argv[1:]

bundle_globals = {
    "__name__": "__main__",
    "__file__": bundle_path,
    "__spec__": None,
}
exec(compile(src, bundle_path, "exec"), bundle_globals)  # noqa: S102
