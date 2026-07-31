"""Shared pytest configuration.

Guarantees that `qector_decoder_v3` resolves to the **installed** package (the
one containing the compiled PyO3 extension) rather than to the repo's
`python/qector_decoder_v3/` source directory, which has no `.pyd`/`.so`.

Why this file exists (todo7 V-13):

`test_comprehensive_suite.py` used to do

    sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

which put `python/` at position 0 of `sys.path`. That shadows the installed
package, so `import qector_decoder_v3` finds the source tree and then fails on
`import qector_decoder_v3.qector_decoder_v3` — the native module is not there.

In the parent process the failure was invisible, because the native module had
already been imported and cached in `sys.modules` before the insert ran. The
damage showed up in **child** processes: `multiprocessing` spawn inherits the
parent's `sys.path`, so every spawned worker in the whole session started from
the poisoned path, died during bootstrap with

    ModuleNotFoundError: No module named 'qector_decoder_v3.qector_decoder_v3'

and left its parent blocked forever on a queue or pool join. Two separate
full-suite hangs traced back to that single line:

  * `test_comprehensive_suite.py::test_decoderpool_basic` — `pool.join()`
  * `test_sinter.py::test_sinter_collect_runs_and_is_sane` — `sinter.collect()`

Both looked like flaky multiprocessing. Neither was.
"""

from __future__ import annotations

import importlib.util
import os
import sys


def _repo_python_dir() -> str:
    """Absolute path of the repo's `python/` directory (parent of `tests/`)."""
    return os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))


def _installed_package_has_native_extension() -> bool:
    """True when an importable `qector_decoder_v3` ships the compiled module.

    Deliberately checks a candidate location on disk rather than importing, so
    this runs before any test has imported the package and cannot be fooled by
    an already-cached `sys.modules` entry.
    """
    repo_dir = _repo_python_dir()
    for entry in sys.path:
        if not entry:
            continue
        if os.path.abspath(entry) == repo_dir:
            continue  # the source tree is what we are trying to avoid
        pkg = os.path.join(entry, "qector_decoder_v3")
        if not os.path.isdir(pkg):
            continue
        for name in os.listdir(pkg):
            if name.startswith("qector_decoder_v3.") and name.endswith((".pyd", ".so")):
                return True
    return False


def _demote_repo_source_dir() -> None:
    """Move the repo `python/` dir to the end of `sys.path`, if present.

    Demoted rather than removed: a source-only checkout with no installed wheel
    should still be able to import the pure-Python layer, it just must not win
    against a real installation.
    """
    repo_dir = _repo_python_dir()
    kept = [p for p in sys.path if not p or os.path.abspath(p) != repo_dir]
    if len(kept) != len(sys.path):
        sys.path[:] = kept + [repo_dir]


if _installed_package_has_native_extension():
    _demote_repo_source_dir()
    # Re-exported so tests (and spawned children, which inherit os.environ) can
    # assert on the decision if they need to.
    os.environ.setdefault("QECTOR_TESTS_USE_INSTALLED", "1")


def pytest_report_header(config):
    """Make the resolution decision visible in the test header."""
    spec = importlib.util.find_spec("qector_decoder_v3")
    origin = spec.origin if spec else "<not found>"
    native = "installed (native)" if os.environ.get("QECTOR_TESTS_USE_INSTALLED") else "repo source"
    return f"qector_decoder_v3 resolved from: {origin}  [{native}]"
