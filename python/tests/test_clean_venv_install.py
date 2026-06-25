"""Clean importability and version parity.

A *true* clean-venv build (create venv, ``maturin build``, ``pip install`` the
wheel, import) is too slow for a unit test; that full round-trip is exercised by
CI — the ``Dockerfile`` builds the wheel from scratch and ``.github/workflows``
runs the suite across a Linux/Windows/macOS x Python 3.9-3.12 matrix.

What *this* test verifies cheaply and deterministically is the property that the
clean build ultimately proves: that ``import qector_decoder_v3`` works from a
fresh interpreter and reports the same ``__version__`` as the in-process import,
and that importing the package does not require the optional Stim ecosystem
packages (stim / pymatching) to be importable.
"""

import subprocess
import sys
import textwrap

import qector_decoder_v3


def test_fresh_subprocess_import_version_parity():
    in_proc_version = qector_decoder_v3.__version__
    code = textwrap.dedent(
        """
        import qector_decoder_v3
        print(qector_decoder_v3.__version__)
        """
    )
    proc = subprocess.run(
        [sys.executable, "-c", code],
        capture_output=True,
        text=True,
        timeout=120,
    )
    assert proc.returncode == 0, (
        f"import failed:\nSTDOUT:\n{proc.stdout}\nSTDERR:\n{proc.stderr}"
    )
    printed = proc.stdout.strip().splitlines()[-1].strip()
    assert printed == in_proc_version, (
        f"subprocess __version__ {printed!r} != in-process {in_proc_version!r}"
    )


def test_import_does_not_require_stim_ecosystem():
    # qector_decoder_v3 must import cleanly without the optional Stim ecosystem
    # being a *prerequisite*. We block 'stim' and 'pymatching' from being
    # importable in the child (by poisoning sys.modules with None, which raises
    # ImportError on import) BEFORE importing qd, then assert qd still imports
    # and reports the right version. The compiled core decoder must also build.
    code = textwrap.dedent(
        """
        import sys
        # Make the optional ecosystem packages unimportable for this child.
        sys.modules["stim"] = None
        sys.modules["pymatching"] = None

        import numpy as np
        import qector_decoder_v3 as qd
        from qector_decoder_v3 import codes

        # Core import succeeded despite stim/pymatching being unavailable.
        assert "qector_decoder_v3" in sys.modules
        c = codes.rotated_surface_code(3)
        H = c.parity_check_matrix()
        s = c.syndrome(c.random_error(0.1, np.random.default_rng(0)))
        corr = np.asarray(qd.BlossomDecoder(c.check_to_qubits, c.n_qubits).decode(s), np.uint8)
        assert np.array_equal((H @ corr) & 1, s), "core decode not faithful"
        print("VERSION", qd.__version__)
        print("CORE_OK")
        """
    )
    proc = subprocess.run(
        [sys.executable, "-c", code],
        capture_output=True,
        text=True,
        timeout=120,
    )
    assert proc.returncode == 0, (
        f"qd import requires stim/pymatching:\nSTDOUT:\n{proc.stdout}\nSTDERR:\n{proc.stderr}"
    )
    out = proc.stdout
    assert f"VERSION {qector_decoder_v3.__version__}" in out, out
    assert "CORE_OK" in out, out
