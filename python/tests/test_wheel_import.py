"""The compiled extension is present, importable, and functional.

A wheel is only useful if the native extension module it bundles actually loads.
This test asserts that:

* ``import qector_decoder_v3.qector_decoder_v3`` (the compiled core) succeeds and
  exposes the expected Rust-defined symbols,
* the compiled artifact file (``.pyd`` on Windows / ``.so`` elsewhere) for the
  *current* interpreter sits next to the package ``__init__.py``, and
* a decoder built directly from the core module decodes faithfully.
"""

import glob
import os
import sysconfig

import numpy as np

import qector_decoder_v3
import qector_decoder_v3.qector_decoder_v3 as core
from qector_decoder_v3 import codes


def test_core_module_exposes_expected_symbols():
    expected = ["BlossomDecoder", "SparseBlossomDecoder", "cuda_is_available"]
    for name in expected:
        assert hasattr(core, name), f"core module missing {name!r}"
    # cuda_is_available is a callable returning a bool.
    assert isinstance(core.cuda_is_available(), bool)


def test_compiled_artifact_exists_for_current_interpreter():
    pkg_dir = os.path.dirname(os.path.abspath(qector_decoder_v3.__file__))
    assert os.path.isfile(os.path.join(pkg_dir, "__init__.py"))

    # The extension that THIS interpreter actually loaded must be a real file
    # on disk next to the package.
    loaded = getattr(core, "__file__", None)
    assert loaded is not None and os.path.isfile(loaded), (
        f"core.__file__ is not a real file: {loaded!r}"
    )
    assert os.path.dirname(os.path.abspath(loaded)) == pkg_dir

    # And it must match this interpreter's extension suffix (e.g. cp311 .pyd).
    ext_suffix = sysconfig.get_config_var("EXT_SUFFIX")  # e.g. .cp311-win_amd64.pyd
    expected = os.path.join(pkg_dir, "qector_decoder_v3" + ext_suffix)
    assert os.path.isfile(expected), (
        f"no compiled extension for this interpreter; expected {expected}"
    )

    # Sanity: at least one .pyd or .so for the package exists.
    artifacts = glob.glob(os.path.join(pkg_dir, "qector_decoder_v3.*.pyd"))
    artifacts += glob.glob(os.path.join(pkg_dir, "qector_decoder_v3.*.so"))
    assert artifacts, f"no compiled extension artifacts found in {pkg_dir}"


def test_decoder_from_core_module_decodes_faithfully():
    code = codes.rotated_surface_code(5)
    H = code.parity_check_matrix()
    rng = np.random.default_rng(11)
    e = code.random_error(0.1, rng)
    s = code.syndrome(e)

    # Build directly from the compiled core (not the Python wrapper). The raw
    # core BlossomDecoder takes edge_weights positionally (the Python wrapper
    # defaults it); pass None for uniform weights.
    dec = core.BlossomDecoder(code.check_to_qubits, code.n_qubits, None)
    corr = np.asarray(dec.decode(s), np.uint8)
    assert corr.shape == (code.n_qubits,)
    assert np.array_equal((H @ corr) & 1, s), "core BlossomDecoder not syndrome-faithful"

    sparse = core.SparseBlossomDecoder(code.check_to_qubits, code.n_qubits)
    corr2 = np.asarray(sparse.decode(s), np.uint8)
    assert np.array_equal((H @ corr2) & 1, s), "core SparseBlossomDecoder not faithful"
