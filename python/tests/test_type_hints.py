"""Section 14: type-hint surface.

Assert the ``py.typed`` marker ships, a representative set of public Python
functions carry annotations on their (annotatable) params and return, and that
every pure-Python submodule imports cleanly.
"""
import importlib
import inspect
import os

import qector_decoder_v3 as qd


def test_py_typed_marker_exists():
    pkg_dir = os.path.dirname(qd.__file__)
    marker = os.path.join(pkg_dir, "py.typed")
    assert os.path.isfile(marker), f"missing py.typed at {marker}"


def _assert_return_annotated(func):
    sig = inspect.signature(func)
    assert sig.return_annotation is not inspect.Signature.empty, (
        f"{func.__qualname__} has no return annotation"
    )


def test_repetition_code_annotations():
    from qector_decoder_v3 import codes

    sig = inspect.signature(codes.repetition_code)
    _assert_return_annotated(codes.repetition_code)
    for name, param in sig.parameters.items():
        assert param.annotation is not inspect.Parameter.empty, (
            f"repetition_code param {name} not annotated"
        )


def test_parse_dem_annotations():
    from qector_decoder_v3 import dem

    sig = inspect.signature(dem.parse_dem)
    _assert_return_annotated(dem.parse_dem)
    for name, param in sig.parameters.items():
        assert param.annotation is not inspect.Parameter.empty, (
            f"parse_dem param {name} not annotated"
        )


def test_decode_with_diagnostics_annotations():
    from qector_decoder_v3 import result

    sig = inspect.signature(result.decode_with_diagnostics)
    _assert_return_annotated(result.decode_with_diagnostics)
    # core params must be annotated
    for name in ("code", "syndrome", "kind"):
        assert sig.parameters[name].annotation is not inspect.Parameter.empty, (
            f"decode_with_diagnostics param {name} not annotated"
        )


def test_benchmark_decoder_annotations():
    from qector_decoder_v3 import benchmarking

    sig = inspect.signature(benchmarking.benchmark_decoder)
    _assert_return_annotated(benchmarking.benchmark_decoder)
    # At least the leading kind and the keyword params carry annotations.
    for name in ("kind", "n_trials", "warmup", "p", "seed"):
        assert sig.parameters[name].annotation is not inspect.Parameter.empty, (
            f"benchmark_decoder param {name} not annotated"
        )


def test_pure_python_submodules_import():
    subs = [
        "codes",
        "dem",
        "result",
        "backend",
        "benchmarking",
        "bposd",
        "belief_matching",
        "workbench",
        "pymatching_compat",
    ]
    for s in subs:
        mod = importlib.import_module(f"qector_decoder_v3.{s}")
        assert mod is not None
