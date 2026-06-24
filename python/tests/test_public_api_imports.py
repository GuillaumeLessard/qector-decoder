"""Section 14: public API stability.

Assert every expected public name on ``qector_decoder_v3`` exists, is the right
kind (class / function / submodule), is present in ``__all__`` where applicable,
and that ``__version__`` is a non-empty string.
"""
import types
import inspect

import qector_decoder_v3 as qd


EXPECTED_CLASSES = [
    "BlossomDecoder",
    "SparseBlossomDecoder",
    "UnionFindDecoder",
    "FastUnionFindDecoder",
    "CPUBatchDecoder",
    "BatchDecoder",
    "CUDABatchDecoder",
    "OpenCLBatchDecoder",
    "BPOSDDecoder",
    "StreamingDecoder",
    "SlidingWindowDecoder",
    "AutoDecoder",
    "BackendConfig",
    "Backend",
    "DecodeResult",
    "BeliefMatching",
    "BpOsdDecoder",
    "PredecodedDecoder",
    "Workbench",
]

EXPECTED_FUNCTIONS = [
    "cuda_is_available",
    "opencl_is_available",
    "decode_with_diagnostics",
]

EXPECTED_SUBMODULES = [
    "codes",
    "dem",
    "result",
    "backend",
    "pymatching_compat",
    "benchmarking",
    "belief_matching",
    "bposd",
    "predecoder",
    "workbench",
]


def test_version_is_nonempty_string():
    v = qd.__version__
    assert isinstance(v, str)
    assert v.strip() != ""


def test_has_dunder_all():
    assert hasattr(qd, "__all__")
    assert isinstance(qd.__all__, (list, tuple))
    assert len(qd.__all__) > 0


def test_expected_classes_present_and_are_classes():
    all_names = set(qd.__all__)
    for name in EXPECTED_CLASSES:
        assert hasattr(qd, name), f"missing class {name}"
        obj = getattr(qd, name)
        assert inspect.isclass(obj), f"{name} is not a class: {type(obj)}"
        assert name in all_names, f"{name} not in __all__"


def test_expected_functions_present_and_callable():
    all_names = set(qd.__all__)
    for name in EXPECTED_FUNCTIONS:
        assert hasattr(qd, name), f"missing function {name}"
        obj = getattr(qd, name)
        assert callable(obj), f"{name} is not callable"
        assert name in all_names, f"{name} not in __all__"


def test_expected_submodules_present_and_are_modules():
    all_names = set(qd.__all__)
    for name in EXPECTED_SUBMODULES:
        assert hasattr(qd, name), f"missing submodule {name}"
        obj = getattr(qd, name)
        assert isinstance(obj, types.ModuleType), f"{name} is not a module: {type(obj)}"
        assert name in all_names, f"{name} not in __all__"


def test_decoder_classes_are_instantiable_on_a_real_code():
    # A representative class actually constructs and decodes, proving the
    # exported symbol is a usable decoder type, not a stub.
    import numpy as np
    from qector_decoder_v3 import codes

    code = codes.repetition_code(3)
    H = code.parity_check_matrix()
    e = np.zeros(code.n_qubits, np.uint8)
    e[0] = 1
    s = ((H @ e) & 1).astype(np.uint8)
    dec = qd.BlossomDecoder(code.check_to_qubits, code.n_qubits)
    corr = np.asarray(dec.decode(s), np.uint8)
    assert np.array_equal((H @ corr) & 1, s)
