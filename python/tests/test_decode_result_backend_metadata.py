"""Section 15: DecodeResult backend / metadata reporting.

When built via ``kind="blossom"`` the backend label is the kind string; when a
prebuilt decoder is supplied via ``decoder=`` the backend label is the decoder's
class name. ``metadata`` is always a dict.
"""
import numpy as np

import qector_decoder_v3 as qd
from qector_decoder_v3 import codes, result


def _case():
    code = codes.repetition_code(5)
    H = code.parity_check_matrix()
    e = np.zeros(code.n_qubits, np.uint8)
    e[0] = 1
    s = ((H @ e) & 1).astype(np.uint8)
    return code, H, s


def test_backend_label_is_kind_string():
    code, H, s = _case()
    res = result.decode_with_diagnostics(code, s, kind="blossom")
    assert res.backend == "blossom"
    assert isinstance(res.metadata, dict)


def test_backend_label_is_decoder_class_name():
    code, H, s = _case()
    dec = qd.BlossomDecoder(code.check_to_qubits, code.n_qubits)
    res = result.decode_with_diagnostics(code, s, decoder=dec)
    assert res.backend == type(dec).__name__ == "BlossomDecoder"
    assert isinstance(res.metadata, dict)


def test_prebuilt_decoder_result_is_faithful():
    code, H, s = _case()
    dec = qd.UnionFindDecoder(code.check_to_qubits, code.n_qubits)
    res = result.decode_with_diagnostics(code, s, decoder=dec)
    assert res.backend == "UnionFindDecoder"
    corr = np.asarray(res.as_uint8(), np.uint8)
    assert np.array_equal((H @ corr) & 1, s)
