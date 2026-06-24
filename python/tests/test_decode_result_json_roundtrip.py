"""Section 15: DecodeResult JSON / dict serialization.

``to_json()`` must produce text that parses with ``json.loads``; ``to_dict()``
must contain the documented diagnostic keys and be fully JSON-serializable
(re-dumpable with ``json.dumps``).
"""
import json

import numpy as np

from qector_decoder_v3 import codes, result


EXPECTED_KEYS = {
    "n_qubits",
    "n_checks",
    "hamming_weight",
    "sparse_indices",
    "backend",
    "syndrome_valid",
}


def _result():
    code = codes.repetition_code(5)
    H = code.parity_check_matrix()
    e = np.zeros(code.n_qubits, np.uint8)
    e[0] = 1
    e[3] = 1
    s = ((H @ e) & 1).astype(np.uint8)
    return result.decode_with_diagnostics(code, s, kind="blossom")


def test_to_json_parses():
    res = _result()
    text = res.to_json()
    assert isinstance(text, str)
    parsed = json.loads(text)
    assert isinstance(parsed, dict)


def test_to_dict_has_expected_keys():
    res = _result()
    d = res.to_dict()
    missing = EXPECTED_KEYS - set(d.keys())
    assert not missing, f"missing keys: {missing}"


def test_to_dict_is_json_serializable():
    res = _result()
    d = res.to_dict()
    # must not raise -- every value JSON-serializable
    dumped = json.dumps(d)
    roundtrip = json.loads(dumped)
    assert roundtrip["n_qubits"] == d["n_qubits"]
    assert roundtrip["n_checks"] == d["n_checks"]


def test_to_dict_values_consistent_with_result():
    res = _result()
    d = res.to_dict()
    assert d["n_qubits"] == res.n_qubits
    assert d["n_checks"] == res.n_checks
    assert d["hamming_weight"] == res.hamming_weight
    assert d["backend"] == res.backend
    assert d["syndrome_valid"] == res.syndrome_valid
