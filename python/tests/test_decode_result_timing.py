"""Section 15: DecodeResult timing diagnostics.

``decode_with_diagnostics`` records ``decode_seconds`` as a non-negative float,
and ``explain()`` renders a decode-time line. No absolute wall-clock thresholds
are asserted (those would flake under load); only the type, sign and presence
of the rendered line.
"""

import numpy as np

from qector_decoder_v3 import codes, result


def _result():
    code = codes.repetition_code(7)
    H = code.parity_check_matrix()
    e = np.zeros(code.n_qubits, np.uint8)
    e[0] = 1
    e[4] = 1
    s = ((H @ e) & 1).astype(np.uint8)
    return result.decode_with_diagnostics(code, s, kind="blossom")


def test_decode_seconds_is_nonnegative_float():
    res = _result()
    assert isinstance(res.decode_seconds, float)
    assert res.decode_seconds >= 0.0


def test_explain_contains_decode_time_line():
    res = _result()
    text = res.explain()
    assert isinstance(text, str)
    assert "decode time" in text.lower()
