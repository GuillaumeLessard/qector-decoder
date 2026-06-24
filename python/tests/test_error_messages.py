"""Section 14: informative error messages.

Error messages from the validation layer must help the caller fix the input:
they mention "unknown", the offending kind, and (for the benchmark dispatcher)
the list of valid choices.
"""
import numpy as np
import pytest

from qector_decoder_v3 import codes, dem, result, benchmarking


def _small_dem():
    text = (
        "error(0.1) D0 L0\n"
        "error(0.1) D0 D1\n"
        "error(0.1) D1 L0\n"
    )
    return dem.parse_dem(text)


def test_decode_with_diagnostics_message_says_unknown():
    code = codes.repetition_code(3)
    H = code.parity_check_matrix()
    e = np.zeros(code.n_qubits, np.uint8)
    e[0] = 1
    s = ((H @ e) & 1).astype(np.uint8)
    with pytest.raises(ValueError, match="unknown") as excinfo:
        result.decode_with_diagnostics(code, s, kind="nonsense")
    assert "unknown" in str(excinfo.value).lower()


def test_make_decoder_message_mentions_kind():
    dm = _small_dem()
    with pytest.raises(ValueError) as excinfo:
        dm.make_decoder("zzz")
    msg = str(excinfo.value)
    assert "zzz" in msg, f"message should mention the bad kind: {msg!r}"


def test_benchmark_decoder_message_lists_choices():
    code = codes.repetition_code(3)
    with pytest.raises(ValueError) as excinfo:
        benchmarking.benchmark_decoder("zzz", code, n_trials=2, warmup=0)
    msg = str(excinfo.value)
    assert "zzz" in msg
    # message enumerates valid decoder kinds
    assert "blossom" in msg and "union_find" in msg, (
        f"message should list valid choices: {msg!r}"
    )
