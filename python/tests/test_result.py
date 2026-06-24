"""Tests for qector_decoder_v3.result — DecodeResult and diagnostics."""

import json

import numpy as np

from qector_decoder_v3 import codes
from qector_decoder_v3.result import DecodeResult, decode_with_diagnostics


def test_decode_with_diagnostics_basic():
    code = codes.repetition_code(11)
    rng = np.random.default_rng(0)
    e = code.random_error(0.1, rng)
    s = code.syndrome(e)
    res = decode_with_diagnostics(code, s, kind="blossom")
    assert isinstance(res, DecodeResult)
    assert res.syndrome_valid is True
    assert res.correction.shape == (11,)
    assert res.decode_seconds is not None and res.decode_seconds >= 0


def test_alternative_encodings_consistent():
    code = codes.rotated_surface_code(5)
    rng = np.random.default_rng(3)
    e = code.random_error(0.1, rng)
    s = code.syndrome(e)
    res = decode_with_diagnostics(code, s, kind="sparse_blossom")
    # sparse indices == nonzero of dense
    assert res.sparse_indices.tolist() == np.nonzero(res.correction)[0].tolist()
    # bit-packed unpacks back to the correction
    unpacked = np.unpackbits(res.bit_packed)[: code.n_qubits]
    assert np.array_equal(unpacked, res.correction)
    assert res.hamming_weight == int(res.correction.sum())


def test_logical_flips_present_when_logicals_known():
    code = codes.repetition_code(9)
    s = code.syndrome(code.random_error(0.15, np.random.default_rng(1)))
    res = decode_with_diagnostics(code, s, kind="blossom")
    assert res.logical_flips is not None
    assert res.logical_flips.shape == (1,)


def test_to_json_is_valid_and_roundtrips_fields():
    code = codes.repetition_code(7)
    s = code.syndrome(code.random_error(0.1, np.random.default_rng(2)))
    res = decode_with_diagnostics(code, s, kind="blossom")
    blob = res.to_json()
    parsed = json.loads(blob)
    assert parsed["n_qubits"] == 7
    assert parsed["syndrome_valid"] is True
    assert isinstance(parsed["sparse_indices"], list)


def test_explain_is_readable():
    code = codes.rotated_surface_code(5)
    s = code.syndrome(code.random_error(0.1, np.random.default_rng(4)))
    res = decode_with_diagnostics(code, s, kind="blossom")
    text = res.explain()
    assert "QECTOR decode diagnostics" in text
    assert "correction weight" in text


def test_reused_decoder_path():
    import qector_decoder_v3 as qd

    code = codes.repetition_code(15)
    dec = qd.BlossomDecoder(code.check_to_qubits, code.n_qubits)
    s = code.syndrome(code.random_error(0.1, np.random.default_rng(9)))
    res = decode_with_diagnostics(code, s, decoder=dec)
    assert res.syndrome_valid is True
    assert res.backend == "BlossomDecoder"


def test_verify_detects_bad_correction():
    code = codes.repetition_code(7)
    s = np.array([1, 0, 0, 0, 0, 0], dtype=np.uint8)
    bad = DecodeResult(
        correction=np.zeros(7, np.uint8),
        syndrome=s,
        n_qubits=7,
        n_checks=6,
    )
    assert bad.verify(code.parity_check_matrix()) is False
    assert bad.syndrome_valid is False
