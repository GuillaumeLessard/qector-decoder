"""Section 15: DecodeResult.bit_packed round-trip.

``np.unpackbits(res.bit_packed)[:n_qubits]`` must reproduce the dense
correction, and a manual dense->packed->dense round-trip must be lossless.
"""
import numpy as np
import pytest

from qector_decoder_v3 import codes, result


@pytest.mark.parametrize("flips", [[], [0], [0, 7], [1, 3, 5, 9]])
def test_unpackbits_recovers_correction(flips):
    code = codes.repetition_code(11)
    H = code.parity_check_matrix()
    nq = code.n_qubits
    e = np.zeros(nq, np.uint8)
    for q in flips:
        e[q] = 1
    s = ((H @ e) & 1).astype(np.uint8)

    res = result.decode_with_diagnostics(code, s, kind="blossom")
    corr = np.asarray(res.as_uint8(), np.uint8)

    packed = np.asarray(res.bit_packed, np.uint8)
    unpacked = np.unpackbits(packed)[:nq]
    assert np.array_equal(unpacked, corr)


def test_dense_to_packed_to_dense_roundtrip():
    code = codes.rotated_surface_code(3)
    H = code.parity_check_matrix()
    nq = code.n_qubits
    rng = np.random.default_rng(5)
    e = (rng.random(nq) < 0.15).astype(np.uint8)
    s = ((H @ e) & 1).astype(np.uint8)

    res = result.decode_with_diagnostics(code, s, kind="blossom")
    corr = np.asarray(res.as_uint8(), np.uint8)

    # re-pack the dense correction ourselves and confirm it matches bit_packed
    repacked = np.packbits(corr)
    assert np.array_equal(repacked, np.asarray(res.bit_packed, np.uint8))
    # and that unpacking our repack recovers the correction
    assert np.array_equal(np.unpackbits(repacked)[:nq], corr)
