"""Tests for `TwoStageDecoder` (C1-03 correlated two-stage matching for CSS codes).

Validates the PyO3 boundary for two-stage decoding with various stage decoder options
(Blossom, Union-Find, Sparse Blossom, BP-OSD with BpMethod::Exact), syndrome decoding,
and error handling.
"""

import numpy as np
import pytest
from qector_decoder_v3 import TwoStageDecoder


def make_css_ring(n: int = 8):
    """Build a CSS ring code with n X checks and n Z checks over n qubits."""
    c2q = []
    types = []
    # n X-checks: check i covers [i, (i+1)%n]
    for i in range(n):
        c2q.append([i, (i + 1) % n])
        types.append(True)
    # n Z-checks: check i covers [i, (i+2)%n]
    for i in range(n):
        c2q.append([i, (i + 2) % n])
        types.append(False)
    return c2q, types, n


def to_uint8_array(corr):
    """Convert returned bytes or sequence into a uint8 NumPy array."""
    if isinstance(corr, (bytes, bytearray)):
        return np.frombuffer(corr, dtype=np.uint8)
    return np.array(corr, dtype=np.uint8)


def test_two_stage_decoder_init_and_properties():
    c2q, types, n = make_css_ring(8)
    dec = TwoStageDecoder(c2q, types, n_qubits=n)
    assert dec.n_qubits == n
    assert dec.n_checks == 2 * n


@pytest.mark.parametrize(
    "x_dec, z_dec",
    [
        ("blossom", "blossom"),
        ("unionfind", "unionfind"),
        ("sparse_blossom", "unionfind"),
        ("bposd", "bposd"),
        ("bposd:0.01:2", "blossom"),
    ],
)
def test_two_stage_decoder_stage_combinations(x_dec, z_dec):
    c2q, types, n = make_css_ring(8)
    dec = TwoStageDecoder(c2q, types, n_qubits=n, x_decoder=x_dec, z_decoder=z_dec)

    # Zero syndrome yields zero correction
    zero_syn = np.zeros(2 * n, dtype=np.uint8)
    corr = to_uint8_array(dec.decode(zero_syn))
    assert len(corr) == n
    assert np.all(corr == 0)


def test_two_stage_decoder_single_error_decoding():
    c2q, types, n = make_css_ring(8)
    dec = TwoStageDecoder(c2q, types, n_qubits=n, x_decoder="blossom", z_decoder="blossom")

    # Physical Z error on qubit 1: triggers X check 0 ([0,1]), X check 1 ([1,2]), and Z check 1 ([1,3])
    syndrome = np.zeros(2 * n, dtype=np.uint8)
    syndrome[0] = 1
    syndrome[1] = 1
    syndrome[n + 1] = 1

    corr = to_uint8_array(dec.decode(syndrome))
    assert len(corr) == n
    assert corr[1] == 1
    # Verify correction is faithful for X checks
    for ci, q_list in enumerate(c2q[:n]):
        parity = sum(int(corr[q]) for q in q_list) % 2
        assert parity == syndrome[ci]


def test_two_stage_decoder_invalid_check_types():
    c2q, types, n = make_css_ring(8)

    # Mismatched length check_types
    with pytest.raises(ValueError, match="check_types length"):
        TwoStageDecoder(c2q, types[:-1], n_qubits=n)

    # Missing Z checks
    with pytest.raises(ValueError, match="needs both X and Z checks"):
        TwoStageDecoder(c2q[:n], [True] * n, n_qubits=n)


def test_two_stage_decoder_invalid_stage_name():
    c2q, types, n = make_css_ring(8)
    with pytest.raises(ValueError, match="unknown stage decoder"):
        TwoStageDecoder(c2q, types, n_qubits=n, x_decoder="invalid_solver")


def test_two_stage_decoder_syndrome_validation():
    c2q, types, n = make_css_ring(8)
    dec = TwoStageDecoder(c2q, types, n_qubits=n)

    with pytest.raises(TypeError, match="uint8"):
        dec.decode(np.zeros(2 * n, dtype=np.int32))

    # Short syndrome returns all-zero correction without panic
    short_syn = np.zeros(3, dtype=np.uint8)
    corr = to_uint8_array(dec.decode(short_syn))
    assert len(corr) == n
    assert np.all(corr == 0)
