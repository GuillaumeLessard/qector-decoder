import numpy as np
import qector_decoder_v3 as qd


def test_blossom_decoder_empty_syndrome():
    """Blossom should return zero correction for empty syndrome."""
    c2q = [[0, 1], [1, 2], [2, 3], [3, 0]]
    dec = qd.BlossomDecoder(c2q)
    syndrome = np.zeros(dec.n_checks, dtype=np.uint8)
    correction = dec.decode(syndrome)
    assert np.all(correction == 0)


def test_blossom_decoder_single_pair():
    """Blossom on a 4-qubit ring with two defects."""
    c2q = [[0, 1], [1, 2], [2, 3], [3, 0]]
    dec = qd.BlossomDecoder(c2q)
    syndrome = np.array([1, 0, 1, 0], dtype=np.uint8)
    correction = dec.decode(syndrome)
    assert len(correction) == dec.n_qubits
    weight = int(correction.sum())
    assert weight % 2 == 0, f"correction weight must be even, got {weight}"


def test_blossom_decoder_with_weights():
    """Blossom with edge weights."""
    c2q = [[0, 1], [1, 2], [2, 3], [3, 0]]
    edge_weights = [1.0, 10.0, 1.0, 10.0]
    dec = qd.BlossomDecoder(c2q, edge_weights=edge_weights)
    syndrome = np.array([1, 0, 1, 0], dtype=np.uint8)
    correction = dec.decode(syndrome)
    assert len(correction) == dec.n_qubits


def test_blossom_decoder_surface_code():
    """Blossom on a surface code."""
    c2q, nq = qd.generate_surface_code_checks(3)
    dec = qd.BlossomDecoder(c2q, n_qubits=nq)
    syndrome = np.zeros(dec.n_checks, dtype=np.uint8)
    # Set two defects
    syndrome[0] = 1
    syndrome[1] = 1
    correction = dec.decode(syndrome)
    assert len(correction) == nq


def test_blossom_decoder_batch():
    """Blossom batch decode."""
    c2q = [[0, 1], [1, 2], [2, 3], [3, 0]]
    dec = qd.BlossomDecoder(c2q)
    syndromes = np.array([[1, 0, 1, 0], [0, 1, 0, 1]], dtype=np.uint8)
    corrections = dec.batch_decode(syndromes)
    assert corrections.shape == (2, dec.n_qubits)
