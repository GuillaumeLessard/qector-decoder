import numpy as np
import qector_decoder_v3 as qd


def test_fast_uf_decoder_decode():
    """Test FastUnionFindDecoder basic decode."""
    c2q = [[0, 1], [1, 2], [2, 3], [3, 0]]
    dec = qd.FastUnionFindDecoder(c2q)
    syndrome = np.array([1, 0, 1, 0], dtype=np.uint8)
    correction = dec.decode(syndrome)
    assert correction.dtype == np.uint8
    assert len(correction) == dec.n_qubits


def test_fast_uf_decoder_batch():
    """Test FastUnionFindDecoder batch decode."""
    c2q = [[0, 1], [1, 2], [2, 3], [3, 0]]
    dec = qd.FastUnionFindDecoder(c2q)
    syndromes = np.array([[1, 0, 1, 0], [0, 1, 0, 1]], dtype=np.uint8)
    corrections = dec.batch_decode(syndromes)
    assert corrections.shape == (2, dec.n_qubits)
    assert corrections.dtype == np.uint8


def test_fast_uf_decoder_empty_syndrome():
    """Fast decoder should return zero correction for empty syndrome."""
    c2q = [[0, 1], [1, 2], [2, 3], [3, 0]]
    dec = qd.FastUnionFindDecoder(c2q)
    syndrome = np.zeros(dec.n_checks, dtype=np.uint8)
    correction = dec.decode(syndrome)
    assert np.all(correction == 0)
