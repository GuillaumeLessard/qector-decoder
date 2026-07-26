"""
Test qector_decoder_v3.pymatching drop-in import shim compatibility.
"""

import numpy as np
import pytest


def test_qector_pymatching_import():
    from qector_decoder_v3.pymatching import Matching

    # Build a simple ring code check matrix (3 checks, 3 qubits)
    H = np.array(
        [
            [1, 1, 0],
            [0, 1, 1],
            [1, 0, 1],
        ],
        dtype=np.uint8,
    )

    m = Matching.from_check_matrix(H)
    assert m.num_detectors == 3
    assert m.num_edges == 3

    # Decode single error on qubit 0 -> checks 0 and 2 fire
    syndrome = np.array([1, 0, 1], dtype=np.uint8)
    corr = m.decode(syndrome)
    assert len(corr) == 3
    assert np.array_equal((H @ corr) % 2, syndrome)

    # Decode batch
    shots = np.array(
        [
            [1, 0, 1],
            [0, 1, 1],
        ],
        dtype=np.uint8,
    )
    batch_corr = m.decode_batch(shots)
    assert batch_corr.shape == (2, 3)
    assert np.array_equal((H @ batch_corr[0]) % 2, shots[0])
    assert np.array_equal((H @ batch_corr[1]) % 2, shots[1])


def test_qector_pymatching_module_alias():
    from qector_decoder_v3 import pymatching

    H = np.array(
        [
            [1, 1, 0],
            [0, 1, 1],
            [1, 0, 1],
        ],
        dtype=np.uint8,
    )

    m = pymatching.Matching.from_check_matrix(H)
    assert m.num_detectors == 3


def test_qector_top_level_matching_import():
    from qector_decoder_v3 import Matching

    H = np.array(
        [
            [1, 1, 0],
            [0, 1, 1],
            [1, 0, 1],
        ],
        dtype=np.uint8,
    )

    m = Matching.from_check_matrix(H)
    assert m.num_detectors == 3
