"""
Explicit API protocol test suite (.decode_correction & .decode_observables).

Verifies that all decoder classes implement the explicit methods required by QECTOR v1.0.
"""

import numpy as np
import pytest

import qector_decoder_v3 as qd
from qector_decoder_v3 import (
    BeliefMatching,
    BlossomDecoder,
    ColourCodeDecoder,
    FastUnionFindDecoder,
    TwoStageDecoder,
)
from qector_decoder_v3.bposd import GraphBPOSDDecoder, MatrixBPOSDDecoder



def _H(c2q, nq):
    H = np.zeros((len(c2q), nq), dtype=np.uint8)
    for ci, qs in enumerate(c2q):
        for q in qs:
            H[ci, q] ^= 1
    return H


def test_blossom_explicit_api():
    c2q = [[0, 1], [1, 2]]
    dec = BlossomDecoder(c2q, n_qubits=3)
    syn = np.array([1, 0], dtype=np.uint8)

    corr = dec.decode_correction(syn)
    assert isinstance(corr, np.ndarray)
    assert corr.shape == (3,)
    H = _H(c2q, 3)
    assert np.array_equal((H @ corr) & 1, syn)

    obs_matrix = np.array([[1, 0, 1]], dtype=np.uint8)
    obs = dec.decode_observables(syn, observables_matrix=obs_matrix)
    assert isinstance(obs, np.ndarray)
    assert np.array_equal(obs, (obs_matrix @ corr) & 1)


def test_fast_uf_explicit_api():
    c2q = [[0, 1], [1, 2]]
    dec = FastUnionFindDecoder(c2q, n_qubits=3)
    syn = np.array([1, 0], dtype=np.uint8)

    corr = dec.decode_correction(syn)
    assert isinstance(corr, np.ndarray)
    assert corr.shape == (3,)
    H = _H(c2q, 3)
    assert np.array_equal((H @ corr) & 1, syn)

    obs_matrix = np.array([[1, 0, 1]], dtype=np.uint8)
    obs = dec.decode_observables(syn, observables_matrix=obs_matrix)
    assert isinstance(obs, np.ndarray)
    assert np.array_equal(obs, (obs_matrix @ corr) & 1)


def test_matrix_bposd_explicit_api():
    H = np.array(
        [
            [1, 1, 0],
            [0, 1, 1],
        ],
        dtype=np.uint8,
    )
    dec = MatrixBPOSDDecoder(H, error_rate=0.05, osd_order=0)
    syn = np.array([1, 0], dtype=np.uint8)

    corr = dec.decode_correction(syn)
    assert isinstance(corr, np.ndarray)
    assert corr.shape == (3,)
    assert np.array_equal((H @ corr) & 1, syn)

    obs_matrix = np.array([[1, 0, 1]], dtype=np.uint8)
    obs = dec.decode_observables(syn, observables_matrix=obs_matrix)
    assert isinstance(obs, np.ndarray)
    assert np.array_equal(obs, (obs_matrix @ corr) & 1)


def test_belief_matching_explicit_api():
    c2q = [[0, 1], [1, 2], [0, 2]]
    dec = BeliefMatching.from_numpy_h(_H(c2q, 3), error_rate=0.05)
    syn = np.array([1, 1, 0], dtype=np.uint8)

    corr = dec.decode_correction(syn)
    assert isinstance(corr, np.ndarray)

    obs = dec.decode_observables(syn)
    assert isinstance(obs, np.ndarray)


def test_two_stage_explicit_api():
    # TwoStageDecoder needs a CSS code with both X and Z sector checks.
    n = 3
    c2q = []
    types = []
    for i in range(n):
        c2q.append([i, (i + 1) % n])
        types.append(True)
    for i in range(n):
        c2q.append([i, (i + 2) % n])
        types.append(False)
    dec = TwoStageDecoder(c2q, types, n_qubits=n)
    syn = np.zeros(2 * n, dtype=np.uint8)

    corr = dec.decode_correction(syn)
    assert isinstance(corr, np.ndarray)
    assert corr.shape == (n,)
    H = _H(c2q, n)
    assert np.array_equal((H @ corr) & 1, syn)

    obs = dec.decode_observables(syn)
    assert isinstance(obs, np.ndarray)


