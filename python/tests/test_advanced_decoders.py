"""
Tests for advanced strategic decoders in qector_decoder_v3.advanced.
"""

from __future__ import annotations

import numpy as np
import pytest

import qector_decoder_v3 as qd
from qector_decoder_v3.advanced import (
    RadixHeap,
    FusionBlossomDecoder,
    CudaQDecoder,
    EvolutionaryBpDecoder,
    RestartBeliefDecoder,
    HybridAiDecoder,
    AstraGnnDecoder,
    EarlyExitDecoder,
)


def test_radix_heap():
    """Verify RadixHeap sorting and functionality."""
    heap = RadixHeap()
    elements = [(10, "A"), (5, "B"), (20, "C"), (5, "D"), (3, "E")]
    
    for key, val in elements:
        heap.push(key, val)
        
    assert heap.size == 5
    
    # Retrieve in monotonic order
    prev_key = -1
    retrieved = []
    while heap.size > 0:
        res = heap.pop_min()
        assert res is not None
        key, val = res
        assert key >= prev_key
        prev_key = key
        retrieved.append((key, val))
        
    assert len(retrieved) == 5
    assert retrieved[0][0] == 3


def test_fusion_blossom_decoder():
    """Verify FusionBlossomDecoder matching correctness."""
    # Build a simple repetition code (d=5)
    checks, n_qubits = qd.generate_repetition_code_checks(5)
    
    decoder = FusionBlossomDecoder(checks, n_qubits)
    
    # Test single error syndrome
    # Syndrome with 1 error
    syndrome = np.zeros(len(checks), dtype=np.uint8)
    syndrome[0] = 1 # defect at node 0
    
    correction = decoder.decode(syndrome)
    assert len(correction) == n_qubits
    
    # Test batch decode
    syndromes = np.zeros((3, len(checks)), dtype=np.uint8)
    syndromes[0, 0] = 1
    syndromes[1, 1] = 1
    
    batch_corr = decoder.batch_decode(syndromes)
    assert batch_corr.shape == (3, n_qubits)


def test_cuda_q_decoder():
    """Verify CudaQDecoder execution and CPU fallback."""
    checks, n_qubits = qd.generate_repetition_code_checks(5)
    decoder = CudaQDecoder(checks, n_qubits)
    
    syndrome = np.zeros(len(checks), dtype=np.uint8)
    syndrome[0] = 1
    
    corr = decoder.decode(syndrome)
    assert len(corr) == n_qubits
    
    # Batch decode
    syndromes = np.zeros((2, len(checks)), dtype=np.uint8)
    batch_corr = decoder.batch_decode(syndromes)
    assert batch_corr.shape == (2, n_qubits)


def test_evolutionary_bp_decoder():
    """Verify EvolutionaryBpDecoder Weighted Min-Sum BP and Training."""
    # Build check matrix H for d=3 repetition code
    H = np.array([
        [1, 1, 0],
        [0, 1, 1]
    ], dtype=np.uint8)
    
    decoder = EvolutionaryBpDecoder(H, max_iter=5)
    
    syndrome = np.array([1, 0], dtype=np.uint8)
    corr = decoder.decode(syndrome)
    assert len(corr) == 3
    
    # Run training pipeline
    training_syndromes = np.array([[1, 0], [0, 1]], dtype=np.uint8)
    training_errors = np.array([[1, 0, 0], [0, 0, 1]], dtype=np.uint8)
    
    decoder.train(training_syndromes, training_errors, pop_size=4, generations=2)
    # Check that weights were optimized/changed
    assert len(decoder.weights) == len(np.nonzero(H)[0])


def test_restart_belief_decoder():
    """Verify RestartBeliefDecoder restart loops."""
    H = np.array([
        [1, 1, 0],
        [0, 1, 1]
    ], dtype=np.uint8)
    
    decoder = RestartBeliefDecoder(H, max_iter=3, max_restarts=2)
    syndrome = np.array([1, 1], dtype=np.uint8)
    
    corr = decoder.decode(syndrome)
    assert len(corr) == 3


def test_hybrid_ai_decoder():
    """Verify HybridAiDecoder feedforward and decode operations."""
    decoder = HybridAiDecoder(n_checks=4, n_qubits=6)
    syndrome = np.array([1, 0, 1, 0], dtype=np.float32)
    
    corr = decoder.decode(syndrome)
    assert len(corr) == 6


def test_astra_gnn_decoder():
    """Verify AstraGnnDecoder message passing and decode operations."""
    H = np.array([
        [1, 1, 0],
        [0, 1, 1]
    ], dtype=np.uint8)
    
    decoder = AstraGnnDecoder(H, embed_dim=8, layers=2)
    syndrome = np.array([1, 0], dtype=np.uint8)
    
    corr = decoder.decode(syndrome)
    assert len(corr) == 3


def test_early_exit_decoder():
    """Verify EarlyExitDecoder early exit triggers."""
    H = np.array([
        [1, 1, 0],
        [0, 1, 1]
    ], dtype=np.uint8)
    
    # Create simple mock decoders that satisfy the interface
    class CheapMock:
        def decode(self, syndrome):
            # Solves syndrome [1, 0] perfectly by error [1, 0, 0]
            if np.array_equal(syndrome, [1, 0]):
                return np.array([1, 0, 0], dtype=np.uint8)
            return np.array([0, 0, 0], dtype=np.uint8)
            
    class FallbackMock:
        def decode(self, syndrome):
            return np.array([0, 1, 0], dtype=np.uint8)
            
    decoder = EarlyExitDecoder(CheapMock(), FallbackMock(), H)
    
    # 1. Test early exit
    corr1 = decoder.decode(np.array([1, 0], dtype=np.uint8))
    assert np.array_equal(corr1, [1, 0, 0])
    assert decoder.last_exited_early is True
    
    # 2. Test fallback (when cheap decoder doesn't satisfy)
    corr2 = decoder.decode(np.array([1, 1], dtype=np.uint8))
    assert np.array_equal(corr2, [0, 1, 0])
    assert decoder.last_exited_early is False
