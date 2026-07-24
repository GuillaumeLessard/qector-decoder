"""Tests for full API surface coverage, closing testing gaps."""

import numpy as np
import pytest
import qector_decoder_v3 as qd


def test_neural_predecoder():
    # 1. NeuralPredecoder coverage
    pre = qd.NeuralPredecoder(n_input=4, n_output=2, n_hidden1=8, n_hidden2=8)
    assert pre.n_input == 4
    assert pre.n_output == 2
    assert pre.n_hidden1 == 8
    assert pre.n_hidden2 == 8

    # syndromes/corrections are flat 1D arrays
    syndromes = np.zeros(10 * 4, dtype=np.uint8)
    corrections = np.zeros(10 * 2, dtype=np.uint8)
    pre.train(syndromes, corrections, n_epochs=3, learning_rate=0.01)

    syn = np.zeros(4, dtype=np.uint8)
    pred = pre.predict(syn)
    corr = pre.decode(syn)
    assert pred.shape == (2,)
    assert corr.shape == (2,)


def test_gnn_and_trainer():
    # 2. GNNPredecoder, DetectorGraph, and GNNTrainer coverage
    checks, n_qubits = qd.generate_ring_code_checks(5)
    n_checks = len(checks)

    # Create GNN
    gnn = qd.GNNPredecoder.new_standard(hidden_size=16, n_layers=2)
    # Check learning rate & l2 lambda getters/setters
    gnn.learning_rate = 0.05
    assert abs(gnn.learning_rate - 0.05) < 1e-5
    gnn.l2_lambda = 0.01
    assert abs(gnn.l2_lambda - 0.01) < 1e-5

    # Create DetectorGraph
    syndrome = np.zeros(n_checks, dtype=np.uint8)
    graph = qd.DetectorGraph(checks, syndrome, n_qubits=n_qubits)
    assert graph.n_nodes == n_checks
    assert graph.n_edges > 0
    assert len(graph.node_features) == n_checks
    assert len(graph.edge_features) == graph.n_edges
    assert len(graph.edge_qubit_id) == graph.n_edges

    # update syndrome
    new_syndrome = np.ones(n_checks, dtype=np.uint8)
    graph.update_syndrome(new_syndrome)

    # forward
    weights = gnn.forward(graph)
    assert len(weights) == graph.n_edges

    # predict_with_node_probs
    edge_weights, node_probs = gnn.predict_with_node_probs(graph)
    assert len(edge_weights) == graph.n_edges
    assert len(node_probs) == graph.n_nodes

    # GNNTrainer
    trainer = qd.GNNTrainer(checks, n_qubits, error_rate=0.1)
    # train
    loss = trainer.train(gnn, n_samples=5, n_epochs=2)
    assert isinstance(loss, float)

    # train_bp
    loss_bp = trainer.train_bp(gnn, n_samples=5, n_epochs=2, max_bp_iter=10)
    assert isinstance(loss_bp, float)

    # generate_dataset
    dataset_size = trainer.generate_dataset(n_samples=5)
    assert dataset_size == 5


def test_bposd_bp_decode():
    checks, n_qubits = qd.generate_ring_code_checks(5)
    dec = qd.BPOSDDecoder(checks, n_qubits)
    syndrome = np.zeros(len(checks), dtype=np.uint8)
    llr = dec.bp_decode(syndrome, max_iterations=10)
    assert llr.shape == (n_qubits,)


def test_sparse_blossom_decode_with_weights():
    checks, n_qubits = qd.generate_ring_code_checks(5)
    dec = qd.SparseBlossomDecoder(checks, n_qubits)
    syndrome = np.zeros(len(checks), dtype=np.uint8)
    # list of (qubit_id, weight)
    weights = [(0, 1.5), (1, 2.0)]
    corr = dec.decode_with_weights(syndrome, weights)
    assert corr.shape == (n_qubits,)


def test_hybrid_decoder_train_bp():
    checks, n_qubits = qd.generate_ring_code_checks(5)
    dec = qd.HybridDecoder(checks, n_qubits)
    loss = dec.train_bp(n_samples=5, n_epochs=2, error_rate=0.1, max_bp_iter=10)
    assert isinstance(loss, float)

    loss_teacher = dec.train(n_samples=5, n_epochs=2, error_rate=0.1)
    assert isinstance(loss_teacher, float)


def test_toy_code_generator():
    checks, n_qubits = qd.generate_toy_code_checks(5)
    assert n_qubits == 25
    assert len(checks) == 50  # distance 5 toy code check size


def test_opencl_available_top_level():
    avail = qd.opencl_is_available()
    assert isinstance(avail, bool)


def test_version_match():
    # Verify __version__ matches Rust package version
    try:
        from qector_decoder_v3.qector_decoder_v3 import __version__ as rust_version

        assert qd.__version__ == rust_version
    except ImportError:
        pass
