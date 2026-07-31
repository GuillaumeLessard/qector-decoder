"""Tests for ONNX export, numpy forward, and runtime backend resolution (T5-11)."""

import importlib.util
import os
import sys
import tempfile

import numpy as np
import pytest

# Load onnx_predecoder module directly without requiring compiled PyO3 native extension
spec = importlib.util.spec_from_file_location(
    "onnx_predecoder",
    os.path.abspath("python/qector_decoder_v3/onnx_predecoder.py"),
)
onnx_predecoder = importlib.util.module_from_spec(spec)
sys.modules["onnx_predecoder"] = onnx_predecoder
spec.loader.exec_module(onnx_predecoder)

resolve_gnn_backend = onnx_predecoder.resolve_gnn_backend
default_ort_providers = onnx_predecoder.default_ort_providers
numpy_gnn_forward = onnx_predecoder.numpy_gnn_forward
load_safetensors_numpy = onnx_predecoder.load_safetensors_numpy
count_mpnn_layers = onnx_predecoder.count_mpnn_layers


def test_resolve_gnn_backend():
    assert resolve_gnn_backend("rust") == "rust"
    assert resolve_gnn_backend("auto") in ("rust", "torch", "onnx")
    with pytest.raises(ValueError, match="Unknown GNN backend"):
        resolve_gnn_backend("invalid_backend_name")


def test_default_ort_providers():
    providers = default_ort_providers()
    assert isinstance(providers, list)
    assert len(providers) > 0
    assert "CPUExecutionProvider" in providers


def test_numpy_gnn_forward_mock_state():
    # Build a minimal weight state dict matching GNNPredecoder (hidden_size=4, 1 layer)
    # node_feat_dim = 2, edge_feat_dim = 2
    state = {
        "layers.0.w_message": np.zeros((4, 4), dtype=np.float32),  # in_dim = 2 + 2 = 4
        "layers.0.b_message": np.zeros(4, dtype=np.float32),
        "layers.0.w_update": np.zeros((4, 6), dtype=np.float32),   # in_dim = 2 + 4 = 6
        "layers.0.b_update": np.zeros(4, dtype=np.float32),
        "edge_readout.w1": np.zeros((4, 10), dtype=np.float32),    # 4 + 4 + 2 = 10
        "edge_readout.b1": np.zeros(4, dtype=np.float32),
        "edge_readout.w2": np.zeros((1, 4), dtype=np.float32),
        "edge_readout.b2": np.zeros(1, dtype=np.float32),
    }
    assert count_mpnn_layers(state) == 1

    node_features = np.ones((4, 2), dtype=np.float32)
    edge_features = np.ones((4, 2), dtype=np.float32)
    edge_src = [0, 1, 2, 3]
    edge_dst = [1, 2, 3, 0]

    out = numpy_gnn_forward(state, node_features, edge_features, edge_src, edge_dst)
    assert isinstance(out, np.ndarray)
    assert out.shape == (4,)
    # Zero weights + softplus(0.0) -> ln(2) approx 0.693147
    assert np.allclose(out, np.log(2.0), atol=1e-3)
