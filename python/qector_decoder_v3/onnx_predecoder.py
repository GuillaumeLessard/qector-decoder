"""ONNX export and optional runtime for GNNPredecoder (T5-11).

Mirrors the hand-rolled Rust MPNN math in ``gnn_layers.rs`` /
``gnn_predecoder.rs`` so exported ONNX graphs can run on ONNX Runtime,
TensorRT, or DirectML without a hard dependency on those engines.

Optional packages (never required at import time):
  * ``torch`` — preferred export path (``torch.onnx.export``)
  * ``onnx`` — model load/check helpers
  * ``onnxruntime`` — inference backend (CPU / CUDA / TensorRT / DirectML)

Environment:
  ``QECTOR_GNN_BACKEND`` = ``auto`` | ``rust`` | ``torch`` | ``onnx``
  ``QECTOR_GNN_ONNX_PATH`` = path to a previously exported ``.onnx`` model
  ``QECTOR_GNN_ONNX_PROVIDERS`` = comma-separated ORT providers
    (default: TensorrtExecutionProvider,DmlExecutionProvider,
     CUDAExecutionProvider,CPUExecutionProvider — first available wins)
"""

from __future__ import annotations

import os
from collections.abc import Sequence
from typing import Any, Optional

import numpy as np

# ---------------------------------------------------------------------------
# Backend resolution
# ---------------------------------------------------------------------------

_VALID_BACKENDS = frozenset({"auto", "rust", "torch", "onnx"})


def resolve_gnn_backend(explicit: Optional[str] = None) -> str:
    """Resolve the active GNN inference backend.

    Priority: explicit argument → ``QECTOR_GNN_BACKEND`` env → ``auto``.
    ``auto`` prefers torch when importable, otherwise rust. ``onnx`` is only
    selected when requested (or when ``QECTOR_GNN_ONNX_PATH`` is set under
    ``auto`` and onnxruntime is available).
    """
    raw = (explicit or os.environ.get("QECTOR_GNN_BACKEND") or "auto").strip().lower()
    if raw not in _VALID_BACKENDS:
        raise ValueError(
            f"Unknown GNN backend {raw!r}; expected one of {sorted(_VALID_BACKENDS)}"
        )
    if raw != "auto":
        return raw

    onnx_path = os.environ.get("QECTOR_GNN_ONNX_PATH")
    if onnx_path and os.path.isfile(onnx_path):
        try:
            import onnxruntime  # noqa: F401

            return "onnx"
        except ImportError:
            pass

    try:
        import torch  # noqa: F401

        return "torch"
    except ImportError:
        return "rust"


def default_ort_providers() -> list[str]:
    """Preferred ORT provider chain for TensorRT / DirectML / CUDA / CPU."""
    raw = os.environ.get("QECTOR_GNN_ONNX_PROVIDERS")
    if raw:
        return [p.strip() for p in raw.split(",") if p.strip()]
    return [
        "TensorrtExecutionProvider",
        "DmlExecutionProvider",
        "CUDAExecutionProvider",
        "CPUExecutionProvider",
    ]


# ---------------------------------------------------------------------------
# SafeTensors helpers (shared key layout with Rust / Torch)
# ---------------------------------------------------------------------------

def load_safetensors_numpy(path: str) -> dict[str, np.ndarray]:
    """Load a GNN safetensors checkpoint as float32 numpy arrays."""
    try:
        from safetensors import safe_open
    except ImportError as exc:
        raise ImportError(
            "safetensors is required to load GNN weights for ONNX export"
        ) from exc

    out: dict[str, np.ndarray] = {}
    with safe_open(path, framework="np") as f:
        for key in f.keys():
            arr = np.asarray(f.get_tensor(key), dtype=np.float32)
            out[key] = arr
    return out


def count_mpnn_layers(state: dict[str, np.ndarray]) -> int:
    n = 0
    while f"layers.{n}.w_message" in state:
        n += 1
    if n == 0:
        raise ValueError("No MPNN layers found in weight state (missing layers.0.w_message)")
    return n


# ---------------------------------------------------------------------------
# Pure-NumPy forward (bit-faithful to Rust f32 MPNN + softplus clamp)
# ---------------------------------------------------------------------------

def _linear(x: np.ndarray, w: np.ndarray, b: np.ndarray) -> np.ndarray:
    """``y = x @ W.T + b`` with W shaped [out, in] (matches Rust row layout)."""
    return x @ w.T + b


def _softplus_clamp(raw: np.ndarray) -> np.ndarray:
    """Match ``positive_weight`` in gnn_predecoder.rs (f64 softplus + clamp)."""
    raw64 = raw.astype(np.float64)
    # Numerically stable softplus
    out = np.empty_like(raw64)
    hi = raw64 > 40.0
    lo = raw64 < -40.0
    mid = ~(hi | lo)
    out[hi] = raw64[hi]
    out[lo] = np.exp(raw64[lo])
    out[mid] = np.log1p(np.exp(raw64[mid]))
    return np.clip(out, 1e-6, 100.0)


def numpy_mpnn_layer(
    node_embeddings: np.ndarray,
    edge_features: np.ndarray,
    edge_src: np.ndarray,
    edge_dst: np.ndarray,
    w_message: np.ndarray,
    b_message: np.ndarray,
    w_update: np.ndarray,
    b_update: np.ndarray,
) -> np.ndarray:
    """One undirected MPNN layer — mirrors ``MessagePassingLayer`` in Rust.

    For every edge (u, v) both directed messages are computed:
      msg_u→v = W_msg · [emb_u ‖ efeat] + b_msg
      msg_v→u = W_msg · [emb_v ‖ efeat] + b_msg
    Each node aggregates the mean of messages *from its neighbours*, then
      new_emb = ReLU(W_upd · [emb ‖ agg] + b_upd)
    """
    n_nodes = node_embeddings.shape[0]
    hidden = w_message.shape[0]
    emb = node_embeddings.astype(np.float32, copy=False)
    efeat = edge_features.astype(np.float32, copy=False)
    src = np.asarray(edge_src, dtype=np.int64)
    dst = np.asarray(edge_dst, dtype=np.int64)

    msg_from_src = _linear(
        np.concatenate([emb[src], efeat], axis=-1), w_message, b_message
    )
    msg_from_dst = _linear(
        np.concatenate([emb[dst], efeat], axis=-1), w_message, b_message
    )

    agg = np.zeros((n_nodes, hidden), dtype=np.float32)
    # Node receives the message *sourced at the neighbour*.
    np.add.at(agg, dst, msg_from_src)
    np.add.at(agg, src, msg_from_dst)

    deg = np.zeros(n_nodes, dtype=np.float32)
    np.add.at(deg, src, 1.0)
    np.add.at(deg, dst, 1.0)
    deg = np.maximum(deg, 1.0)
    agg = agg / deg[:, None]

    upd_in = np.concatenate([emb, agg], axis=-1)
    pre = _linear(upd_in, w_update, b_update)
    return np.maximum(pre, 0.0)


def numpy_gnn_forward(
    state: dict[str, np.ndarray],
    node_features: np.ndarray,
    edge_features: np.ndarray,
    edge_src: Sequence[int],
    edge_dst: Sequence[int],
) -> np.ndarray:
    """Full GNNPredecoder forward in NumPy (Rust-faithful, no ML deps)."""
    n_layers = count_mpnn_layers(state)
    emb = np.asarray(node_features, dtype=np.float32)
    efeat = np.asarray(edge_features, dtype=np.float32)
    src = np.asarray(edge_src, dtype=np.int64)
    dst = np.asarray(edge_dst, dtype=np.int64)

    for i in range(n_layers):
        emb = numpy_mpnn_layer(
            emb,
            efeat,
            src,
            dst,
            state[f"layers.{i}.w_message"],
            state[f"layers.{i}.b_message"],
            state[f"layers.{i}.w_update"],
            state[f"layers.{i}.b_update"],
        )

    src_emb = emb[src]
    dst_emb = emb[dst]
    readout_in = np.concatenate([src_emb, dst_emb, efeat], axis=-1)
    h = np.maximum(_linear(readout_in, state["edge_readout.w1"], state["edge_readout.b1"]), 0.0)
    raw = _linear(h, state["edge_readout.w2"], state["edge_readout.b2"]).reshape(-1)
    return _softplus_clamp(raw)


# ---------------------------------------------------------------------------
# Torch model helpers + ONNX export
# ---------------------------------------------------------------------------

def build_torch_model_from_state(
    state: dict[str, np.ndarray],
    *,
    dtype: str = "float32",
):
    """Construct a ``TorchGNNPredecoder`` and load ``state`` weights."""
    import torch

    from .torch_predecoder import TorchGNNPredecoder

    n_layers = count_mpnn_layers(state)
    w0 = state["layers.0.w_message"]
    hidden_size = int(w0.shape[0])
    # layer 0: w_message is [H, node_feat + edge_feat]
    # edge_feat_dim from edge_readout input: 2*H + edge_feat
    w1 = state["edge_readout.w1"]
    edge_feat_dim = int(w1.shape[1] - 2 * hidden_size)
    node_feat_dim = int(w0.shape[1] - edge_feat_dim)

    model = TorchGNNPredecoder(node_feat_dim, edge_feat_dim, hidden_size, n_layers)
    torch_dtype = torch.float32 if dtype == "float32" else torch.float64
    model = model.to(dtype=torch_dtype)

    sd = model.state_dict()
    loaded = {}
    for k, v in state.items():
        if k not in sd:
            raise KeyError(f"Unexpected weight key {k!r} not in TorchGNNPredecoder")
        loaded[k] = torch.tensor(v, dtype=torch_dtype)
    model.load_state_dict(loaded, strict=True)
    model.eval()
    return model


def export_gnn_to_onnx(
    weights_or_model: Any,
    onnx_path: str,
    *,
    n_nodes: int = 8,
    n_edges: int = 12,
    opset: int = 17,
    dtype: str = "float32",
) -> str:
    """Export a GNNPredecoder (weights path, state dict, or Torch module) to ONNX.

    The graph uses dynamic axes on N (nodes) and E (edges) so the same file
    works for any detector-graph size at inference time. Returns ``onnx_path``.
    """
    import torch

    if isinstance(weights_or_model, str):
        state = load_safetensors_numpy(weights_or_model)
        model = build_torch_model_from_state(state, dtype=dtype)
    elif isinstance(weights_or_model, dict):
        model = build_torch_model_from_state(weights_or_model, dtype=dtype)
    else:
        model = weights_or_model
        model.eval()

    # Infer feature dims from first layer / readout.
    first = model.layers[0]
    node_feat_dim = first.node_feat_dim
    edge_feat_dim = first.edge_feat_dim
    torch_dtype = next(model.parameters()).dtype

    node_features = torch.zeros(n_nodes, node_feat_dim, dtype=torch_dtype)
    edge_features = torch.zeros(n_edges, edge_feat_dim, dtype=torch_dtype)
    # Valid dummy topology: a simple path so scatter indices stay in range.
    edge_src = torch.arange(n_edges, dtype=torch.int64) % n_nodes
    edge_dst = (edge_src + 1) % n_nodes

    input_names = ["node_features", "edge_features", "edge_src", "edge_dst"]
    output_names = ["edge_weights"]
    dynamic_axes = {
        "node_features": {0: "n_nodes"},
        "edge_features": {0: "n_edges"},
        "edge_src": {0: "n_edges"},
        "edge_dst": {0: "n_edges"},
        "edge_weights": {0: "n_edges"},
    }

    os.makedirs(os.path.dirname(os.path.abspath(onnx_path)) or ".", exist_ok=True)

    # torch.onnx.export works without the ``onnx`` package for writing the file;
    # optional ``onnx.checker`` validation runs only when onnx is installed.
    torch.onnx.export(
        model,
        (node_features, edge_features, edge_src, edge_dst),
        onnx_path,
        input_names=input_names,
        output_names=output_names,
        dynamic_axes=dynamic_axes,
        opset_version=opset,
        do_constant_folding=True,
    )

    try:
        import onnx

        onnx.checker.check_model(onnx.load(onnx_path))
    except ImportError:
        pass
    except Exception as exc:  # pragma: no cover - defensive
        raise RuntimeError(f"ONNX export produced an invalid model: {exc}") from exc

    return onnx_path


# ---------------------------------------------------------------------------
# ONNX Runtime backend
# ---------------------------------------------------------------------------

class OnnxGNNRuntime:
    """Optional ONNX Runtime session for GNN pre-decoding.

    Providers are resolved from ``QECTOR_GNN_ONNX_PROVIDERS`` (TensorRT →
    DirectML → CUDA → CPU by default). Construction fails closed with a clear
    error if ``onnxruntime`` is not installed — callers should treat this as an
    opt-in backend, never a hard dependency.
    """

    def __init__(self, onnx_path: str, providers: Optional[Sequence[str]] = None):
        try:
            import onnxruntime as ort
        except ImportError as exc:
            raise ImportError(
                "onnxruntime is required for the ONNX GNN backend. "
                "Install with: pip install onnxruntime  "
                "(or onnxruntime-gpu / the DirectML build for GPU engines)"
            ) from exc

        if not os.path.isfile(onnx_path):
            raise FileNotFoundError(f"ONNX model not found: {onnx_path}")

        available = set(ort.get_available_providers())
        wanted = list(providers) if providers is not None else default_ort_providers()
        selected = [p for p in wanted if p in available]
        if not selected:
            selected = ["CPUExecutionProvider"] if "CPUExecutionProvider" in available else list(available)

        self.onnx_path = onnx_path
        self.providers = selected
        self.session = ort.InferenceSession(onnx_path, providers=selected)
        self._input_names = [i.name for i in self.session.get_inputs()]
        self._output_names = [o.name for o in self.session.get_outputs()]

    def forward(
        self,
        node_features: Any,
        edge_features: Any,
        edge_src: Sequence[int],
        edge_dst: Sequence[int],
    ) -> list[float]:
        feeds = {
            "node_features": np.asarray(node_features, dtype=np.float32),
            "edge_features": np.asarray(edge_features, dtype=np.float32),
            "edge_src": np.asarray(edge_src, dtype=np.int64),
            "edge_dst": np.asarray(edge_dst, dtype=np.int64),
        }
        # Only pass inputs the session actually declares (name-stable export).
        feeds = {k: v for k, v in feeds.items() if k in self._input_names}
        outs = self.session.run(self._output_names, feeds)
        return np.asarray(outs[0], dtype=np.float64).reshape(-1).tolist()

    def forward_graph(self, graph: Any) -> list[float]:
        """Run on a DetectorGraph-like object (``.node_features``, etc.)."""
        return self.forward(
            graph.node_features,
            graph.edge_features,
            graph.edge_src,
            graph.edge_dst,
        )


def try_create_onnx_runtime(onnx_path: Optional[str] = None) -> Optional[OnnxGNNRuntime]:
    """Best-effort ONNX runtime construction; returns None if unavailable."""
    path = onnx_path or os.environ.get("QECTOR_GNN_ONNX_PATH")
    if not path or not os.path.isfile(path):
        return None
    try:
        return OnnxGNNRuntime(path)
    except ImportError:
        return None
