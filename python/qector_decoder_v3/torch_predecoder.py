"""PyTorch reference implementation of the Rust GNNPredecoder.

Layer math is kept bit-faithful to ``src/gnn_layers.rs`` so weights round-trip
through safetensors and ONNX export (T5-11) matches the native forward pass.

Message passing is **undirected**: every edge (u, v) contributes both
``u→v`` and ``v→u`` messages (sourced at each endpoint's embedding), matching
the Rust ``forward_cached_with_endpoints`` scheme.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn


class TorchMessagePassingLayer(nn.Module):
    def __init__(self, hidden_size: int, node_feat_dim: int, edge_feat_dim: int):
        super().__init__()
        self.hidden_size = hidden_size
        self.node_feat_dim = node_feat_dim
        self.edge_feat_dim = edge_feat_dim

        self.w_message = nn.Parameter(torch.empty(hidden_size, node_feat_dim + edge_feat_dim))
        self.b_message = nn.Parameter(torch.zeros(hidden_size))

        self.w_update = nn.Parameter(torch.empty(hidden_size, node_feat_dim + hidden_size))
        self.b_update = nn.Parameter(torch.zeros(hidden_size))

        self.reset_parameters()

    def reset_parameters(self):
        # Match the initialization scale in Rust (He-style uniform init)
        scale_msg = (2.0 / (self.node_feat_dim + self.edge_feat_dim)) ** 0.5
        nn.init.uniform_(self.w_message, -scale_msg, scale_msg)

        scale_upd = (2.0 / (self.node_feat_dim + self.hidden_size)) ** 0.5
        nn.init.uniform_(self.w_update, -scale_upd, scale_upd)

    def forward(self, node_embeddings, edge_features, edge_src, edge_dst):
        # node_embeddings: [N, node_feat_dim] (or hidden_size after layer 0)
        # edge_features:   [E, edge_feat_dim]
        # edge_src/dst:    [E] long

        # --- Phase 1: bidirectional messages (Rust-faithful) -----------------
        # msg_from_src = W · [emb[src] ‖ efeat]  → delivered to dst
        # msg_from_dst = W · [emb[dst] ‖ efeat]  → delivered to src
        emb_src = node_embeddings[edge_src]
        emb_dst = node_embeddings[edge_dst]

        msg_from_src = F.linear(torch.cat([emb_src, edge_features], dim=-1), self.w_message, self.b_message)
        msg_from_dst = F.linear(torch.cat([emb_dst, edge_features], dim=-1), self.w_message, self.b_message)

        # --- Phase 2: mean-aggregate messages from neighbours ----------------
        n_nodes = node_embeddings.size(0)
        hidden = self.hidden_size
        device = msg_from_src.device
        dtype = msg_from_src.dtype

        aggregated_sum = torch.zeros(n_nodes, hidden, dtype=dtype, device=device)
        # dst receives message sourced at src; src receives message sourced at dst
        aggregated_sum.scatter_add_(0, edge_dst.unsqueeze(-1).expand(-1, hidden), msg_from_src)
        aggregated_sum.scatter_add_(0, edge_src.unsqueeze(-1).expand(-1, hidden), msg_from_dst)

        degree = torch.zeros(n_nodes, dtype=dtype, device=device)
        ones = torch.ones(edge_src.size(0), dtype=dtype, device=device)
        degree.scatter_add_(0, edge_src, ones)
        degree.scatter_add_(0, edge_dst, ones)
        degree = degree.clamp(min=1.0)
        aggregated = aggregated_sum / degree.unsqueeze(-1)

        # --- Phase 3: node update + ReLU ------------------------------------
        upd_input = torch.cat([node_embeddings, aggregated], dim=-1)
        return F.relu(F.linear(upd_input, self.w_update, self.b_update))


class TorchEdgeReadoutMLP(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int, output_dim: int):
        super().__init__()
        self.w1 = nn.Parameter(torch.empty(hidden_dim, input_dim))
        self.b1 = nn.Parameter(torch.zeros(hidden_dim))
        self.w2 = nn.Parameter(torch.empty(output_dim, hidden_dim))
        self.b2 = nn.Parameter(torch.zeros(output_dim))
        self.reset_parameters()

    def reset_parameters(self):
        scale1 = (2.0 / self.w1.size(1)) ** 0.5
        nn.init.uniform_(self.w1, -scale1, scale1)
        scale2 = (2.0 / self.w2.size(1)) ** 0.5
        nn.init.uniform_(self.w2, -scale2, scale2)

    def forward(self, x):
        h = F.relu(F.linear(x, self.w1, self.b1))
        out = F.linear(h, self.w2, self.b2)
        return out


class TorchGNNPredecoder(nn.Module):
    def __init__(
        self,
        node_feat_dim: int = 10,
        edge_feat_dim: int = 8,
        hidden_size: int = 16,
        n_layers: int = 2,
    ):
        super().__init__()
        self.node_feat_dim = node_feat_dim
        self.edge_feat_dim = edge_feat_dim
        self.hidden_size = hidden_size

        self.layers = nn.ModuleList()
        current_node_dim = node_feat_dim
        for _ in range(n_layers):
            self.layers.append(TorchMessagePassingLayer(hidden_size, current_node_dim, edge_feat_dim))
            current_node_dim = hidden_size

        readout_input_dim = hidden_size + hidden_size + edge_feat_dim
        self.edge_readout = TorchEdgeReadoutMLP(readout_input_dim, hidden_size, 1)

    def forward(self, node_features, edge_features, edge_src, edge_dst):
        node_embeddings = node_features
        for layer in self.layers:
            node_embeddings = layer(node_embeddings, edge_features, edge_src, edge_dst)

        src_emb = node_embeddings[edge_src]
        dst_emb = node_embeddings[edge_dst]
        readout_input = torch.cat([src_emb, dst_emb, edge_features], dim=-1)

        raw_out = self.edge_readout(readout_input).squeeze(-1)  # [E]
        # Match Rust positive_weight: softplus + clamp to [1e-6, 100]
        adjusted_weights = F.softplus(raw_out).clamp(1e-6, 100.0)
        return adjusted_weights

    def save_weights(self, path: str):
        from safetensors.torch import save_file

        # Cast to float64 to match Rust's on-disk safetensors format exactly
        state_dict = {k: v.detach().to(torch.float64).cpu() for k, v in self.state_dict().items()}
        save_file(state_dict, path)

    def load_weights(self, path: str):
        from safetensors.torch import load_file

        loaded = load_file(path)
        dtype = next(self.parameters()).dtype
        state_dict = {k: v.to(dtype) for k, v in loaded.items()}
        self.load_state_dict(state_dict)
