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
        # Match the initialization scale in Rust (Xavier/He style uniform init)
        scale_msg = (2.0 / (self.node_feat_dim + self.edge_feat_dim)) ** 0.5
        nn.init.uniform_(self.w_message, -scale_msg, scale_msg)

        scale_upd = (2.0 / (self.node_feat_dim + self.hidden_size)) ** 0.5
        nn.init.uniform_(self.w_update, -scale_upd, scale_upd)

    def forward(self, node_embeddings, edge_features, edge_src, edge_dst):
        # node_embeddings: [N, hidden_size]
        # edge_features: [E, edge_feat_dim]
        # edge_src, edge_dst: [E] (long tensors)

        # 1. Message calculation
        src_embeddings = node_embeddings[edge_src]  # [E, current_node_dim]
        msg_input = torch.cat([src_embeddings, edge_features], dim=-1)  # [E, current_node_dim + edge_feat_dim]
        messages = F.linear(msg_input, self.w_message, self.b_message)  # [E, hidden_size]

        # 2. Aggregation (undirected, average over incident edges)
        N = node_embeddings.size(0)
        H = self.hidden_size

        sum_src = torch.zeros(N, H, dtype=messages.dtype, device=messages.device)
        sum_src.scatter_add_(0, edge_src.unsqueeze(-1).expand(-1, H), messages)

        sum_dst = torch.zeros(N, H, dtype=messages.dtype, device=messages.device)
        sum_dst.scatter_add_(0, edge_dst.unsqueeze(-1).expand(-1, H), messages)

        aggregated_sum = sum_src + sum_dst

        deg_src = torch.zeros(N, dtype=messages.dtype, device=messages.device)
        deg_src.scatter_add_(0, edge_src, torch.ones_like(edge_src, dtype=messages.dtype))

        deg_dst = torch.zeros(N, dtype=messages.dtype, device=messages.device)
        deg_dst.scatter_add_(0, edge_dst, torch.ones_like(edge_dst, dtype=messages.dtype))

        degree = (deg_src + deg_dst).clamp(min=1.0)
        aggregated = aggregated_sum / degree.unsqueeze(-1)

        # 3. Update
        upd_input = torch.cat([node_embeddings, aggregated], dim=-1)  # [N, current_node_dim + hidden_size]
        new_embeddings = F.relu(F.linear(upd_input, self.w_update, self.b_update))

        return new_embeddings


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
    def __init__(self, node_feat_dim: int = 10, edge_feat_dim: int = 8, hidden_size: int = 16, n_layers: int = 2):
        super().__init__()
        self.node_feat_dim = node_feat_dim
        self.edge_feat_dim = edge_feat_dim
        self.hidden_size = hidden_size

        # Layers
        self.layers = nn.ModuleList()
        current_node_dim = node_feat_dim
        for _ in range(n_layers):
            self.layers.append(TorchMessagePassingLayer(hidden_size, current_node_dim, edge_feat_dim))
            current_node_dim = hidden_size

        # Readout MLP
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
        adjusted_weights = F.softplus(raw_out).clamp(1e-6, 100.0)
        return adjusted_weights

    def save_weights(self, path: str):
        from safetensors.torch import save_file

        # Cast to float64 to match Rust's expected binary format exactly
        state_dict = {k: v.to(torch.float64).cpu() for k, v in self.state_dict().items()}
        save_file(state_dict, path)

    def load_weights(self, path: str):
        from safetensors.torch import load_file

        loaded = load_file(path)
        dtype = next(self.parameters()).dtype
        state_dict = {k: v.to(dtype) for k, v in loaded.items()}
        self.load_state_dict(state_dict)
