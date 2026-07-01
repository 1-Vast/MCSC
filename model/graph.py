"""Small graph knowledge network used by the isolated GKN DTA experiment."""
from __future__ import annotations

import torch
import torch.nn as nn


def dense_normalized_adjacency(num_nodes: int, edges: list[tuple[int, int]], device: torch.device) -> torch.Tensor:
    adj = torch.eye(num_nodes, dtype=torch.float32, device=device)
    if edges:
        src = torch.as_tensor([a for a, _ in edges], dtype=torch.long, device=device)
        dst = torch.as_tensor([b for _, b in edges], dtype=torch.long, device=device)
        adj[src, dst] = 1.0
        adj[dst, src] = 1.0
    deg = adj.sum(dim=1).clamp_min(1.0)
    scale = deg.pow(-0.5)
    return scale[:, None] * adj * scale[None, :]


class GraphKnowledgeNetwork(nn.Module):
    """Two-layer dense GCN for train-target mechanism graphs."""

    def __init__(self, input_dim: int, hidden_dim: int, output_dim: int, dropout: float = 0.1) -> None:
        super().__init__()
        self.lin1 = nn.Linear(input_dim, hidden_dim)
        self.lin2 = nn.Linear(hidden_dim, output_dim)
        self.norm = nn.LayerNorm(output_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, node_feat: torch.Tensor, adjacency: torch.Tensor) -> torch.Tensor:
        h = adjacency @ node_feat
        h = torch.relu(self.lin1(h))
        h = self.dropout(h)
        h = adjacency @ h
        return self.norm(self.lin2(h))
