"""Target-domain graph prototypes and residual trust gating for PRISM."""
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


class TargetDomainGraphEncoder(nn.Module):
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


class ResidualTrustGate(nn.Module):
    """Gate residual correction using pair, memory, and domain context."""

    def __init__(
        self,
        pair_dim: int,
        mem_dim: int,
        domain_dim: int,
        hidden: int,
        dropout: float,
        min_gamma: float = 0.0,
    ) -> None:
        super().__init__()
        self.mem_dim = int(mem_dim)
        self.domain_dim = int(domain_dim)
        self.min_gamma = float(min_gamma)
        in_dim = int(pair_dim) + self.mem_dim + self.domain_dim
        self.net = nn.Sequential(
            nn.LayerNorm(in_dim),
            nn.Linear(in_dim, hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, 1),
        )

    def forward(
        self,
        pair_repr: torch.Tensor,
        mem_feat: torch.Tensor | None,
        domain_dist: torch.Tensor | None,
    ) -> torch.Tensor:
        parts = [pair_repr]
        if self.mem_dim:
            if mem_feat is None:
                mem_feat = pair_repr.new_zeros((pair_repr.shape[0], self.mem_dim))
            parts.append(mem_feat)
        if self.domain_dim:
            if domain_dist is None:
                domain_dist = pair_repr.new_zeros((pair_repr.shape[0], self.domain_dim))
            parts.append(domain_dist)
        raw = torch.sigmoid(self.net(torch.cat(parts, dim=-1)).squeeze(-1))
        return self.min_gamma + (1.0 - self.min_gamma) * raw
