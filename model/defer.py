"""Domain-aware defer gate for the isolated GKN enhanced model."""
from __future__ import annotations

import torch
import torch.nn as nn


class DomainDeferGate(nn.Module):
    """Gate residual correction using pair, memory, and prototype-distance context."""

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
