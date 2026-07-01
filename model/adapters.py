"""Multimodal descriptor adapters for PRISM."""
from __future__ import annotations

from math import ceil
from typing import Sequence

import torch
import torch.nn as nn


def _split_sizes(width: int, bins: int) -> list[int]:
    chunk = int(ceil(width / bins))
    sizes = []
    remaining = width
    while remaining > 0:
        size = min(chunk, remaining)
        sizes.append(size)
        remaining -= size
    return sizes


class ChunkScaleProjector(nn.Module):
    """Project one descriptor split into several local tokens."""

    def __init__(self, input_dim: int, bins: int, d_model: int, dropout: float) -> None:
        super().__init__()
        self.sizes = _split_sizes(input_dim, bins)
        self.parts = nn.ModuleList([
            nn.Sequential(
                nn.LayerNorm(size),
                nn.Linear(size, d_model),
                nn.GELU(),
                nn.Dropout(dropout),
            )
            for size in self.sizes
        ])

    def forward(self, feat: torch.Tensor) -> torch.Tensor:
        chunks = torch.split(feat, self.sizes, dim=-1)
        return torch.stack([project(chunk) for project, chunk in zip(self.parts, chunks)], dim=1)


class MultiScaleModalityAdapter(nn.Module):
    """Map a frozen descriptor into global and local modality tokens."""

    def __init__(
        self,
        input_dim: int,
        d_model: int,
        dropout: float,
        scales: Sequence[int] = (1, 4),
    ) -> None:
        super().__init__()
        self.projectors = nn.ModuleList([
            ChunkScaleProjector(input_dim, bins, d_model, dropout)
            for bins in scales
        ])
        self.refine = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, d_model * 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model * 2, d_model),
        )
        self.delta_gate = nn.Linear(d_model, d_model)
        self.token_embed = nn.Parameter(torch.zeros(1, sum(len(p.sizes) for p in self.projectors), d_model))
        nn.init.normal_(self.token_embed, mean=0.0, std=0.02)

    def forward(self, feat: torch.Tensor) -> torch.Tensor:
        tokens = torch.cat([projector(feat) for projector in self.projectors], dim=1)
        tokens = tokens + self.token_embed
        return tokens + torch.sigmoid(self.delta_gate(tokens)) * self.refine(tokens)
