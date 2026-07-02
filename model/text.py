"""Mechanism text projection for PRISM."""
from __future__ import annotations

import torch
import torch.nn as nn

from model.tokens import MultiScaleDescriptorAdapter


class MechanismTextProjector(nn.Module):
    """Map cached mechanism text hashes into the shared DTA width."""

    def __init__(self, input_dim: int, d_model: int, dropout: float) -> None:
        super().__init__()
        self.adapter = MultiScaleDescriptorAdapter(input_dim, d_model, dropout, scales=(1, 4))

    def forward(self, text_feat: torch.Tensor) -> torch.Tensor:
        return self.adapter(text_feat)

    def pool(self, text_feat: torch.Tensor) -> torch.Tensor:
        return self.forward(text_feat).mean(dim=1)
