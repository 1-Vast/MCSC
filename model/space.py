"""Shared latent space initializer for drug-target modality fusion."""
from __future__ import annotations

import torch
import torch.nn as nn


class SharedSpaceInitializer(nn.Module):
    """Gated projection from drug/target tokens into a shared interaction token."""

    def __init__(self, d_model: int, dropout: float) -> None:
        super().__init__()
        self.gate = nn.Sequential(
            nn.LayerNorm(d_model * 2),
            nn.Linear(d_model * 2, d_model),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model, d_model),
        )
        self.delta = nn.Sequential(
            nn.LayerNorm(d_model * 2),
            nn.Linear(d_model * 2, d_model * 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model * 2, d_model),
        )
        self.norm = nn.LayerNorm(d_model)

    def forward(self, drug_tokens: torch.Tensor, target_tokens: torch.Tensor) -> torch.Tensor:
        drug_pool = drug_tokens.mean(dim=1)
        target_pool = target_tokens.mean(dim=1)
        joined = torch.cat([drug_pool, target_pool], dim=-1)
        gate = torch.sigmoid(self.gate(joined))
        fused = gate * drug_pool + (1.0 - gate) * target_pool
        return self.norm(fused + self.delta(joined)).unsqueeze(1)
