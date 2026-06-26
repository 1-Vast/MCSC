"""Neural residual refiner used by the current MCSC mainline."""
from __future__ import annotations

from typing import Sequence

import torch
import torch.nn as nn


def _norm_layer(width: int, norm: str) -> nn.Module | None:
    if norm == "batch":
        return nn.BatchNorm1d(width)
    if norm == "layer":
        return nn.LayerNorm(width)
    if norm == "none":
        return None
    raise ValueError(f"unknown norm: {norm}")


def _bounded_residual(raw: torch.Tensor, residual_scale: float | None) -> torch.Tensor:
    if residual_scale is None or residual_scale <= 0:
        return raw
    return float(residual_scale) * torch.tanh(raw / float(residual_scale))


class ResidualRefiner(nn.Module):
    """Compact residual corrector on top of a frozen scalar prior."""

    def __init__(
        self,
        drug_dim: int,
        target_dim: int,
        hidden: Sequence[int] = (256, 128, 64),
        dropout: float = 0.3,
        norm: str = "batch",
        residual_scale: float | None = None,
    ) -> None:
        super().__init__()
        self.residual_scale = residual_scale
        layers: list[nn.Module] = []
        in_dim = drug_dim + target_dim
        for width in hidden:
            layers.append(nn.Linear(in_dim, width))
            norm_mod = _norm_layer(width, norm)
            if norm_mod is not None:
                layers.append(norm_mod)
            layers += [nn.GELU(), nn.Dropout(dropout)]
            in_dim = width
        self.trunk = nn.Sequential(*layers)
        self.residual_head = nn.Linear(in_dim, 1)
        self.gate_head = nn.Linear(in_dim, 1)

    def _residual_gate(self, drug_feat: torch.Tensor, target_feat: torch.Tensor):
        h = self.trunk(torch.cat([drug_feat, target_feat], dim=-1))
        residual = _bounded_residual(self.residual_head(h).squeeze(-1), self.residual_scale)
        gate = torch.sigmoid(self.gate_head(h)).squeeze(-1)
        return residual, gate

    def forward(self, drug_feat: torch.Tensor, target_feat: torch.Tensor, prior: torch.Tensor) -> torch.Tensor:
        residual, gate = self._residual_gate(drug_feat, target_feat)
        return prior + gate * residual

    @torch.no_grad()
    def gate_mean(self, drug_feat: torch.Tensor, target_feat: torch.Tensor) -> float:
        self.eval()
        _, gate = self._residual_gate(drug_feat, target_feat)
        return float(gate.mean().item())
