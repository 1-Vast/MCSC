"""Memory-calibrated Transformer refiners for PRISM."""
from __future__ import annotations

import torch
import torch.nn as nn

from model.adapters import MultiScaleModalityAdapter
from model.attention import GatedCrossAttentionBlock
from model.space import SharedSpaceInitializer


def _bounded_residual(raw: torch.Tensor, residual_scale: float | None) -> torch.Tensor:
    if residual_scale is None or residual_scale <= 0:
        return raw
    return float(residual_scale) * torch.tanh(raw / float(residual_scale))


class ResidualPath(nn.Module):
    """Two-layer residual path used by the Transformer regression head."""

    def __init__(self, width: int, hidden: int, dropout: float) -> None:
        super().__init__()
        self.in_proj = nn.Linear(width, hidden)
        self.body = nn.Sequential(
            nn.LayerNorm(hidden),
            nn.Linear(hidden, hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, hidden),
            nn.Dropout(dropout),
        )
        self.out_norm = nn.LayerNorm(hidden)

    def forward(self, feat: torch.Tensor) -> torch.Tensor:
        base = self.in_proj(feat)
        return self.out_norm(base + self.body(base))


class PrismMemoryRefiner(nn.Module):
    """Pure PRISM memory-calibrated cross-attention residual refiner."""

    def __init__(
        self,
        drug_dim: int,
        target_dim: int,
        d_model: int = 384,
        n_heads: int = 8,
        n_layers: int = 4,
        ff_dim: int = 1536,
        dropout: float = 0.2,
        residual_scale: float | None = None,
        mem_dim: int = 0,
        text_dim: int = 0,
    ) -> None:
        super().__init__()
        if text_dim:
            raise ValueError("pure PrismMemoryRefiner does not accept text/LLM features")
        self.residual_scale = residual_scale
        self.mem_dim = int(mem_dim)
        self.drug_adapter = MultiScaleModalityAdapter(drug_dim, d_model, dropout, scales=(1, 4))
        self.target_adapter = MultiScaleModalityAdapter(target_dim, d_model, dropout, scales=(1, 4))
        self.type_embed = nn.Parameter(torch.zeros(3, d_model))
        self.space = SharedSpaceInitializer(d_model, dropout)
        self.cross_modal_layers = nn.ModuleList([
            GatedCrossAttentionBlock(d_model, n_heads, ff_dim, dropout)
            for _ in range(n_layers)
        ])
        feature_dim = d_model * 5
        head_width = max(128, d_model)
        self.path_a = ResidualPath(feature_dim, head_width, dropout)
        self.path_b = ResidualPath(feature_dim, head_width, dropout)
        self.merge = nn.Sequential(
            nn.LayerNorm(head_width * 2),
            nn.Linear(head_width * 2, d_model),
            nn.GELU(),
            nn.Dropout(dropout),
        )
        self.residual_head = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, max(64, d_model // 2)),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(max(64, d_model // 2), 1),
        )
        self.gate_head = nn.Sequential(
            nn.LayerNorm(d_model + self.mem_dim),
            nn.Linear(d_model + self.mem_dim, max(64, d_model // 2)),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(max(64, d_model // 2), 1),
        )
        nn.init.normal_(self.type_embed, mean=0.0, std=0.02)

    def _mem_or_zeros(self, token: torch.Tensor, mem_feat: torch.Tensor | None) -> torch.Tensor | None:
        if self.mem_dim == 0:
            return None
        if mem_feat is not None:
            return mem_feat
        return token.new_zeros((token.shape[0], self.mem_dim))

    def encode(
        self,
        drug_feat: torch.Tensor,
        target_feat: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        drug_tokens = self.drug_adapter(drug_feat) + self.type_embed[0].view(1, 1, -1)
        target_tokens = self.target_adapter(target_feat) + self.type_embed[1].view(1, 1, -1)
        shared_tokens = self.space(drug_tokens, target_tokens) + self.type_embed[2].view(1, 1, -1)
        for block in self.cross_modal_layers:
            drug_tokens, target_tokens, shared_tokens = block(drug_tokens, target_tokens, shared_tokens)
        return drug_tokens.mean(dim=1), target_tokens.mean(dim=1), shared_tokens.mean(dim=1)

    def _path(self, drug_feat: torch.Tensor, target_feat: torch.Tensor) -> torch.Tensor:
        drug_pool, target_pool, shared_pool = self.encode(drug_feat, target_feat)
        fused = torch.cat([
            drug_pool,
            target_pool,
            shared_pool,
            (drug_pool - target_pool).abs(),
            drug_pool * target_pool,
        ], dim=-1)
        return self.merge(torch.cat([self.path_a(fused), self.path_b(fused)], dim=-1))

    def modality_pools(
        self,
        drug_feat: torch.Tensor,
        target_feat: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        return self.encode(drug_feat, target_feat)

    def pair_representation(self, drug_feat: torch.Tensor, target_feat: torch.Tensor) -> torch.Tensor:
        return self._path(drug_feat, target_feat)

    def residual_gate(
        self,
        drug_feat: torch.Tensor,
        target_feat: torch.Tensor,
        mem_feat: torch.Tensor | None = None,
        text_feat: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        _ = text_feat
        path = self._path(drug_feat, target_feat)
        residual = _bounded_residual(self.residual_head(path).squeeze(-1), self.residual_scale)
        mem = self._mem_or_zeros(path, mem_feat)
        gate_in = path if mem is None else torch.cat([path, mem], dim=-1)
        gate = torch.sigmoid(self.gate_head(gate_in)).squeeze(-1)
        return residual, gate

    def forward(
        self,
        drug_feat: torch.Tensor,
        target_feat: torch.Tensor,
        prior: torch.Tensor,
        mem_feat: torch.Tensor | None = None,
        text_feat: torch.Tensor | None = None,
    ) -> torch.Tensor:
        residual, gate = self.residual_gate(drug_feat, target_feat, mem_feat, text_feat)
        return prior + gate * residual
