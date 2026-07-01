"""PRISM selective refiner with GKN prototypes and DeepSeek-QC signals."""
from __future__ import annotations

import torch
import torch.nn as nn

from model.defer import DomainDeferGate
from model.promptfusion import PromptProfileAdapter
from model.refiners import PrismMemoryRefiner, _bounded_residual
from model.text import MechanismTextAdapter


class PrismSelectiveRefiner(nn.Module):
    """PRISM extension with mechanism text, GKN prototypes, and selective defer."""

    def __init__(
        self,
        drug_dim: int,
        target_dim: int,
        text_dim: int,
        domain_dim: int,
        d_model: int = 384,
        n_heads: int = 8,
        n_layers: int = 4,
        ff_dim: int = 1536,
        dropout: float = 0.2,
        residual_scale: float | None = None,
        mem_dim: int = 0,
        min_gamma: float = 0.0,
        prompt_profile_dim: int = 0,
        prompt_fusion: str = "off",
    ) -> None:
        super().__init__()
        self.residual_scale = residual_scale
        self.prompt_adapter = (
            PromptProfileAdapter(prompt_profile_dim, d_model, prompt_fusion, n_heads, dropout)
            if prompt_profile_dim and prompt_fusion != "off" else None
        )
        self.align_proj = nn.Linear(prompt_profile_dim, d_model) if prompt_profile_dim else None
        self.base = PrismMemoryRefiner(
            drug_dim,
            target_dim,
            d_model=d_model,
            n_heads=n_heads,
            n_layers=n_layers,
            ff_dim=ff_dim,
            dropout=dropout,
            residual_scale=residual_scale,
            mem_dim=0,
            text_dim=0,
        )
        self.text_adapter = MechanismTextAdapter(text_dim, d_model, dropout) if text_dim else None
        self.text_merge = nn.Sequential(
            nn.LayerNorm(d_model * 2),
            nn.Linear(d_model * 2, d_model),
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
        # Pair-affinity auxiliary head: a small head that regresses directly to y from the
        # cross-attn pair representation, used only as a training-time loss to force the
        # backbone to encode pair-affinity signal in the fused representation. NOT used at
        # inference: the prediction pathway remains prior + gamma * residual.
        self.pair_affinity_head = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, max(64, d_model // 2)),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(max(64, d_model // 2), 1),
        )
        self.defer_gate = DomainDeferGate(
            pair_dim=d_model,
            mem_dim=mem_dim,
            domain_dim=domain_dim,
            hidden=max(64, d_model // 2),
            dropout=dropout,
            min_gamma=min_gamma,
        )

    def profile_align(self, avg_profile: torch.Tensor) -> torch.Tensor:
        return self.align_proj(avg_profile) if self.align_proj is not None else avg_profile

    def pair_representation(
        self,
        drug_feat: torch.Tensor,
        target_feat: torch.Tensor,
        text_feat: torch.Tensor | None = None,
    ) -> torch.Tensor:
        pair = self.base.pair_representation(drug_feat, target_feat)
        if self.text_adapter is None or text_feat is None:
            return pair
        text_pool = self.text_adapter.pool(text_feat)
        return self.text_merge(torch.cat([pair, text_pool], dim=-1))

    def residual_gate(
        self,
        drug_feat: torch.Tensor,
        target_feat: torch.Tensor,
        mem_feat: torch.Tensor | None = None,
        text_feat: torch.Tensor | None = None,
        domain_dist: torch.Tensor | None = None,
        prompt_categories: torch.Tensor | None = None,
        prompt_coverage: torch.Tensor | None = None,
        return_pair: bool = False,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        pair = self.pair_representation(drug_feat, target_feat, text_feat)
        if self.prompt_adapter is not None and prompt_categories is not None:
            pair = pair + self.prompt_adapter(pair, prompt_categories, prompt_coverage)
        residual = _bounded_residual(self.residual_head(pair).squeeze(-1), self.residual_scale)
        gamma = self.defer_gate(pair, mem_feat, domain_dist)
        if return_pair:
            return residual, gamma, pair
        return residual, gamma

    def pair_affinity(self, pair: torch.Tensor) -> torch.Tensor:
        """Auxiliary affinity prediction from the pair representation (train-time only)."""
        return self.pair_affinity_head(pair).squeeze(-1)

    def forward(
        self,
        drug_feat: torch.Tensor,
        target_feat: torch.Tensor,
        prior: torch.Tensor,
        mem_feat: torch.Tensor | None = None,
        text_feat: torch.Tensor | None = None,
        domain_dist: torch.Tensor | None = None,
        prompt_categories: torch.Tensor | None = None,
        prompt_coverage: torch.Tensor | None = None,
    ) -> torch.Tensor:
        residual, gamma = self.residual_gate(
            drug_feat, target_feat, mem_feat, text_feat, domain_dist,
            prompt_categories, prompt_coverage,
        )
        return prior + gamma * residual
