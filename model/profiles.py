"""Mechanism profile fusion for PRISM."""
from __future__ import annotations

import torch
import torch.nn as nn


class MechanismProfileFusion(nn.Module):
    """Fuse a [B,4,text_dim] family mechanism profile into a target-context vector."""

    def __init__(self, text_dim: int, d_model: int, fusion: str = "mlp-conv-attn",
                 n_heads: int = 4, dropout: float = 0.1) -> None:
        super().__init__()
        self.fusion = fusion
        self.use_conv = fusion in ("mlp-conv", "mlp-conv-attn")
        self.use_attn = fusion == "mlp-conv-attn"
        self.cat_proj = nn.Linear(text_dim, d_model)
        self.mlp = nn.Sequential(
            nn.LayerNorm(text_dim),
            nn.Linear(text_dim, d_model),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model, d_model),
        )
        self.alpha_mlp = nn.Parameter(torch.zeros(()))
        if self.use_conv:
            self.conv1 = nn.Conv1d(d_model, d_model, kernel_size=1)
            self.conv3 = nn.Conv1d(d_model, d_model, kernel_size=3, padding=1)
            self.conv_merge = nn.Linear(2 * d_model, d_model)
            self.alpha_conv = nn.Parameter(torch.zeros(()))
        if self.use_attn:
            self.q_proj = nn.Linear(d_model, d_model)
            self.attn = nn.MultiheadAttention(d_model, n_heads, dropout=dropout, batch_first=True)
            self.alpha_attn = nn.Parameter(torch.zeros(()))

    def forward(
        self,
        target_repr: torch.Tensor,
        categories: torch.Tensor,
        coverage: torch.Tensor,
    ) -> torch.Tensor:
        cov = coverage.unsqueeze(-1)
        denom = coverage.sum(dim=1, keepdim=True).clamp_min(1.0)
        avg_profile = (categories * cov).sum(dim=1) / denom
        context = self.alpha_mlp * self.mlp(avg_profile)
        cat_tokens = self.cat_proj(categories) * cov
        if self.use_conv:
            x = cat_tokens.transpose(1, 2)
            c = torch.cat([self.conv1(x).mean(-1), self.conv3(x).mean(-1)], dim=-1)
            context = context + self.alpha_conv * self.conv_merge(c)
        if self.use_attn:
            q = self.q_proj(target_repr).unsqueeze(1)
            key_pad = coverage <= 0
            all_absent = key_pad.all(dim=1)
            key_pad = key_pad.clone()
            key_pad[all_absent] = False
            attn_out, _ = self.attn(q, cat_tokens, cat_tokens, key_padding_mask=key_pad)
            attn_out = attn_out.squeeze(1)
            attn_out = torch.where(all_absent.unsqueeze(-1), torch.zeros_like(attn_out), attn_out)
            context = context + self.alpha_attn * attn_out
        return context
