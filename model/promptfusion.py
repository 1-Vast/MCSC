"""PromptSE+-inspired fusion of LLM mechanism profiles into the isolated GKN refiner.

Adapts PromptSE+'s MLP / Inception-1D-conv / cross-attention branches with learnable residual
weights. All residual weights are zero-initialized, so at initialization the adapter contributes
nothing and the previous best GKN candidate is reproduced exactly. Isolated: imported only by the
enhanced model when the prompt-profile flag is enabled; the mainline never imports this.
"""
from __future__ import annotations

import torch
import torch.nn as nn


class PromptProfileAdapter(nn.Module):
    """Fuse a [B,4,text_dim] family mechanism profile into a d_model target-context vector."""

    def __init__(self, text_dim: int, d_model: int, fusion: str = "mlp-conv-attn",
                 n_heads: int = 4, dropout: float = 0.1) -> None:
        super().__init__()
        self.fusion = fusion
        self.use_conv = fusion in ("mlp-conv", "mlp-conv-attn")
        self.use_attn = fusion == "mlp-conv-attn"
        self.cat_proj = nn.Linear(text_dim, d_model)
        # MLP branch over the averaged (non-None) profile vector.
        self.mlp = nn.Sequential(
            nn.LayerNorm(text_dim),
            nn.Linear(text_dim, d_model),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model, d_model),
        )
        self.alpha_mlp = nn.Parameter(torch.zeros(()))
        if self.use_conv:
            # Inception-style 1D conv over the 4 category tokens (channels = d_model).
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
        target_repr: torch.Tensor,      # [B, d_model] conventional target/pair context (query)
        categories: torch.Tensor,       # [B, 4, text_dim] family profile category vectors
        coverage: torch.Tensor,         # [B, 4] presence mask (1 if category present)
    ) -> torch.Tensor:
        cov = coverage.unsqueeze(-1)                                  # [B,4,1]
        denom = coverage.sum(dim=1, keepdim=True).clamp_min(1.0)      # [B,1]
        avg_profile = (categories * cov).sum(dim=1) / denom           # [B, text_dim]
        context = self.alpha_mlp * self.mlp(avg_profile)
        cat_tokens = self.cat_proj(categories) * cov                  # [B,4,d_model]
        if self.use_conv:
            x = cat_tokens.transpose(1, 2)                           # [B, d_model, 4]
            c = torch.cat([self.conv1(x).mean(-1), self.conv3(x).mean(-1)], dim=-1)
            context = context + self.alpha_conv * self.conv_merge(c)
        if self.use_attn:
            q = self.q_proj(target_repr).unsqueeze(1)                # [B,1,d_model]
            key_pad = coverage <= 0                                  # [B,4] True where absent
            all_absent = key_pad.all(dim=1)
            key_pad = key_pad.clone()
            key_pad[all_absent] = False                             # avoid NaN rows; masked by alpha at init
            attn_out, _ = self.attn(q, cat_tokens, cat_tokens, key_padding_mask=key_pad)
            attn_out = attn_out.squeeze(1)
            attn_out = torch.where(all_absent.unsqueeze(-1), torch.zeros_like(attn_out), attn_out)
            context = context + self.alpha_attn * attn_out
        return context
