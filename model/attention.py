"""Cross-modal attention blocks for PRISM."""
from __future__ import annotations

import torch
import torch.nn as nn


def attention_entropy(weights: torch.Tensor) -> torch.Tensor:
    """Return normalized attention entropy for weights shaped B x H x Q x K."""
    weights = weights.float().clamp_min(1e-8)
    entropy = -(weights * weights.log()).sum(dim=-1)
    denom = torch.log(torch.tensor(weights.shape[-1], dtype=weights.dtype, device=weights.device)).clamp_min(1e-8)
    return entropy / denom


class GatedCrossAttentionBlock(nn.Module):
    """Adaptive gated bidirectional cross-attention cascade."""

    def __init__(self, d_model: int, n_heads: int, ff_dim: int, dropout: float) -> None:
        super().__init__()
        self.drug_attn = nn.MultiheadAttention(d_model, n_heads, dropout=dropout, batch_first=True)
        self.target_attn = nn.MultiheadAttention(d_model, n_heads, dropout=dropout, batch_first=True)
        self.shared_attn = nn.MultiheadAttention(d_model, n_heads, dropout=dropout, batch_first=True)
        self.drug_base = nn.Linear(d_model, d_model)
        self.target_base = nn.Linear(d_model, d_model)
        self.shared_base = nn.Linear(d_model, d_model)
        self.drug_gate = nn.Linear(d_model * 2, d_model)
        self.target_gate = nn.Linear(d_model * 2, d_model)
        self.shared_gate = nn.Linear(d_model * 2, d_model)
        self.drug_norm = nn.LayerNorm(d_model)
        self.target_norm = nn.LayerNorm(d_model)
        self.shared_norm = nn.LayerNorm(d_model)
        self.context_norm = nn.LayerNorm(d_model)
        self.drug_ff = self._ffn(d_model, ff_dim, dropout)
        self.target_ff = self._ffn(d_model, ff_dim, dropout)
        self.shared_ff = self._ffn(d_model, ff_dim, dropout)
        self.drop = nn.Dropout(dropout)

    @staticmethod
    def _ffn(d_model: int, ff_dim: int, dropout: float) -> nn.Sequential:
        return nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, ff_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(ff_dim, d_model),
            nn.Dropout(dropout),
        )

    def _mix(
        self,
        tokens: torch.Tensor,
        context: torch.Tensor,
        attn: nn.MultiheadAttention,
        base: nn.Linear,
        gate: nn.Linear,
        norm: nn.LayerNorm,
        ff: nn.Sequential,
        return_stats: bool = False,
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        query = norm(tokens)
        ctx = self.context_norm(context)
        exchanged, attn_weights = attn(
            query,
            ctx,
            ctx,
            need_weights=return_stats,
            average_attn_weights=False,
        )
        transformed = base(query)
        weights = torch.sigmoid(gate(torch.cat([transformed, exchanged], dim=-1)))
        mixed = (1.0 - weights) * transformed + weights * exchanged
        out = tokens + self.drop(mixed)
        out = out + ff(out)
        if not return_stats:
            return out
        entropy = attention_entropy(attn_weights).mean(dim=(1, 2), keepdim=False).unsqueeze(-1)
        peak = attn_weights.float().amax(dim=-1).mean(dim=(1, 2), keepdim=False).unsqueeze(-1)
        return out, torch.cat([entropy, peak], dim=-1)

    def forward(
        self,
        drug_tokens: torch.Tensor,
        target_tokens: torch.Tensor,
        shared_tokens: torch.Tensor,
        return_stats: bool = False,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor] | tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        drug_context = torch.cat([target_tokens, shared_tokens], dim=1)
        target_context = torch.cat([drug_tokens, shared_tokens], dim=1)
        drug_result = self._mix(
            drug_tokens, drug_context, self.drug_attn, self.drug_base,
            self.drug_gate, self.drug_norm, self.drug_ff, return_stats=return_stats,
        )
        if return_stats:
            drug_next, drug_stats = drug_result
        else:
            drug_next = drug_result
            drug_stats = None
        target_result = self._mix(
            target_tokens, target_context, self.target_attn, self.target_base,
            self.target_gate, self.target_norm, self.target_ff, return_stats=return_stats,
        )
        if return_stats:
            target_next, target_stats = target_result
        else:
            target_next = target_result
            target_stats = None
        shared_context = torch.cat([drug_next, target_next, shared_tokens], dim=1)
        shared_result = self._mix(
            shared_tokens, shared_context, self.shared_attn, self.shared_base,
            self.shared_gate, self.shared_norm, self.shared_ff, return_stats=return_stats,
        )
        if return_stats:
            shared_next, shared_stats = shared_result
            stats = torch.stack([drug_stats, target_stats, shared_stats], dim=1).mean(dim=1)
            return drug_next, target_next, shared_next, stats
        shared_next = shared_result
        return drug_next, target_next, shared_next
