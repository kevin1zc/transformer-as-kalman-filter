"""
Filterformer model for non-linear/non-Markovian filtering
discrete-time proxy implementation.

This module implements a causal multi-layer transformer tailored for
filtering tasks. It is inspired by
"Transformers Can Solve Non-Linear and Non-Markovian Filtering Problems in
Continuous Time For Conditionally Gaussian Signals" by Horvath, Kratsios,
Limmer, and Yang (2023) [Filterformer].

Reference: https://arxiv.org/pdf/2310.19603

Design notes:
- Causal multi-head self-attention to respect filtering causality.
- Learned positional encoding for sequence position awareness.
- Outputs either state mean only, or mean and (diagonal) covariance
  parameters, enabling training with MSE or NLL objectives.

This is a discrete-time practical implementation suitable for sequence
datasets as used in this repository. It does not implement the
continuous-time limit nor the exact Wasserstein-geometry-tailored
attention from the paper, but follows the core architectural spirit of
a causal transformer for filtering.
"""

from __future__ import annotations

from typing import Optional, Tuple

import torch
from torch import nn, Tensor

from config import TransformerConfig


class LearnedPositionalEncoding(nn.Module):
    def __init__(self, d_model: int, max_len: int):
        super().__init__()
        self.pe = nn.Parameter(torch.zeros(1, max_len, d_model))
        nn.init.normal_(self.pe, std=0.02)

    def forward(self, x: Tensor) -> Tensor:
        return x + self.pe[:, : x.size(1), :]


class CausalTransformerBlock(nn.Module):
    def __init__(self, d_model: int, n_heads: int, dropout: float):
        super().__init__()
        self.attn = nn.MultiheadAttention(
            embed_dim=d_model,
            num_heads=n_heads,
            dropout=dropout,
            batch_first=True,
        )
        self.attn_norm = nn.LayerNorm(d_model)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, 4 * d_model),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(4 * d_model, d_model),
        )
        self.ffn_norm = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    @staticmethod
    def causal_mask(T: int, device: torch.device) -> Tensor:
        # Float mask with -inf on masked positions to avoid kernel issues
        mask = torch.full((T, T), 0.0, device=device)
        upper = torch.triu(torch.ones(T, T, device=device), 1) > 0
        mask = mask.masked_fill(upper, float("-inf"))
        return mask

    def forward(self, x: Tensor) -> Tensor:
        T = x.size(1)
        mask = self.causal_mask(T, x.device)
        attn_out, _ = self.attn(x, x, x, attn_mask=mask)
        x = self.attn_norm(x + self.dropout(attn_out))
        ffn_out = self.ffn(x)
        x = self.ffn_norm(x + self.dropout(ffn_out))
        return x


class Filterformer(nn.Module):
    """Filterformer: causal transformer for sequence filtering.

    Args:
        n_obs: number of observation channels
        n_state: latent state dimension to estimate
        n_ctrl: number of control channels (0 if none)
        cfg: TransformerConfig for core dimensions and options
        num_layers: number of transformer blocks
        output_covariance: if True, output diagonal covariance parameters in
            addition to mean (shape: (B, T, n_state) for both).

    Forward:
        Inputs: y (B, T, n_obs), optional u (B, T, n_ctrl)
        Returns: x_mean if output_covariance is False; otherwise
        (x_mean, x_std)
    """

    def __init__(
        self,
        n_obs: int,
        n_state: int,
        n_ctrl: int = 0,
        cfg: TransformerConfig | None = None,
        *,
        num_layers: int = 4,
        output_covariance: bool = False,
    ):
        super().__init__()
        if cfg is None:
            cfg = TransformerConfig()

        in_dim = n_obs + (n_ctrl if n_ctrl > 0 else 0)
        self.in_proj = nn.Linear(in_dim, cfg.d_model)

        # Always use positional encoding regardless of config
        self.pos_encoding = LearnedPositionalEncoding(cfg.d_model, cfg.max_len)

        blocks = []
        for _ in range(num_layers):
            block = CausalTransformerBlock(
                cfg.d_model, cfg.n_heads, cfg.dropout
            )
            blocks.append(block)
        self.blocks = nn.ModuleList(blocks)

        self.out_mean = nn.Linear(cfg.d_model, n_state)
        self.output_covariance = output_covariance
        if output_covariance:
            # Parameterize std via softplus to ensure positivity
            self.out_log_std = nn.Linear(cfg.d_model, n_state)
            self.softplus = nn.Softplus()

    def forward(
        self,
        y: Tensor,
        u: Optional[Tensor] = None,
    ) -> Tensor | Tuple[Tensor, Tensor]:
        if u is not None:
            x_in = torch.cat([y, u], dim=-1)
        else:
            x_in = y
        h = self.in_proj(x_in)
        h = self.pos_encoding(h)

        for blk in self.blocks:
            h = blk(h)

        mean = self.out_mean(h)
        if not self.output_covariance:
            return mean
        log_std = self.out_log_std(h)
        std = self.softplus(log_std) + 1e-6
        return mean, std
