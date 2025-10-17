"""
One-layer Transformer model for state estimation.

This module implements a single-layer causal transformer that can learn
to approximate Kalman filter behavior for linear dynamical systems.
"""

from __future__ import annotations

from typing import Optional

import torch
from torch import nn, Tensor

from config import TransformerConfig


class LearnedPositionalEncoding(nn.Module):
    """Learned positional encoding for transformer."""

    def __init__(self, d_model: int, max_len: int):
        super().__init__()
        self.pe = nn.Parameter(torch.zeros(1, max_len, d_model))
        nn.init.normal_(self.pe, std=0.02)

    def forward(self, x: Tensor) -> Tensor:
        T = x.size(1)
        return x + self.pe[:, :T, :]


class OneLayerTransformer(nn.Module):
    """Single-layer causal Transformer for state estimation.

    This model processes sequences of observations (and optional controls)
    to estimate the underlying state of a linear dynamical system.
    """

    def __init__(
        self,
        n_obs: int,
        n_state: int,
        n_ctrl: int = 0,
        cfg: TransformerConfig = None,
    ):
        super().__init__()
        if cfg is None:
            cfg = TransformerConfig()

        in_dim = n_obs + (n_ctrl if n_ctrl > 0 else 0)
        self.in_proj = nn.Linear(in_dim, cfg.d_model)

        # Optional positional encoding
        if cfg.use_positional_encoding:
            self.pos_encoding = LearnedPositionalEncoding(
                cfg.d_model, cfg.max_len
            )
        else:
            self.pos_encoding = None

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=cfg.d_model,
            nhead=cfg.n_heads,
            dim_feedforward=4 * cfg.d_model,
            dropout=cfg.dropout,
            activation="gelu",
            batch_first=True,
            norm_first=False,  # avoid nested-tensor warning path
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=1)
        self.out_proj = nn.Linear(cfg.d_model, n_state)

    @staticmethod
    def causal_mask(T: int, device: torch.device) -> Tensor:
        """Create causal mask for autoregressive behavior (float -inf mask).
        """
        mask = torch.full((T, T), 0.0, device=device)
        upper = torch.triu(torch.ones(T, T, device=device), 1) > 0
        mask = mask.masked_fill(upper, float("-inf"))
        return mask

    def forward(self, y: Tensor, u: Optional[Tensor] = None) -> Tensor:
        """Forward pass through the transformer.

        Args:
            y: (B, T, n_obs) observation sequences
            u: (B, T, n_ctrl) or None, control sequences
        Returns:
            x_hat: (B, T, n_state) state estimates
        """
        if u is not None:
            inp = torch.cat([y, u], dim=-1)
        else:
            inp = y
        h = self.in_proj(inp)

        # Apply positional encoding if enabled
        if self.pos_encoding is not None:
            h = self.pos_encoding(h)

        T = h.size(1)
        mask = self.causal_mask(T, h.device)
        h = self.encoder(h, mask=mask)
        x_hat = self.out_proj(h)
        return x_hat
