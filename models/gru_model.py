"""
GRU model for state estimation.

This module implements a GRU-based model that can learn to approximate
Kalman filter behavior for linear dynamical systems.
"""

from __future__ import annotations

from typing import Optional

import torch
from torch import nn, Tensor

from config import GRUConfig


class GRUStateEstimator(nn.Module):
    """GRU-based state estimator.

    This model processes sequences of observations (and optional controls)
    to estimate the underlying state of a linear dynamical system.
    """

    def __init__(
        self, n_obs: int, n_state: int, n_ctrl: int = 0, cfg: GRUConfig = None
    ):
        super().__init__()
        if cfg is None:
            cfg = GRUConfig()

        in_dim = n_obs + (n_ctrl if n_ctrl > 0 else 0)
        self.gru = nn.GRU(
            input_size=in_dim,
            hidden_size=cfg.hidden_size,
            num_layers=cfg.num_layers,
            batch_first=True,
            dropout=cfg.dropout if cfg.dropout > 0 else 0,
            bidirectional=cfg.bidirectional,
        )
        self.out_proj = nn.Linear(cfg.hidden_size, n_state)

    def forward(self, y: Tensor, u: Optional[Tensor] = None) -> Tensor:
        """Forward pass through the GRU.

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

        gru_out, _ = self.gru(inp)
        x_hat = self.out_proj(gru_out)
        return x_hat
