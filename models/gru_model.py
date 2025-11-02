"""
GRU model for state estimation.

This module implements a GRU-based model that can learn to approximate
filtering behavior for nonlinear dynamical systems.
"""

from __future__ import annotations

from typing import Optional

import torch
from torch import nn, Tensor


class GRUConfig:
    """Configuration for GRU models."""

    def __init__(
        self,
        hidden_size: int = 128,
        num_layers: int = 2,
        dropout: float = 0.1,
        bidirectional: bool = False,
    ):
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.dropout = dropout
        self.bidirectional = bidirectional


class GRUStateEstimator(nn.Module):
    """GRU-based state estimator for nonlinear systems.

    This model processes sequences of observations (and optional controls)
    to estimate the underlying state of nonlinear dynamical systems.
    Enhanced with additional nonlinear processing layers for better
    handling of complex dynamics.
    """

    def __init__(
        self, n_obs: int, n_state: int, n_ctrl: int = 0, cfg: GRUConfig = None
    ):
        super().__init__()
        if cfg is None:
            cfg = GRUConfig()

        in_dim = n_obs + (n_ctrl if n_ctrl > 0 else 0)

        # Input preprocessing for nonlinear systems
        self.input_proj = nn.Sequential(
            nn.Linear(in_dim, cfg.hidden_size),
            nn.ReLU(),
            nn.Dropout(cfg.dropout if cfg.dropout > 0 else 0),
        )

        # GRU layers
        self.gru = nn.GRU(
            input_size=cfg.hidden_size,
            hidden_size=cfg.hidden_size,
            num_layers=cfg.num_layers,
            batch_first=True,
            dropout=cfg.dropout if cfg.dropout > 0 else 0,
            bidirectional=cfg.bidirectional,
        )

        # Output processing with nonlinear layers
        gru_output_size = cfg.hidden_size * (2 if cfg.bidirectional else 1)
        self.output_proj = nn.Sequential(
            nn.Linear(gru_output_size, cfg.hidden_size),
            nn.ReLU(),
            nn.Dropout(cfg.dropout if cfg.dropout > 0 else 0),
            nn.Linear(cfg.hidden_size, n_state),
        )

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

        # Input preprocessing
        inp_processed = self.input_proj(inp)

        # GRU processing
        gru_out, _ = self.gru(inp_processed)

        # Output processing
        x_hat = self.output_proj(gru_out)
        return x_hat
