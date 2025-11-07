"""
Mamba and Mamba2 models for state estimation.

Based on mamba-ssm package: https://github.com/state-spaces/mamba
- Mamba: Selective state space model with hardware-aware algorithm
- Mamba2: Improved version with structured state space duality (SSD)
"""

from __future__ import annotations

from typing import Optional

import torch
from torch import nn, Tensor
from mamba_ssm import Mamba, Mamba2
from config.model_config import MambaConfig, Mamba2Config


class MambaStateEstimator(nn.Module):
    """
    Mamba-based state estimator for filtering tasks.

    Uses selective state space layers for efficient sequence modeling.
    """

    def __init__(
        self,
        n_obs: int,
        n_state: int,
        n_ctrl: int = 0,
        cfg: Optional[MambaConfig] = None,
    ):
        super().__init__()
        cfg = cfg or MambaConfig()
        self.n_obs, self.n_state, self.n_ctrl = n_obs, n_state, n_ctrl
        self.cfg = cfg

        # Observation embedding
        self.obs_embed = nn.Linear(n_obs, cfg.d_model)

        # Stack of Mamba layers
        self.layers = nn.ModuleList(
            [
                Mamba(
                    d_model=cfg.d_model,
                    d_state=cfg.d_state,
                    d_conv=cfg.d_conv,
                    expand=cfg.expand,
                )
                for _ in range(cfg.n_layers)
            ]
        )

        # Layer normalization between layers
        self.norms = nn.ModuleList(
            [nn.LayerNorm(cfg.d_model) for _ in range(cfg.n_layers)]
        )

        # Output decoder
        self.decoder = nn.Sequential(
            nn.LayerNorm(cfg.d_model),
            nn.Linear(cfg.d_model, cfg.d_model // 2),
            nn.GELU(),
            nn.Linear(cfg.d_model // 2, n_state),
        )

    def forward(self, Y: Tensor, U: Optional[Tensor] = None) -> Tensor:
        """
        Apply Mamba layers for state estimation.

        Args:
            Y: (B, T, n_obs) observation sequence
            U: (B, T, n_ctrl) control sequence (unused)
        Returns:
            X_hat: (B, T, n_state) estimated states
        """
        B, T, _ = Y.shape

        # Embed observations
        x = self.obs_embed(Y)  # (B, T, d_model)

        # Apply Mamba layers with residual connections
        for layer, norm in zip(self.layers, self.norms):
            x = x + layer(norm(x))

        # Decode to states
        X_hat = self.decoder(x)  # (B, T, n_state)

        return X_hat


class Mamba2StateEstimator(nn.Module):
    """
    Mamba2-based state estimator for filtering tasks.

    Uses improved SSD formulation for better performance and efficiency.
    """

    def __init__(
        self,
        n_obs: int,
        n_state: int,
        n_ctrl: int = 0,
        cfg: Optional[Mamba2Config] = None,
    ):
        super().__init__()
        cfg = cfg or Mamba2Config()
        self.n_obs, self.n_state, self.n_ctrl = n_obs, n_state, n_ctrl
        self.cfg = cfg

        # Observation embedding
        self.obs_embed = nn.Linear(n_obs, cfg.d_model)

        # Stack of Mamba2 layers
        self.layers = nn.ModuleList(
            [
                Mamba2(
                    d_model=cfg.d_model,
                    d_state=cfg.d_state,
                    d_conv=cfg.d_conv,
                    expand=cfg.expand,
                    headdim=cfg.headdim,
                )
                for _ in range(cfg.n_layers)
            ]
        )

        # Layer normalization between layers
        self.norms = nn.ModuleList(
            [nn.LayerNorm(cfg.d_model) for _ in range(cfg.n_layers)]
        )

        # Output decoder
        self.decoder = nn.Sequential(
            nn.LayerNorm(cfg.d_model),
            nn.Linear(cfg.d_model, cfg.d_model // 2),
            nn.GELU(),
            nn.Linear(cfg.d_model // 2, n_state),
        )

    def forward(self, Y: Tensor, U: Optional[Tensor] = None) -> Tensor:
        """
        Apply Mamba2 layers for state estimation.

        Args:
            Y: (B, T, n_obs) observation sequence
            U: (B, T, n_ctrl) control sequence (unused)
        Returns:
            X_hat: (B, T, n_state) estimated states
        """
        B, T, _ = Y.shape

        # Embed observations
        x = self.obs_embed(Y)  # (B, T, d_model)

        # Apply Mamba2 layers with residual connections
        for layer, norm in zip(self.layers, self.norms):
            x = x + layer(norm(x))

        # Decode to states
        X_hat = self.decoder(x)  # (B, T, n_state)

        return X_hat
