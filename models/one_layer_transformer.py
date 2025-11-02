"""
One-Layer Transformer implementation from arXiv:2312.06937

This module implements the exact Transformer Filter construction from the paper
"Can a Transformer Represent a Kalman Filter?" The architecture uses quadratic
embeddings φ(x, y) and represents Nadaraya-Watson kernel smoothing via softmax
self-attention to approximate the Kalman Filter.

Reference: Section 3-4 of arXiv:2312.06937
"""

from __future__ import annotations

from typing import Optional

import torch
from torch import nn, Tensor
from config.model_config import OneLayerTransformerConfig


class QuadraticEmbedding(nn.Module):
    """Quadratic embedding φ: R^n → R^ℓ from Theorem 1."""

    def __init__(self, d: int):
        super().__init__()
        self.embed_dim = 1 + d + (d * (d + 1) // 2)

    def forward(self, u: Tensor) -> Tensor:
        """Embed input u: φ(u) = [1, u_1, ..., u_d, u_i*u_j for i<=j]."""
        orig, u_flat = u.shape, u.reshape(-1, u.shape[-1])
        d = u_flat.shape[1]
        lin = torch.cat(
            [torch.ones(u_flat.shape[0], 1, device=u.device), u_flat], dim=1
        )

        # Vectorized quadratic terms: compute all i*j products at once
        quad = torch.empty(u_flat.shape[0], 0, device=u.device)
        if d > 0:
            # Broadcast: (B, d, 1) * (B, 1, d) -> (B, d, d)
            outer = u_flat.unsqueeze(-1) * u_flat.unsqueeze(-2)
            # Extract upper triangular part (including diagonal) in row-major order
            triu_indices = torch.triu_indices(d, d, device=u.device)
            quad = outer[:, triu_indices[0], triu_indices[1]]

        return torch.cat([lin, quad], dim=1).reshape(*orig[:-1], self.embed_dim)


class OneLayerTransformer(nn.Module):
    """One-Layer Transformer from arXiv:2312.06937: Nadaraya-Watson kernel smoothing via softmax attention."""

    def __init__(
        self,
        n_obs: int,
        n_state: int,
        n_ctrl: int = 0,
        cfg: Optional[OneLayerTransformerConfig] = None,
        L: Optional[Tensor] = None,
        A: Optional[Tensor] = None,
        C: Optional[Tensor] = None,
    ):
        super().__init__()
        cfg = cfg or OneLayerTransformerConfig()
        self.H, self.beta, self.n_obs, self.n_state, self.n_ctrl, self.cfg = (
            cfg.horizon,
            cfg.beta,
            n_obs,
            n_state,
            n_ctrl,
            cfg,
        )

        # Kalman filter matrices
        for name, val, def_shape in [
            ("L", L, (n_state, n_obs)),
            ("A", A, (n_state, n_state)),
            ("C", C, (n_obs, n_state)),
        ]:
            if val is not None:
                self.register_buffer(name, val)
            else:
                setattr(self, name, nn.Parameter(torch.randn(*def_shape) * 0.1))

        self.embedding = QuadraticEmbedding(n_state + n_obs)
        embed_dim = self.embedding.embed_dim
        self.A_attn = nn.Parameter(torch.eye(embed_dim) * self.beta)

    def kalman_step(self, x_prev: Tensor, y_prev: Tensor) -> Tensor:
        """One-step Kalman update: (A - LC)x_{t-1} + Ly_{t-1}."""
        return (self.A - self.L @ self.C) @ x_prev.transpose(
            -1, -2
        ) + self.L @ y_prev.transpose(-1, -2)

    def forward(self, y: Tensor, u: Optional[Tensor] = None) -> Tensor:
        """Forward pass: Nadaraya-Watson kernel smoothing with Kalman updates."""
        B, T = y.shape[:2]
        device = y.device

        # Pre-compute Kalman matrix for efficiency
        A_LC = self.A - self.L @ self.C
        L_C = self.L @ self.C

        # Pre-allocate attention computation buffers
        A_attn_T = self.A_attn.T

        # Initialize with first observation
        x_hat_list = [(self.L @ y[:, 0].transpose(-1, -2)).transpose(-1, -2).squeeze(1)]

        for t in range(1, T):
            hist_len = min(t, self.H)
            start_idx = max(0, t - hist_len)

            # Get past states from list (stack into tensor)
            x_past = torch.stack(
                x_hat_list[-hist_len:], dim=1
            )  # (B, hist_len, n_state)
            y_past = y[:, start_idx:t]  # (B, hist_len, n_obs)
            y_curr = y[:, t : t + 1]  # (B, 1, n_obs)

            # Vectorized Kalman updates for all past states
            x_updates = torch.einsum("ij,bkj->bki", A_LC, x_past) + torch.einsum(
                "ij,bkj->bki", L_C, y_past
            )

            # Attention via quadratic embeddings
            xy_past = torch.cat([x_past, y_past], dim=-1)
            xy_query = torch.cat([x_hat_list[-1].unsqueeze(1), y_curr], dim=-1)

            q_past = self.embedding(xy_past)
            q_query = self.embedding(xy_query)

            # Attention scores
            attn_scores = torch.bmm(q_query, (q_past @ A_attn_T).transpose(1, 2))
            attn_weights = nn.functional.softmax(attn_scores, dim=-1)

            # Weighted sum of Kalman updates
            x_hat_list.append(torch.bmm(attn_weights, x_updates).squeeze(1))

        return torch.stack(x_hat_list, dim=1)
