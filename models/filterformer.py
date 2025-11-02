"""
Filterformer model for state estimation.

This module implements the Filterformer architecture from the paper:
"Transformers Can Solve Non-Linear and Non-Markovian Filtering Problems
in Continuous Time For Conditionally Gaussian Signals"
(arXiv:2310.19603)

The Filterformer consists of three phases:
1. Pathwise Attention: Encodes continuous-time paths into finite-dimensional vectors
2. MLP Transformation: Processes the encoded features
3. Geometric Attention: Decodes to Gaussian distributions in Wasserstein space

Reference: Section 3 of the paper
"""

from __future__ import annotations

from typing import Optional

import torch
from torch import nn, Tensor
from config.model_config import FilterformerConfig


class SimilarityScore(nn.Module):
    """Similarity score: maps path to similarities with reference paths."""

    def __init__(self, n_ref_paths: int, n_sim: int, n_obs: int):
        super().__init__()
        self.register_buffer("ref_paths", torch.randn(n_ref_paths, 100, n_obs) * 0.1)
        self.A = nn.Parameter(torch.randn(n_sim, n_sim) * 0.1)
        self.B = nn.Parameter(torch.randn(n_sim, n_ref_paths) * 0.1)
        self.a = nn.Parameter(torch.zeros(n_sim))
        self.b = nn.Parameter(torch.zeros(n_sim))
        self.register_buffer("query_times", torch.linspace(0, 1, 100))

    def forward(self, y: Tensor) -> Tensor:
        """Compute similarity scores between y and reference paths."""
        B, T, _ = y.shape
        if T != 100:
            y = nn.functional.interpolate(
                y.transpose(1, 2), size=100, mode="linear", align_corners=True
            ).transpose(1, 2)
        # Compute max distance efficiently: (B, n_ref, T) -> (B, n_ref)
        dists = torch.norm(y.unsqueeze(1) - self.ref_paths.unsqueeze(0), dim=-1).max(
            dim=2
        )[0]
        # Vectorized computation: (B, n_ref) -> (B, n_sim)
        inner = (
            nn.functional.relu(torch.einsum("ij,bj->bi", self.B, dists)) @ self.A.t()
            + self.b
        )
        return nn.functional.softmax(inner, dim=-1) + self.a.unsqueeze(0)


class PositionalEncoding(nn.Module):
    """Positional encoding: extracts path snapshots at query times."""

    def __init__(self, n_pos: int, n_time: int, n_obs: int):
        super().__init__()
        self.times = nn.Parameter(torch.linspace(0, 1, n_time))
        self.U = nn.Parameter(torch.randn(n_pos, n_time) * 0.1)
        self.V = nn.Parameter(torch.randn(n_pos, n_obs) * 0.1)

    def forward(self, y: Tensor) -> Tensor:
        """Compute positional encoding from path snapshots."""
        times_norm = (self.times * (y.shape[1] - 1)).long().clamp(0, y.shape[1] - 1)
        y_sampled = y[:, times_norm, :]  # (B, n_time, n_obs)
        return torch.einsum("nt,bto->bno", self.U, y_sampled) + self.V.unsqueeze(0)


class PathwiseAttention(nn.Module):
    """Pathwise attention: combines similarity scores and positional encoding."""

    def __init__(
        self,
        n_ref_paths: int,
        n_sim: int,
        n_pos: int,
        n_time: int,
        n_obs: int,
        encoding_dim: int,
    ):
        super().__init__()
        assert n_sim == n_pos
        self.similarity = SimilarityScore(n_ref_paths, n_sim, n_obs)
        self.positional = PositionalEncoding(n_pos, n_time, n_obs)
        self.C = nn.Linear(n_sim * n_obs + 1, encoding_dim)

    def forward(self, y: Tensor, t: Optional[Tensor] = None) -> Tensor:
        """Encode path through similarity and positional features."""
        feat = (self.similarity(y).unsqueeze(-1) * self.positional(y)).flatten(1)
        t_val = (
            t
            if t is not None
            else torch.ones(y.shape[0], 1, device=y.device) * y.size(1)
        )
        return self.C(torch.cat([t_val, feat], dim=1))


class GeometricAttention(nn.Module):
    """Geometric attention: decodes to Gaussian distributions in Wasserstein space."""

    def __init__(self, input_dim: int, n_geometric: int, n_state: int):
        super().__init__()
        self.W = nn.Linear(input_dim, n_geometric)
        self.means = nn.Parameter(torch.randn(n_geometric, n_state) * 0.1)
        self.cov_factors = nn.Parameter(
            torch.randn(n_geometric, n_state, n_state) * 0.1
        )

    def forward(self, v: Tensor) -> tuple[Tensor, Tensor]:
        """Compute mixture Gaussian parameters from features."""
        w = nn.functional.softmax(self.W(v), dim=-1)
        A_safe = (
            self.cov_factors
            + torch.eye(self.means.shape[1], device=v.device).unsqueeze(0) * 0.01
        )
        # Precompute all A^T @ A matrices: (n_geometric, n_state, n_state)
        AtA = torch.bmm(
            A_safe.transpose(1, 2), A_safe
        )  # (n_geometric, n_state, n_state)
        # Weighted sum: (B, n_geometric) -> (B, n_state, n_state)
        cov = torch.einsum("bi,ijk->bjk", w, AtA)
        return w @ self.means, cov


class Filterformer(nn.Module):
    """Filterformer: F̂ = g-attn ∘ MLP ∘ attn (arXiv:2310.19603)."""

    def __init__(
        self,
        n_obs: int,
        n_state: int,
        n_ctrl: int = 0,
        cfg: Optional[FilterformerConfig] = None,
    ):
        super().__init__()
        cfg = cfg or FilterformerConfig()
        self.n_obs, self.n_state, self.n_ctrl, self.cfg = n_obs, n_state, n_ctrl, cfg

        self.pathwise_attn = PathwiseAttention(
            cfg.n_ref_paths, cfg.n_sim, cfg.n_pos, cfg.n_time, n_obs, cfg.encoding_dim
        )

        dims = [cfg.encoding_dim] + cfg.mlp_hidden_dims
        mlp_layers = [nn.Linear(dims[i], dims[i + 1]) for i in range(len(dims) - 1)]
        mlp_layers = [
            x for pair in zip(mlp_layers, [nn.ReLU()] * len(mlp_layers)) for x in pair
        ]
        if cfg.dropout > 0:
            mlp_layers.extend([nn.Dropout(cfg.dropout)] * (len(mlp_layers) // 2))
        self.mlp = nn.Sequential(*mlp_layers)

        mlp_output_dim = (
            cfg.mlp_hidden_dims[-1] if cfg.mlp_hidden_dims else cfg.encoding_dim
        )
        self.geometric_attn = GeometricAttention(
            mlp_output_dim, cfg.n_geometric, n_state
        )

    def forward(self, Y: Tensor, U: Optional[Tensor] = None) -> Tensor:
        """Estimate states from observation sequences."""
        X_hat_list = []
        for t in range(Y.shape[1]):
            encoded = self.pathwise_attn(Y[:, : t + 1, :])
            features = self.mlp(encoded)
            mean, _ = self.geometric_attn(features)
            X_hat_list.append(mean)
        return torch.stack(X_hat_list, dim=1)
