"""
Filterformer for nonlinear state estimation (arXiv:2310.19603).

Architecture (per paper Section 3, Definition 5):
    F̂ = g-attn ∘ MLP ∘ attn

This is the paper's exact formulation: takes a complete observation path and
outputs a distribution. For discrete-time filtering, we apply this at each timestep
to the history y[0:t], producing a sequence of state estimates.

NOTE: The paper is designed for continuous-time paths and smoothing. Adapting it
to discrete-time causal filtering requires processing each timestep with its history.
"""

from __future__ import annotations

from typing import Optional

import torch
from torch import nn, Tensor
from config.model_config import FilterformerConfig


class SimilarityScore(nn.Module):
    """Similarity score (Definition 1): computes similarity to reference paths."""

    def __init__(self, n_ref_paths: int, n_sim: int, n_obs: int, horizon: int = 100):
        super().__init__()
        self.n_ref_paths = n_ref_paths
        self.n_sim = n_sim
        self.horizon = horizon

        # Learnable reference paths (paper mentions they can be from data or synthetic)
        self.ref_paths = nn.Parameter(torch.randn(n_ref_paths, horizon, n_obs) * 0.5)
        self.B = nn.Parameter(torch.randn(n_sim, n_ref_paths) * 0.1)
        self.A = nn.Parameter(torch.randn(n_sim, n_sim) * 0.1)
        self.b = nn.Parameter(torch.zeros(n_sim))
        self.a = nn.Parameter(torch.zeros(n_sim))

    def forward(self, y: Tensor) -> Tensor:
        """
        Compute similarity scores (Eq. 4): sim^{θ_0}_T(y) = Softmax(A·ReLU(B·||y-y^(n)||) + b) + a

        Args:
            y: (B, T, n_obs) observation path
        Returns:
            (B, n_sim) similarity scores
        """
        B, T, _ = y.shape

        # Resample or pad to fixed horizon for comparison
        if T != self.horizon:
            # Use last-value padding (more stable than interpolation for short sequences)
            if T < self.horizon:
                y_ext = torch.zeros(B, self.horizon, y.size(2), device=y.device)
                y_ext[:, :T, :] = y
                # Extend with last value
                y_ext[:, T:, :] = y[:, -1:, :]
            else:
                # Downsample by taking evenly spaced points
                indices = torch.linspace(0, T - 1, self.horizon, device=y.device).long()
                y_ext = y[:, indices, :]
        else:
            y_ext = y

        # Compute sup norm ||y - y^(n)|| for each reference path: max over time
        # (B, n_ref, T) -> (B, n_ref)
        dists = torch.norm(
            y_ext.unsqueeze(1) - self.ref_paths.unsqueeze(0), dim=-1
        ).max(dim=2)[0]

        # Apply transformations: ReLU(B·dists) -> A·... + b -> Softmax -> + a
        inner = torch.relu(self.B @ dists.T).T  # (B, n_sim)
        transformed = inner @ self.A.T + self.b  # (B, n_sim)
        scores = torch.softmax(transformed, dim=-1) + self.a  # (B, n_sim)

        return scores


class PositionalEncoding(nn.Module):
    """Positional encoding (Definition 2): samples path at query times."""

    def __init__(self, n_pos: int, n_time: int, n_obs: int):
        super().__init__()
        self.n_time = n_time
        # Learnable query times in [0, 1]
        self.times = nn.Parameter(torch.linspace(0, 1, n_time))
        self.U = nn.Parameter(torch.randn(n_pos, n_time) * 0.1)
        self.V = nn.Parameter(torch.randn(n_pos, n_obs) * 0.1)

    def forward(self, y: Tensor) -> Tensor:
        """
        Compute positional encoding (Eq. 5): pos^{θ_1}_T(y) = U·(⊕ y_{t_j}) + V

        Args:
            y: (B, T, n_obs) observation path
        Returns:
            (B, n_pos, n_obs) positional features
        """
        B, T, _ = y.shape

        # Sample at learned time indices
        indices = (self.times * (T - 1)).long().clamp(0, T - 1)
        y_sampled = y[:, indices, :]  # (B, n_time, n_obs)

        # U·y_sampled + V (broadcast)
        encoded = torch.einsum("pt,bto->bpo", self.U, y_sampled) + self.V.unsqueeze(0)

        return encoded


class PathwiseAttention(nn.Module):
    """Pathwise attention (Definition 3): combines similarity and positional encoding."""

    def __init__(
        self,
        n_ref_paths: int,
        n_sim: int,
        n_pos: int,
        n_time: int,
        n_obs: int,
        encoding_dim: int,
        horizon: int = 100,
    ):
        super().__init__()
        assert n_sim == n_pos, "Paper requires n_sim == n_pos"

        self.similarity = SimilarityScore(n_ref_paths, n_sim, n_obs, horizon)
        self.positional = PositionalEncoding(n_pos, n_time, n_obs)

        # Linear projection of flattened features + time
        self.C = nn.Linear(n_sim * n_obs + 1, encoding_dim)

    def forward(self, y: Tensor, t: Optional[float] = None) -> Tensor:
        """
        Compute pathwise attention (Eq. 6): attn^θ_T(t, y) = [t, C·vec(sim(y) ⊙ pos(y))]

        Args:
            y: (B, T, n_obs) observation path
            t: current time (if None, uses sequence length)
        Returns:
            (B, encoding_dim) encoded features
        """
        B = y.size(0)

        # Compute components
        sim_scores = self.similarity(y)  # (B, n_sim)
        pos_feats = self.positional(y)  # (B, n_pos, n_obs)

        # Element-wise product and flatten: sim ⊙ pos
        # (B, n_sim, 1) * (B, n_pos, n_obs) = (B, n_sim, n_obs)
        combined = sim_scores.unsqueeze(-1) * pos_feats  # (B, n_sim, n_obs)
        flattened = combined.flatten(1)  # (B, n_sim * n_obs)

        # Add time component
        if t is None:
            t = float(y.size(1))
        t_tensor = torch.full((B, 1), t, device=y.device, dtype=y.dtype)

        # Project through C
        features_with_time = torch.cat([t_tensor, flattened], dim=1)
        encoded = self.C(features_with_time)

        return encoded


class GeometricAttention(nn.Module):
    """Geometric attention (Definition 4): decodes to Gaussian mixture."""

    def __init__(self, input_dim: int, n_geometric: int, n_state: int):
        super().__init__()
        self.n_geometric = n_geometric
        self.n_state = n_state

        # Weight computation
        self.W = nn.Linear(input_dim, n_geometric)

        # Mixture components: means and covariance factors
        self.means = nn.Parameter(torch.randn(n_geometric, n_state))
        self.cov_factors = nn.Parameter(
            torch.randn(n_geometric, n_state, n_state) * 0.1
        )

    def forward(self, v: Tensor) -> tuple[Tensor, Tensor]:
        """
        Compute Gaussian mixture parameters via geometric attention.

        Paper formula: g-attn(v) = (Σ w_n·m^(n), Σ w_n·(A^(n))^T·A^(n))
        where w = P_Δ(W·v) (projection onto simplex, implemented via softmax)

        Args:
            v: (B, input_dim) features from MLP
        Returns:
            mean: (B, n_state) mixture mean
            cov: (B, n_state, n_state) mixture covariance
        """
        # Compute mixture weights via softmax (projects onto simplex)
        w = torch.softmax(self.W(v), dim=-1)  # (B, n_geometric)

        # Compute mixture mean: Σ w_n · m^(n)
        mean = w @ self.means  # (B, n_state)

        # Compute mixture covariance: Σ w_n · (A^(n))^T·A^(n)
        # First compute all A^T·A
        AtA = torch.bmm(
            self.cov_factors.transpose(1, 2), self.cov_factors
        )  # (n_geometric, n_state, n_state)
        # Weighted sum
        cov = torch.einsum("bn,nij->bij", w, AtA)  # (B, n_state, n_state)

        return mean, cov


class Filterformer(nn.Module):
    """
    Filterformer (Definition 5, Eq. 8): F̂ = g-attn ∘ fˆ ∘ attn

    Paper formulation: takes complete path C([0:T], R^{d_Y}) and outputs
    a Gaussian distribution N_{d_X}.

    For discrete-time filtering, we apply this architecture at each timestep t,
    processing y[0:t] to estimate x_t.
    """

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

        # 1. Pathwise Attention
        self.pathwise_attn = PathwiseAttention(
            cfg.n_ref_paths,
            cfg.n_sim,
            cfg.n_pos,
            cfg.n_time,
            n_obs,
            cfg.encoding_dim,
            cfg.horizon,
        )

        # 2. MLP - LayerNorm + GELU to avoid variance collapse from deep ReLU networks
        mlp_dim = cfg.encoding_dim * 2
        self.mlp = nn.Sequential(
            nn.LayerNorm(cfg.encoding_dim),
            nn.Linear(cfg.encoding_dim, mlp_dim),
            nn.GELU(),
            nn.Dropout(cfg.dropout) if cfg.dropout > 0 else nn.Identity(),
            nn.Linear(mlp_dim, cfg.encoding_dim),
        )

        # 3. Geometric Attention
        self.geometric_attn = GeometricAttention(
            cfg.encoding_dim, cfg.n_geometric, n_state
        )

    def forward(self, Y: Tensor, U: Optional[Tensor] = None) -> Tensor:
        """
        Apply Filterformer for discrete-time causal filtering.

        For each timestep t, apply: x_t = g-attn(MLP(attn(y[0:t])))

        Note: The paper's architecture is designed for complete fixed-length paths.
        Applying it to variable-length sequences y[0:t] causes pathwise attention
        collapse (padding/interpolation creates similar encodings). This is a
        fundamental architecture-task mismatch.

        Args:
            Y: (B, T, n_obs) observation sequence
            U: (B, T, n_ctrl) control sequence (unused, for API compatibility)
        Returns:
            X_hat: (B, T, n_state) estimated states
        """
        B, T, _ = Y.shape
        X_hat_list = []

        for t in range(T):
            # Extract history up to current time
            y_hist = Y[:, : t + 1, :]  # (B, t+1, n_obs)

            # Apply the three-stage pipeline: attn → MLP → g-attn
            encoded = self.pathwise_attn(y_hist, t=float(t + 1))
            features = self.mlp(encoded)
            mean, _ = self.geometric_attn(features)

            X_hat_list.append(mean)

        return torch.stack(X_hat_list, dim=1)


class FilterformerPractical(nn.Module):
    """
    Practical adaptation for discrete-time causal filtering.

    Replaces pathwise attention with causal self-attention.
    Keeps LayerNorm + GELU structure to avoid variance collapse.
    """

    def __init__(
        self,
        n_obs: int,
        n_state: int,
        n_ctrl: int = 0,
        cfg: Optional[FilterformerConfig] = None,
    ):
        super().__init__()
        cfg = cfg or FilterformerConfig()
        self.n_obs, self.n_state, self.n_ctrl = n_obs, n_state, n_ctrl
        hidden_dim = cfg.encoding_dim

        self.obs_embed = nn.Linear(n_obs, hidden_dim)
        self.temporal_attn = nn.MultiheadAttention(
            hidden_dim, num_heads=4, dropout=cfg.dropout, batch_first=True
        )

        mlp_dim = hidden_dim * 2
        self.mlp = nn.Sequential(
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, mlp_dim),
            nn.GELU(),
            nn.Dropout(cfg.dropout) if cfg.dropout > 0 else nn.Identity(),
            nn.Linear(mlp_dim, hidden_dim),
        )

        self.state_decoder = nn.Sequential(
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.GELU(),
            nn.Linear(hidden_dim // 2, n_state),
        )

    def forward(self, Y: Tensor, U: Optional[Tensor] = None) -> Tensor:
        """Causal filtering: obs → embed → self-attention → MLP → decode."""
        B, T, _ = Y.shape

        obs_encoded = self.obs_embed(Y)
        causal_mask = torch.triu(torch.ones(T, T, device=Y.device), diagonal=1).bool()

        attn_out, _ = self.temporal_attn(
            obs_encoded,
            obs_encoded,
            obs_encoded,
            attn_mask=causal_mask,
            need_weights=False,
        )

        features = self.mlp(attn_out.reshape(B * T, -1))
        X_hat = self.state_decoder(features).reshape(B, T, self.n_state)

        return X_hat
