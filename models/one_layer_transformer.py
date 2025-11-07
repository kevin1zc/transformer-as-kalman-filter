"""
One-Layer Transformer for state estimation (arXiv:2312.06937).

Trainable causal self-attention transformer that learns filtering from data.
"""

from __future__ import annotations

from typing import Optional, Tuple

import torch
from torch import nn, Tensor
from config.model_config import OneLayerTransformerConfig


class OneLayerTransformer(nn.Module):
    """Causal self-attention transformer for state estimation."""

    def __init__(
        self,
        n_obs: int,
        n_state: int,
        n_ctrl: int = 0,
        cfg: Optional[OneLayerTransformerConfig] = None,
    ):
        super().__init__()
        cfg = cfg or OneLayerTransformerConfig()
        hidden_dim = max(128, 2 * n_state)

        self.obs_embed = nn.Linear(n_obs, hidden_dim)
        self.attn_query = nn.Linear(hidden_dim, hidden_dim)
        self.attn_key = nn.Linear(hidden_dim, hidden_dim)
        self.attn_value = nn.Linear(hidden_dim, hidden_dim)
        self.scale = hidden_dim**-0.5
        self.state_proj = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim), nn.ReLU(), nn.Linear(hidden_dim, n_state)
        )

    def forward(self, y: Tensor, u: Optional[Tensor] = None) -> Tensor:
        """Forward: y (B,T,n_obs) -> x_hat (B,T,n_state)."""
        B, T, _ = y.shape
        h = self.obs_embed(y)

        # Causal self-attention
        Q, K, V = self.attn_query(h), self.attn_key(h), self.attn_value(h)
        scores = torch.bmm(Q, K.transpose(1, 2)) * self.scale
        mask = torch.triu(torch.ones(T, T, device=y.device), diagonal=1).bool()
        scores = scores.masked_fill(mask.unsqueeze(0), float("-inf"))
        attn_weights = torch.softmax(scores, dim=-1)
        h_attn = torch.bmm(attn_weights, V)

        return self.state_proj(h_attn)


class KalmanFilterExact(nn.Module):
    """Oracle Kalman Filter baseline (NOT a transformer)."""

    def __init__(
        self,
        n_obs: int,
        n_state: int,
        A: Tensor,
        H: Tensor,
        Q: Tensor,
        R: Tensor,
        use_time_varying: bool = False,
        horizon: int = 100,
    ):
        super().__init__()
        self.n_state = n_state
        self.register_buffer("A", A)
        self.register_buffer("H", H)

        if use_time_varying:
            K_seq = self._compute_time_varying_gains(A, H, Q, R, horizon)
            self.register_buffer("K_seq", K_seq)
        else:
            K, _ = self._compute_steady_state_gain(A, H, Q, R)
            self.register_buffer("K", K)

    @staticmethod
    def _compute_steady_state_gain(
        A: Tensor,
        H: Tensor,
        Q: Tensor,
        R: Tensor,
        tol: float = 1e-9,
        max_iter: int = 20000,
    ) -> Tuple[Tensor, Tensor]:
        """Riccati iteration for steady-state K."""
        device, n = A.device, A.size(0)
        I, P = torch.eye(n, device=device), torch.eye(n, device=device)
        for _ in range(max_iter):
            P_prev = P
            Pp = A @ P @ A.T + Q
            S = H @ Pp @ H.T + R
            K = torch.linalg.solve(S, (Pp @ H.T).T).T
            P = (I - K @ H) @ Pp
            if torch.norm(P - P_prev) < tol:
                break
        return K, P

    @staticmethod
    def _compute_time_varying_gains(
        A: Tensor, H: Tensor, Q: Tensor, R: Tensor, horizon: int
    ) -> Tensor:
        """Riccati recursion for time-varying K_t."""
        device, n = A.device, A.size(0)
        I, P = torch.eye(n, device=device), torch.eye(n, device=device)
        K_list = []
        for _ in range(horizon):
            Pp = A @ P @ A.T + Q
            S = H @ Pp @ H.T + R
            Kt = torch.linalg.solve(S, (Pp @ H.T).T).T
            K_list.append(Kt)
            P = (I - Kt @ H) @ Pp
        return torch.stack(K_list, dim=0)

    def forward(self, y: Tensor, u: Optional[Tensor] = None) -> Tensor:
        """Direct KF recursion: x̂_t = (I - K H) A x̂_{t-1} + K y_t."""
        B, T, _ = y.shape
        device = y.device
        I = torch.eye(self.n_state, device=device)

        if hasattr(self, "K_seq"):
            # Time-varying
            x = (self.K_seq[0] @ y[:, 0:1].transpose(-1, -2)).transpose(-1, -2)
            x_hats = [x]
            for t in range(1, T):
                A_eff = (I - self.K_seq[t] @ self.H) @ self.A
                x = x @ A_eff.T + y[:, t : t + 1] @ self.K_seq[t].T
                x_hats.append(x)
            return torch.cat(x_hats, dim=1)
        else:
            # Steady-state
            A_eff = (I - self.K @ self.H) @ self.A
            x = (self.K @ y[:, 0:1].transpose(-1, -2)).transpose(-1, -2)
            x_hats = [x]
            for t in range(1, T):
                x = x @ A_eff.T + y[:, t : t + 1] @ self.K.T
                x_hats.append(x)
            return torch.cat(x_hats, dim=1)
