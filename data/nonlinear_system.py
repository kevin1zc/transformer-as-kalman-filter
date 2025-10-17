"""
Nonlinear system generator and dataset for testing Filterformer.

We implement a discrete-time nonlinear state space model with optional
controls:
    x_{t+1} = f(x_t, u_t) + w_t
    y_t     = g(x_t) + v_t

Example choices (chaotic-ish but stable with noise):
  - f: smooth nonlinearity combining tanh and quadratic mixing
  - g: nonlinear observation (e.g., sin/cos on subsets of the state)

Noise w_t and v_t are Gaussian with user-specified standard deviations.
"""

from __future__ import annotations

from typing import Optional, Tuple

import torch
from torch import Tensor
from torch.utils.data import Dataset


def get_device(prefer: str = "cuda") -> torch.device:
    if prefer == "cuda" and torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


@torch.no_grad()
def f_nonlinear(x: Tensor, u: Optional[Tensor]) -> Tensor:
    """Nonlinear transition function f(x,u) -> x_next.

    x: (B, n_x)
    u: (B, n_u) or None
    """
    # Smooth nonlinearity with cross terms; coefficients chosen to be
    # stable-ish
    z = torch.tanh(x)
    # Bounded quadratic-like term to avoid explosion
    quad = 0.05 * torch.tanh(x * x)
    mix = 0.01 * (
        x @ torch.full((x.size(-1), x.size(-1)), 0.02, device=x.device)
    )
    x_next = 0.85 * x + z + quad + mix
    if u is not None and u.numel() > 0:
        Bu = 0.2 * u  # simple linear control influence
        # pad/control to state dimension if needed
        if Bu.size(-1) < x_next.size(-1):
            pad = torch.zeros(
                Bu.size(0), x_next.size(-1) - Bu.size(-1), device=x.device
            )
            Bu = torch.cat([Bu, pad], dim=-1)
        elif Bu.size(-1) > x_next.size(-1):
            Bu = Bu[..., : x_next.size(-1)]
        x_next = x_next + Bu
    return x_next


@torch.no_grad()
def g_nonlinear(x: Tensor, n_obs: int) -> Tensor:
    """Nonlinear observation function g(x) -> y.

    We map state to observations using sin/cos and linear heads.
    """
    s = torch.sin(x)
    c = torch.cos(x)
    feats = torch.cat([x, s, c], dim=-1)
    # Random but fixed projection per call: use a simple deterministic map
    # for reproducibility. Use the first n_obs columns of a normalized matrix.
    # Stateless function: build a fixed tensor on the same device.
    in_dim = feats.size(-1)
    # Use a Hadamard-like random map seeded by device; fallback to
    # normalized ones
    W = torch.arange(
        in_dim * n_obs, device=x.device, dtype=feats.dtype
    ).reshape(in_dim, n_obs)
    W = (W % 7 - 3).float()  # small integers in [-3,3]
    W = W / (W.norm(dim=0, keepdim=True) + 1e-6)
    y = feats @ W
    return y


class NonlinearSystem:
    def __init__(
        self,
        n_state: int,
        n_obs: int,
        n_ctrl: int = 0,
        process_noise_std: float = 0.05,
        meas_noise_std: float = 0.05,
        device: Optional[torch.device] = None,
    ):
        self.n_state = n_state
        self.n_obs = n_obs
        self.n_ctrl = n_ctrl
        self.q = process_noise_std
        self.r = meas_noise_std
        self.device = device or get_device()

    @torch.no_grad()
    def generate(
        self,
        horizon: int,
        batch_size: int,
        *,
        noisy: bool = True,
        u_seq: Optional[Tensor] = None,
        x0: Optional[Tensor] = None,
    ) -> Tuple[Tensor, Tensor, Optional[Tensor]]:
        B = batch_size
        n_x = self.n_state
        n_u = self.n_ctrl
        X = torch.zeros(B, horizon, n_x, device=self.device)
        Y = torch.zeros(B, horizon, self.n_obs, device=self.device)
        U = (
            torch.zeros(B, horizon, n_u, device=self.device)
            if n_u > 0
            else None
        )

        if x0 is None:
            x = torch.randn(B, n_x, device=self.device)
        else:
            x = x0.to(self.device)
            if x.dim() == 1:
                x = x.unsqueeze(0).expand(B, -1)

        for t in range(horizon):
            if U is not None:
                if u_seq is None:
                    u_t = torch.zeros(B, n_u, device=self.device)
                else:
                    u_t = u_seq[:, t]
                U[:, t] = u_t
            else:
                u_t = None

            x = f_nonlinear(x, u_t)
            if noisy and self.q > 0:
                x = x + torch.randn_like(x) * self.q
            y = g_nonlinear(x, self.n_obs)
            if noisy and self.r > 0:
                y = y + torch.randn_like(y) * self.r

            X[:, t] = x
            Y[:, t] = y

        return X, Y, U


class NonlinearSequenceDataset(Dataset):
    def __init__(
        self,
        sys: NonlinearSystem,
        num_traj: int,
        horizon: int,
        random_controls: bool = True,
        control_std: float = 0.0,
        noisy: bool = True,
    ):
        super().__init__()
        self.sys = sys
        self.num_traj = num_traj
        self.horizon = horizon
        self.random_controls = random_controls
        self.control_std = control_std
        self.noisy = noisy
        if sys.n_ctrl > 0 and random_controls and control_std > 0:
            self.Uall = (
                torch.randn(num_traj, horizon, sys.n_ctrl, device=sys.device)
                * control_std
            )
        else:
            self.Uall = None

    def __len__(self) -> int:
        return self.num_traj

    def __getitem__(self, idx: int):
        u_seq = None
        if self.sys.n_ctrl > 0 and self.Uall is not None:
            u_seq = self.Uall[idx: idx + 1].clone()
        X, Y, U = self.sys.generate(
            self.horizon,
            batch_size=1,
            noisy=self.noisy,
            u_seq=u_seq,
        )
        X, Y = X.squeeze(0), Y.squeeze(0)
        if U is not None:
            U = U.squeeze(0)
        result = {"x": X.cpu(), "y": Y.cpu()}
        if U is not None:
            result["u"] = U.cpu()
        return result
