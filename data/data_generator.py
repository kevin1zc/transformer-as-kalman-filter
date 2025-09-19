"""
Data generation utilities for Linear Dynamical System (LDS).
"""

from __future__ import annotations

from typing import Optional, Tuple

import torch
from torch import Tensor
from torch.utils.data import Dataset

from config import LDSConfig


def get_device(prefer: str = "cuda") -> torch.device:
    """Get the best available device."""
    if prefer == "cuda" and torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def set_seed(seed: int = 1337) -> None:
    """Set random seeds for reproducibility."""
    import random
    import numpy as np

    random.seed(seed)
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


class LDS:
    """Linear Dynamical System x_{t+1} = A x_t + B u_t + w,  y_t = H x_t + v.

    Tensors live on `device`. Provides both full-trajectory generation and
    step-by-step API.
    """

    def __init__(
        self,
        A: Tensor,
        B: Optional[Tensor],
        H: Tensor,
        process_noise_std: float,
        meas_noise_std: float,
        device: Optional[torch.device] = None,
    ):
        device = device or get_device()
        self.A = A.to(device)
        self.B = B.to(device) if B is not None else None
        self.H = H.to(device)
        self.q = process_noise_std
        self.r = meas_noise_std
        self.device = device
        self._x_cur: Optional[Tensor] = None  # for step API

    @staticmethod
    def random(cfg: LDSConfig, *, seed: int = 0) -> "LDS":
        """Create a random stable LDS."""
        set_seed(seed)
        device = cfg.device or get_device()
        # Random stable A by scaling spectral radius
        A = torch.randn(cfg.n_state, cfg.n_state, device=device)
        with torch.no_grad():
            u, s, v = torch.linalg.svd(A)
            s = s / s.max() * cfg.stable_radius
            A = u @ torch.diag(s) @ v
        B = None
        if cfg.n_ctrl > 0:
            B = torch.randn(cfg.n_state, cfg.n_ctrl, device=device) * 0.5
        H = torch.randn(cfg.n_obs, cfg.n_state, device=device)
        return LDS(A, B, H, cfg.process_noise_std, cfg.meas_noise_std, device)

    # ---------------- Step-by-step API ----------------
    def reset(self, x0: Optional[Tensor] = None) -> Tensor:
        """Reset the system to initial state."""
        if x0 is None:
            x0 = torch.randn(self.A.size(0), device=self.device)
        self._x_cur = x0
        return x0

    @torch.no_grad()
    def step(
        self, u: Optional[Tensor] = None, *, noisy: bool = True
    ) -> Tuple[Tensor, Tensor, Tensor]:
        """Take one step of the LDS."""
        assert self._x_cur is not None, "Call reset() before step()"
        x = self._x_cur
        if self.B is not None:
            if u is None:
                u = torch.zeros(self.B.size(1), device=self.device)
            x_next = self.A @ x + self.B @ u
        else:
            x_next = self.A @ x
        if noisy and self.q > 0:
            x_next = x_next + torch.randn_like(x_next) * self.q
        y = self.H @ x_next
        if noisy and self.r > 0:
            y = y + torch.randn(self.H.size(0), device=self.device) * self.r
        self._x_cur = x_next
        return x_next, y, (u if u is not None else torch.empty(0, device=self.device))

    # ---------------- Whole-trajectory generation ----------------
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
        """Generate full trajectories.

        Returns
            X:  (B, T, n_state)   true latent states (on self.device)
            Y:  (B, T, n_obs)     observations (on self.device)
            U:  (B, T, n_ctrl) or None (on self.device)
        """
        Bsz = batch_size
        n_x = self.A.size(0)
        n_y = self.H.size(0)
        n_u = self.B.size(1) if self.B is not None else 0

        X = torch.zeros(Bsz, horizon, n_x, device=self.device)
        Y = torch.zeros(Bsz, horizon, n_y, device=self.device)
        U = torch.zeros(Bsz, horizon, n_u, device=self.device) if n_u > 0 else None

        if x0 is None:
            x = torch.randn(Bsz, n_x, device=self.device)
        else:
            x = x0.to(self.device)
            if x.dim() == 1:
                x = x.unsqueeze(0).expand(Bsz, -1)

        for t in range(horizon):
            if self.B is not None:
                if u_seq is None:
                    u_t = torch.zeros(Bsz, n_u, device=self.device)
                else:
                    u_t = u_seq[:, t]
                x = x @ self.A.T + u_t @ self.B.T
            else:
                x = x @ self.A.T
            if noisy and self.q > 0:
                x = x + torch.randn_like(x) * self.q
            y = x @ self.H.T
            if noisy and self.r > 0:
                y = y + torch.randn_like(y) * self.r
            X[:, t] = x
            Y[:, t] = y
            if U is not None:
                U[:, t] = u_t
        return X, Y, U


class LDSSequenceDataset(Dataset):
    """Dataset for training many trajectories (CPU outputs for pin_memory=True)."""

    def __init__(
        self,
        lds: LDS,
        num_traj: int,
        horizon: int,
        random_controls: bool = True,
        control_std: float = 0.0,
        noisy: bool = True,
    ):
        super().__init__()
        self.lds = lds
        self.num_traj = num_traj
        self.horizon = horizon
        self.random_controls = random_controls
        self.control_std = control_std
        self.noisy = noisy
        # Pre-generate controls on LDS device for speed; will .cpu() on output
        if lds.B is not None and random_controls and control_std > 0:
            self.Uall = (
                torch.randn(num_traj, horizon, lds.B.size(1), device=lds.device)
                * control_std
            )
        else:
            self.Uall = None

    def __len__(self) -> int:
        return self.num_traj

    def __getitem__(self, idx: int):
        u_seq = None
        if self.lds.B is not None and self.Uall is not None:
            u_seq = self.Uall[idx : idx + 1].clone()  # (1, T, n_u) on lds.device
        X, Y, U = self.lds.generate(
            self.horizon, batch_size=1, noisy=self.noisy, u_seq=u_seq
        )
        X, Y = X.squeeze(0), Y.squeeze(0)
        if U is not None:
            U = U.squeeze(0)
        # Return CPU tensors so DataLoader with pin_memory=True can pin
        result = {
            "x": X.cpu(),
            "y": Y.cpu(),
        }
        if U is not None:
            result["u"] = U.cpu()
        return result
