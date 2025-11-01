"""
Simple Linear Dynamical System implementation.

Implements x_{t+1} = A x_t + B u_t + w,  y_t = H x_t + v
where w and v are Gaussian noise.
"""

from __future__ import annotations

from typing import Optional, Tuple

import torch
from torch import Tensor

from data.base_system import DynamicSystemDataset


class SimpleLinearSystemDataset(DynamicSystemDataset):
    """Linear dynamic system dataset with random systems per sample."""

    @staticmethod
    def _generate_random_systems(
        num_systems: int,
        n_state: int,
        n_obs: int,
        n_ctrl: int = 0,
        stable_radius: float = 0.95,
        device: Optional[torch.device] = None,
    ) -> Tuple[Tensor, Optional[Tensor], Tensor]:
        """Generate random system parameters.

        Args:
            num_systems: Number of systems to generate
            n_state: Dimension of state vector
            n_obs: Dimension of observation vector
            n_ctrl: Dimension of control input vector (0 if no control)
            stable_radius: Spectral radius for stability
            device: PyTorch device for computation

        Returns:
            A: (num_systems, n_state, n_state) A matrices
            B: (num_systems, n_state, n_ctrl) or None B matrices
            H: (num_systems, n_obs, n_state) H matrices
        """
        # Generate random A matrices
        A = torch.randn(num_systems, n_state, n_state, device=device)

        # Make each A stable by scaling spectral radius
        with torch.no_grad():
            for i in range(num_systems):
                u, s, v = torch.linalg.svd(A[i])
                s = s / s.max() * stable_radius
                A[i] = u @ torch.diag(s) @ v

        # Generate B matrices if needed
        B = None
        if n_ctrl > 0:
            B = torch.randn(num_systems, n_state, n_ctrl, device=device) * 0.5

        # Generate H matrices
        H = torch.randn(num_systems, n_obs, n_state, device=device)

        return A, B, H

    @staticmethod
    def _generate_trajectories(
        A: Tensor,
        B: Optional[Tensor],
        H: Tensor,
        horizon: int,
        process_noise_std: float,
        meas_noise_std: float,
        u_seq: Optional[Tensor] = None,
        x0: Optional[Tensor] = None,
        noisy: bool = True,
    ) -> Tuple[Tensor, Tensor, Optional[Tensor]]:
        """Generate trajectories from system parameters.

        Args:
            A: (B, n_state, n_state) A matrices
            B: (B, n_state, n_ctrl) or None B matrices
            H: (B, n_obs, n_state) H matrices
            horizon: Length of trajectories
            process_noise_std: Standard deviation of process noise
            meas_noise_std: Standard deviation of measurement noise
            u_seq: Optional control sequence (B, horizon, n_ctrl)
            x0: Optional initial state (B, n_state)
            noisy: Whether to add noise

        Returns:
            X: (B, horizon, n_state) true latent states
            Y: (B, horizon, n_obs) observations
            U: (B, horizon, n_ctrl) or None control inputs
        """
        device = A.device
        Bsz = A.size(0)
        n_state = A.size(1)
        n_obs = H.size(1)
        n_ctrl = B.size(2) if B is not None else 0

        X = torch.zeros(Bsz, horizon, n_state, device=device)
        Y = torch.zeros(Bsz, horizon, n_obs, device=device)
        U = torch.zeros(Bsz, horizon, n_ctrl, device=device) if n_ctrl > 0 else None

        if x0 is None:
            x = torch.randn(Bsz, n_state, device=device)
        else:
            x = x0.to(device)
            if x.dim() == 1:
                x = x.unsqueeze(0).expand(Bsz, -1)

        for t in range(horizon):
            # State update: x_i = x_i @ A_i^T + u_t @ B_i^T
            x = torch.einsum("bi,bij->bj", x, A)
            if B is not None:
                if u_seq is None:
                    u_t = torch.zeros(Bsz, n_ctrl, device=device)
                else:
                    u_t = u_seq[:, t]
                x = x + torch.einsum("bi,bij->bj", u_t, B)
                if U is not None:
                    U[:, t] = u_t

            if noisy and process_noise_std > 0:
                x = x + torch.randn_like(x) * process_noise_std

            # Observation: y_i = H_i @ x_i
            y = torch.einsum("bij,bj->bi", H, x)
            if noisy and meas_noise_std > 0:
                y = y + torch.randn_like(y) * meas_noise_std

            X[:, t] = x
            Y[:, t] = y

        return X, Y, U

    def __init__(
        self,
        n_state: int,
        n_obs: int,
        n_ctrl: int,
        num_distinct_systems: int,
        num_traj_per_system: int,
        horizon: int,
        process_noise_std: float,
        meas_noise_std: float,
        random_controls: bool = True,
        control_std: float = 0.0,
        noisy: bool = True,
        stable_radius: float = 0.95,
        device: Optional[torch.device] = None,
        seed: Optional[int] = None,
    ):
        # Initialize base class
        super().__init__(
            num_distinct_systems=num_distinct_systems,
            num_traj_per_system=num_traj_per_system,
            horizon=horizon,
            n_state=n_state,
            n_obs=n_obs,
            n_ctrl=n_ctrl,
            random_controls=random_controls,
            control_std=control_std,
            noisy=noisy,
            device=device,
            seed=seed,
            data_name="linear trajectories",
        )

        self.process_noise_std = process_noise_std
        self.meas_noise_std = meas_noise_std
        self.stable_radius = stable_radius

        # Generate all random system parameters at once
        A, B, H = self._generate_random_systems(
            num_systems=num_distinct_systems,
            n_state=n_state,
            n_obs=n_obs,
            n_ctrl=n_ctrl,
            stable_radius=stable_radius,
            device=self.device,
        )

        # Pre-generate controls if needed
        if n_ctrl > 0 and random_controls and control_std > 0:
            U_per_sys = (
                torch.randn(
                    num_distinct_systems,
                    num_traj_per_system,
                    horizon,
                    n_ctrl,
                    device=self.device,
                )
                * control_std
            )
        else:
            U_per_sys = None

        # Generate all trajectories at once
        # Repeat each system num_traj_per_system times
        A_repeated = A.repeat_interleave(num_traj_per_system, dim=0)
        B_repeated = (
            B.repeat_interleave(num_traj_per_system, dim=0) if B is not None else None
        )
        H_repeated = H.repeat_interleave(num_traj_per_system, dim=0)

        # Reshape U from (num_sys, num_traj_per_sys, horizon, n_ctrl) to (num_traj, horizon, n_ctrl)
        U_for_generate = None
        if U_per_sys is not None:
            U_for_generate = U_per_sys.reshape(self.num_traj, horizon, n_ctrl)

        X, Y, _ = self._generate_trajectories(
            A=A_repeated,
            B=B_repeated,
            H=H_repeated,
            horizon=horizon,
            process_noise_std=process_noise_std,
            meas_noise_std=meas_noise_std,
            u_seq=U_for_generate,
            noisy=noisy,
        )

        # Store all data on CPU
        self._store_data(X, Y, U_for_generate)
