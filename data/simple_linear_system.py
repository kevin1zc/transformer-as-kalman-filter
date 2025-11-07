"""
Linear Dynamical System Dataset.

Implements discrete-time linear state-space model:
    x_{t+1} = A·x_t + B·u_t + w_t
    y_t = H·x_t + v_t

where w_t and v_t are independent Gaussian noise processes.
Generates multiple trajectories from a single system instance with randomly
generated stable system matrices A, B, H.
"""

from __future__ import annotations

from typing import Optional, Tuple

import torch
from torch import Tensor

from data.base_system import DynamicSystemDataset, make_controls


class SimpleLinearSystemDataset(DynamicSystemDataset):
    """Linear state-space system dataset.

    Generates trajectories from a single linear system with randomly generated
    stable transition and observation matrices. All trajectories share the same
    system dynamics but have different noise realizations and initial conditions.

    Attributes:
        A: System transition matrix (n_state, n_state)
        H: Observation matrix (n_obs, n_state)
        B: Control matrix (n_state, n_ctrl) or None
        process_noise_std: Process noise standard deviation
        meas_noise_std: Measurement noise standard deviation
        stable_radius: Spectral radius bound for stability
    """

    @staticmethod
    def _generate_single_system(
        n_state: int,
        n_obs: int,
        n_ctrl: int = 0,
        stable_radius: float = 0.95,
        device: Optional[torch.device] = None,
    ) -> Tuple[Tensor, Optional[Tensor], Tensor]:
        """Generate random stable system matrices.

        Args:
            n_state: State dimension
            n_obs: Observation dimension
            n_ctrl: Control dimension (0 if uncontrolled)
            stable_radius: Spectral radius bound (< 1 for stability)
            device: PyTorch device

        Returns:
            Tuple of (A, B, H) where:
                A: (n_state, n_state) stable transition matrix
                B: (n_state, n_ctrl) control matrix or None
                H: (n_obs, n_state) observation matrix
        """
        # Generate A matrix with constrained spectral radius via SVD scaling
        A = torch.randn(n_state, n_state, device=device)
        with torch.no_grad():
            u, s, v = torch.linalg.svd(A)
            A = u @ torch.diag(s * stable_radius / s.max()) @ v

        # Generate B matrix if needed
        B = None
        if n_ctrl > 0:
            B = torch.randn(n_state, n_ctrl, device=device) * 0.5

        # Generate H matrix
        H = torch.randn(n_obs, n_state, device=device)

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
        noisy: bool = True,
    ) -> Tuple[Tensor, Tensor, Optional[Tensor]]:
        """Generate multiple trajectories from a single linear system.

        Args:
            A: (n_state, n_state) transition matrix
            B: (n_state, n_ctrl) control matrix or None
            H: (n_obs, n_state) observation matrix
            horizon: Number of time steps per trajectory
            process_noise_std: Process noise standard deviation
            meas_noise_std: Measurement noise standard deviation
            u_seq: Control sequences (num_traj, horizon, n_ctrl) or None
            noisy: Whether to add Gaussian noise

        Returns:
            Tuple of (X, Y, U) where:
                X: (num_traj, horizon, n_state) true states
                Y: (num_traj, horizon, n_obs) noisy observations
                U: (num_traj, horizon, n_ctrl) controls or None
        """
        device = A.device
        Bsz = u_seq.size(0) if u_seq is not None else 1
        n_state, n_obs, n_ctrl = (
            A.size(0),
            H.size(0),
            (B.size(1) if B is not None else 0),
        )

        X = torch.zeros(Bsz, horizon, n_state, device=device)
        Y = torch.zeros(Bsz, horizon, n_obs, device=device)
        U = torch.zeros(Bsz, horizon, n_ctrl, device=device) if n_ctrl > 0 else None

        x = torch.randn(Bsz, n_state, device=device)

        for t in range(horizon):
            # State update: x = x @ A + u @ B
            x = x @ A.T
            if B is not None and n_ctrl > 0:
                u_t = u_seq[:, t]
                x = x + u_t @ B.T
                if U is not None:
                    U[:, t] = u_t

            if noisy and process_noise_std > 0:
                x = x + torch.randn_like(x) * process_noise_std

            # Observation: y = H @ x
            y = x @ H.T
            if noisy and meas_noise_std > 0:
                y = y + torch.randn_like(y) * meas_noise_std

            X[:, t], Y[:, t] = x, y

        return X, Y, U

    def __init__(
        self,
        n_state: int,
        n_obs: int,
        n_ctrl: int,
        num_traj: int,
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
        super().__init__(
            num_traj,
            horizon,
            n_state,
            n_obs,
            n_ctrl,
            random_controls,
            control_std,
            noisy,
            device,
            seed,
            data_name="linear trajectories",
        )
        self.process_noise_std, self.meas_noise_std, self.stable_radius = (
            process_noise_std,
            meas_noise_std,
            stable_radius,
        )

        A, B, H = self._generate_single_system(
            n_state, n_obs, n_ctrl, stable_radius, self.device
        )

        # Generate controls (or dummy for batching)
        u_seq = make_controls(
            num_traj,
            horizon,
            n_ctrl,
            random_controls,
            control_std,
            self.device,
        )

        X, Y, U_actual = self._generate_trajectories(
            A, B, H, horizon, process_noise_std, meas_noise_std, u_seq, noisy=noisy
        )

        U_for_store = u_seq if n_ctrl > 0 else None
        self._store_data(X, Y, U_for_store)
        self.A, self.H, self.B = A.cpu(), H.cpu(), (B.cpu() if B is not None else None)
