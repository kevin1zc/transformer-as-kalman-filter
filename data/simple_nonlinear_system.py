"""
Nonlinear Dynamical System Dataset.

Implements discrete-time nonlinear state-space model:
    x_{t+1} = f(x_t, u_t) + w_t
    y_t = g(x_t) + v_t

where f and g are randomly generated nonlinear functions composed of:
  - Transition f: linear, tanh, sin, cos, quadratic, cubic, and mixed terms
  - Observation g: linear, sin, cos, tanh, and squared transformations

Each system uses different randomly sampled weights for nonlinear components,
ensuring diverse dynamics. Noise w_t and v_t are independent Gaussian processes.
Generates multiple trajectories from a single system instance with different
initial conditions and noise realizations.
"""

from __future__ import annotations

from typing import Optional, Tuple

import torch
from torch import Tensor

from data.base_system import DynamicSystemDataset, get_device, make_controls


class SimpleNonlinearSystemDataset(DynamicSystemDataset):
    """Nonlinear state-space system dataset.

    Generates trajectories from a single nonlinear system with randomly generated
    transition and observation functions. All trajectories share the same system
    dynamics but have different noise realizations and initial conditions.

    Attributes:
        f_params: Transition function parameters (dict)
        g_params: Observation function parameters (dict)
        g_proj: Observation projection matrix (feature_dim, n_obs)
        process_noise_std: Process noise standard deviation
        meas_noise_std: Measurement noise standard deviation
    """

    @staticmethod
    def _generate_single_system_params(
        n_state: int,
        n_ctrl: int,
        device: Optional[torch.device] = None,
    ) -> Tuple[dict, dict]:
        """Generate random nonlinear system parameters.

        Args:
            n_state: State dimension
            n_ctrl: Control dimension (0 if uncontrolled)
            device: PyTorch device

        Returns:
            Tuple of (f_params, g_params) dictionaries containing:
                f_params: Transition function weights (linear, nonlinear terms)
                g_params: Observation function weights (linear, nonlinear terms)
        """
        device = device or get_device()

        # Transition function f: Create TRULY dynamic nonlinear system
        # Key: Strong oscillations + moderate polynomials + damping to prevent drift
        f_params = {
            "linear": torch.rand(1, device=device)[0] * 0.15
            + 0.15,  # 0.15-0.30 (weak persistence)
            "sin": torch.rand(1, device=device)[0] * 0.8
            + 0.8,  # 0.8-1.6 (STRONG oscillation)
            "cos": torch.rand(1, device=device)[0] * 0.8
            + 0.8,  # 0.8-1.6 (STRONG oscillation)
            "tanh": torch.rand(1, device=device)[0] * 0.4 + 0.3,  # 0.3-0.7 (saturation)
            "quad": torch.rand(1, device=device)[0] * 0.2
            + 0.1,  # 0.1-0.3 (moderate quadratic)
            "cubic": torch.rand(1, device=device)[0] * 0.15
            + 0.05,  # 0.05-0.20 (small cubic)
            "mix": torch.rand(1, device=device)[0] * 0.3
            + 0.2,  # 0.2-0.5 (state coupling)
            "mix_matrix": torch.randn(n_state, n_state, device=device) * 0.3,
            "control": torch.rand(1, device=device)[0] * 0.3 + 0.1,
            "damping": torch.rand(1, device=device)[0] * 0.15
            + 0.10,  # 0.10-0.25 (stronger damping)
            "centering": torch.rand(1, device=device)[0] * 0.05
            + 0.02,  # 0.02-0.07 (pull toward zero)
        }
        if n_ctrl > 0:
            f_params["control_matrix"] = (
                torch.randn(n_state, n_ctrl, device=device) * 0.3
            )

        # Observation function g: More nonlinear observations
        g_params = {
            "linear": torch.rand(1, device=device)[0] * 0.4 + 0.3,  # 0.3-0.7
            "sin": torch.rand(1, device=device)[0] * 0.8 + 0.4,  # 0.4-1.2 (stronger)
            "cos": torch.rand(1, device=device)[0] * 0.8 + 0.4,  # 0.4-1.2 (stronger)
            "tanh": torch.rand(1, device=device)[0] * 0.6 + 0.3,  # 0.3-0.9 (stronger)
            "square": torch.rand(1, device=device)[0] * 0.5 + 0.3,  # 0.3-0.8 (stronger)
        }

        return f_params, g_params

    @staticmethod
    def _generate_single_system_trajectories(
        f_params: dict,
        g_params: dict,
        n_state: int,
        n_obs: int,
        n_ctrl: int,
        horizon: int,
        process_noise_std: float,
        meas_noise_std: float,
        u_seq: Optional[Tensor] = None,
        noisy: bool = True,
        device: Optional[torch.device] = None,
    ) -> Tuple[Tensor, Tensor, Optional[Tensor], Tensor]:
        """Generate multiple trajectories from a single nonlinear system.

        Args:
            f_params: Transition function parameters (dict)
            g_params: Observation function parameters (dict)
            n_state: State dimension
            n_obs: Observation dimension
            n_ctrl: Control dimension
            horizon: Number of time steps per trajectory
            process_noise_std: Process noise standard deviation
            meas_noise_std: Measurement noise standard deviation
            u_seq: Control sequences (num_traj, horizon, n_ctrl) or None
            noisy: Whether to add Gaussian noise
            device: PyTorch device

        Returns:
            Tuple of (X, Y, U, g_proj) where:
                X: (num_traj, horizon, n_state) true states
                Y: (num_traj, horizon, n_obs) noisy observations
                U: (num_traj, horizon, n_ctrl) controls or None
                g_proj: (feature_dim, n_obs) observation projection matrix
        """
        batch_size = u_seq.size(0) if u_seq is not None else 1

        # Construct observation projection matrix from active g features
        g_active = {
            k: g_params[k] > 0 for k in ["linear", "sin", "cos", "tanh", "square"]
        }
        feature_dim = n_state * sum(g_active.values())
        g_proj = torch.randn(feature_dim, n_obs, device=device) * 0.5
        g_proj = g_proj / (g_proj.norm(dim=0, keepdim=True) + 1e-6)

        X = torch.zeros(batch_size, horizon, n_state, device=device)
        Y = torch.zeros(batch_size, horizon, n_obs, device=device)
        U = (
            torch.zeros(batch_size, horizon, n_ctrl, device=device)
            if n_ctrl > 0
            else None
        )

        # Start with diverse initial states in a reasonable range
        x = torch.randn(batch_size, n_state, device=device) * 1.5

        for t in range(horizon):
            u_t = u_seq[:, t] if u_seq is not None and n_ctrl > 0 else None

            # Apply TRULY NONLINEAR transition function f
            # Design: Strong oscillations for dynamics + restoring forces for stability
            x_next = f_params["linear"] * x

            # Strong oscillatory terms (create rich dynamics)
            if f_params["sin"] > 0:
                x_next = x_next + f_params["sin"] * torch.sin(x * 2.5)
            if f_params["cos"] > 0:
                x_next = x_next + f_params["cos"] * torch.cos(x * 2.5)

            # Saturation for boundedness
            if f_params["tanh"] > 0:
                x_next = x_next + f_params["tanh"] * torch.tanh(x * 1.5)

            # Moderate polynomial terms (clipped to prevent explosion)
            if f_params["quad"] > 0:
                x_next = x_next + f_params["quad"] * torch.clamp(x * x, -3, 3)
            if f_params["cubic"] > 0:
                x_next = x_next + f_params["cubic"] * torch.clamp(x * x * x, -3, 3)

            # State coupling (cross-dimensional interactions)
            if f_params["mix"] > 0:
                x_next = x_next + f_params["mix"] * (x @ f_params["mix_matrix"].T)

            # Restoring forces (prevent drift and explosion)
            if f_params["damping"] > 0:
                x_next = x_next - f_params["damping"] * x_next  # Velocity damping
            if f_params["centering"] > 0:
                x_next = x_next - f_params["centering"] * x  # Pull toward origin

            # Add control input
            if u_t is not None:
                Bu = f_params["control"] * (
                    u_t @ f_params["control_matrix"].T
                    if "control_matrix" in f_params
                    else u_t
                )
                if Bu.size(-1) != n_state:
                    Bu = torch.cat(
                        [
                            Bu,
                            torch.zeros(
                                Bu.size(0), max(0, n_state - Bu.size(-1)), device=device
                            ),
                        ],
                        dim=-1,
                    )[:, :n_state]
                x_next = x_next + Bu

            if noisy and process_noise_std > 0:
                x_next = x_next + torch.randn_like(x_next) * process_noise_std

            # Apply observation function g
            features = []
            if g_active["linear"]:
                features.append(g_params["linear"] * x_next)
            if g_active["sin"]:
                features.append(g_params["sin"] * torch.sin(x_next))
            if g_active["cos"]:
                features.append(g_params["cos"] * torch.cos(x_next))
            if g_active["tanh"]:
                features.append(g_params["tanh"] * torch.tanh(x_next))
            if g_active["square"]:
                features.append(g_params["square"] * torch.tanh(x_next * x_next))

            y = torch.cat(features, dim=-1) @ g_proj
            if noisy and meas_noise_std > 0:
                y = y + torch.randn_like(y) * meas_noise_std

            X[:, t], Y[:, t] = x_next, y
            if U is not None and u_t is not None:
                U[:, t] = u_t
            x = x_next

        return X, Y, U, g_proj

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
            data_name="nonlinear trajectories",
        )
        self.process_noise_std, self.meas_noise_std = process_noise_std, meas_noise_std

        f_params, g_params = self._generate_single_system_params(
            n_state, n_ctrl, self.device
        )

        # Generate controls (or dummy for batching)
        u_seq = make_controls(
            num_traj, horizon, n_ctrl, random_controls, control_std, self.device
        )

        X, Y, U, g_proj = self._generate_single_system_trajectories(
            f_params,
            g_params,
            n_state,
            n_obs,
            n_ctrl,
            horizon,
            process_noise_std,
            meas_noise_std,
            u_seq,
            noisy=noisy,
            device=self.device,
        )

        # Only store actual controls if n_ctrl > 0
        U_for_store = u_seq if n_ctrl > 0 else None
        self._store_data(X, Y, U_for_store)
        # Store system parameters for filter evaluation
        self.f_params, self.g_params, self.g_proj = f_params, g_params, g_proj
