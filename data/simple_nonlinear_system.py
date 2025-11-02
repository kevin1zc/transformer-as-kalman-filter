"""
Simple Nonlinear System implementation.

Implements a discrete-time nonlinear state space model with optional controls:
    x_{t+1} = f(x_t, u_t) + w_t
    y_t     = g(x_t) + v_t

The system functions f and g are randomly generated for each system instance,
providing diverse nonlinear dynamics for testing. Each system uses:
  - f: Random combination of nonlinearities (tanh, sin, cos, polynomials)
  - g: Random observation mapping with various transformations

Noise w_t and v_t are Gaussian with user-specified standard deviations.
"""

from __future__ import annotations

from typing import Optional, Tuple

import torch
from torch import Tensor

from data.base_system import DynamicSystemDataset, get_device


class SimpleNonlinearSystemDataset(DynamicSystemDataset):
    """Nonlinear dynamic system dataset with random systems per sample."""

    @staticmethod
    def _generate_random_system_params_batch(
        num_systems: int,
        n_state: int,
        n_ctrl: int,
        device: Optional[torch.device] = None,
    ) -> Tuple[dict, dict]:
        """Generate random system parameters for all systems at once."""
        device = device or get_device()

        # f parameters: transition function weights
        f_params = {
            "linear": torch.rand(num_systems, device=device) * 0.25 + 0.7,  # 0.7-0.95
            "tanh": torch.rand(num_systems, device=device) * 0.2 + 0.1,  # 0.1-0.3
            "sin": torch.rand(num_systems, device=device) * 0.2,  # 0.0-0.2
            "cos": torch.rand(num_systems, device=device) * 0.2,  # 0.0-0.2
            "quad": torch.rand(num_systems, device=device) * 0.1,  # 0.0-0.1
            "cubic": torch.rand(num_systems, device=device) * 0.05,  # 0.0-0.05
            "mix": torch.rand(num_systems, device=device) * 0.05,  # 0.0-0.05
            "mix_matrix": torch.randn(num_systems, n_state, n_state, device=device)
            * 0.02,
            "control": torch.rand(num_systems, device=device) * 0.3 + 0.1,  # 0.1-0.4
        }
        if n_ctrl > 0:
            f_params["control_matrix"] = (
                torch.randn(num_systems, n_state, n_ctrl, device=device) * 0.3
            )

        # g parameters: observation function weights
        g_params = {
            "linear": torch.rand(num_systems, device=device) * 0.5 + 0.3,  # 0.3-0.8
            "sin": torch.rand(num_systems, device=device) * 0.4,  # 0.0-0.4
            "cos": torch.rand(num_systems, device=device) * 0.4,  # 0.0-0.4
            "tanh": torch.rand(num_systems, device=device) * 0.3,  # 0.0-0.3
            "square": torch.rand(num_systems, device=device) * 0.2,  # 0.0-0.2
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
        """Generate trajectories for a single system with multiple trajectories."""
        batch_size = u_seq.size(0) if u_seq is not None else 1

        # Build g projection matrix
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

        x = torch.randn(batch_size, n_state, device=device)

        for t in range(horizon):
            u_t = u_seq[:, t] if u_seq is not None and n_ctrl > 0 else None

            # Apply transition function f
            x_next = f_params["linear"] * x
            if f_params["tanh"] > 0:
                x_next += f_params["tanh"] * torch.tanh(x)
            if f_params["sin"] > 0:
                x_next += f_params["sin"] * torch.sin(x)
            if f_params["cos"] > 0:
                x_next += f_params["cos"] * torch.cos(x)
            if f_params["quad"] > 0:
                x_next += f_params["quad"] * torch.tanh(x * x)
            if f_params["cubic"] > 0:
                x_next += f_params["cubic"] * torch.tanh(x * x * x)
            if f_params["mix"] > 0:
                x_next += f_params["mix"] * (x @ f_params["mix_matrix"].T)

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
        num_distinct_systems: int,
        num_traj_per_system: int,
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
            num_distinct_systems,
            num_traj_per_system,
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

        f_params_all, g_params_all = self._generate_random_system_params_batch(
            num_distinct_systems, n_state, n_ctrl, self.device
        )

        # Pre-generate controls if needed
        # Generate controls for batch_size even if n_ctrl=0 (to control batch size)
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
        elif n_ctrl == 0:
            # Generate dummy controls to control batch size (will be ignored in generation)
            U_per_sys = torch.zeros(
                num_distinct_systems,
                num_traj_per_system,
                horizon,
                1,
                device=self.device,
            )
        else:
            U_per_sys = None

        # Generate trajectories for each system
        X_list, Y_list, U_list, g_proj_list = [], [], [], []
        for sys_idx in range(num_distinct_systems):
            f_params = {
                k: v[sys_idx] if v.dim() == 1 else v[sys_idx]
                for k, v in f_params_all.items()
            }
            g_params = {k: v[sys_idx] for k, v in g_params_all.items()}
            f_params["mix_matrix"] = f_params_all["mix_matrix"][sys_idx]
            if "control_matrix" in f_params_all:
                f_params["control_matrix"] = f_params_all["control_matrix"][sys_idx]

            u_seq = U_per_sys[sys_idx] if U_per_sys is not None else None
            X_sys, Y_sys, U_sys, g_proj = self._generate_single_system_trajectories(
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
            X_list.append(X_sys)
            Y_list.append(Y_sys)
            U_list.append(U_sys if U_sys is not None else None)
            g_proj_list.append(g_proj)

        X_all, Y_all = torch.cat(X_list, dim=0), torch.cat(Y_list, dim=0)
        U_all = torch.cat(U_list, dim=0) if any(u is not None for u in U_list) else None

        self._store_data(X_all, Y_all, U_all)
        self.f_params_all, self.g_params_all, self.g_proj_all = (
            f_params_all,
            g_params_all,
            g_proj_list,
        )
