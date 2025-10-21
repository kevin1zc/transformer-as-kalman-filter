"""
Nonlinear system generator and dataset for testing Filterformer.

We implement a discrete-time nonlinear state space model with optional
controls:
    x_{t+1} = f(x_t, u_t) + w_t
    y_t     = g(x_t) + v_t

The system functions f and g are randomly generated for each system instance,
providing diverse nonlinear dynamics for testing. Each system uses:
  - f: Random combination of nonlinearities (tanh, sin, cos, polynomials)
  - g: Random observation mapping with various transformations

Noise w_t and v_t are Gaussian with user-specified standard deviations.
"""

from __future__ import annotations

from typing import Optional, Tuple, Dict, Any
import random

import torch
from torch import Tensor
from torch.utils.data import Dataset


def get_device(prefer: str = "cuda") -> torch.device:
    if prefer == "cuda" and torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


class RandomSystemGenerator:
    """Generates random nonlinear system functions f and g."""
    
    def __init__(self, n_state: int, n_obs: int, n_ctrl: int = 0, seed: Optional[int] = None):
        self.n_state = n_state
        self.n_obs = n_obs
        self.n_ctrl = n_ctrl
        self.rng = random.Random(seed)
        
        # Generate random parameters for f and g
        self._generate_f_params()
        self._generate_g_params()
    
    def _generate_f_params(self):
        """Generate random parameters for transition function f."""
        # Random combination of nonlinearities
        self.f_linear_weight = self.rng.uniform(0.7, 0.95)  # Stability weight
        self.f_tanh_weight = self.rng.uniform(0.1, 0.3)
        self.f_sin_weight = self.rng.uniform(0.0, 0.2)
        self.f_cos_weight = self.rng.uniform(0.0, 0.2)
        self.f_quad_weight = self.rng.uniform(0.0, 0.1)
        self.f_cubic_weight = self.rng.uniform(0.0, 0.05)
        
        # Random mixing matrix for cross terms
        self.f_mix_weight = self.rng.uniform(0.0, 0.05)
        self.f_mix_matrix = torch.randn(self.n_state, self.n_state) * 0.02
        
        # Control influence
        self.f_control_weight = self.rng.uniform(0.1, 0.4)
        if self.n_ctrl > 0:
            self.f_control_matrix = torch.randn(self.n_state, self.n_ctrl) * 0.3
    
    def _generate_g_params(self):
        """Generate random parameters for observation function g."""
        # Random combination of transformations
        self.g_linear_weight = self.rng.uniform(0.3, 0.8)
        self.g_sin_weight = self.rng.uniform(0.0, 0.4)
        self.g_cos_weight = self.rng.uniform(0.0, 0.4)
        self.g_tanh_weight = self.rng.uniform(0.0, 0.3)
        self.g_square_weight = self.rng.uniform(0.0, 0.2)
        
        # Random projection matrix
        feature_dim = self.n_state * (1 + int(self.g_sin_weight > 0) + int(self.g_cos_weight > 0) + 
                                    int(self.g_tanh_weight > 0) + int(self.g_square_weight > 0))
        self.g_projection = torch.randn(feature_dim, self.n_obs) * 0.5
        self.g_projection = self.g_projection / (self.g_projection.norm(dim=0, keepdim=True) + 1e-6)
    
    @torch.no_grad()
    def f(self, x: Tensor, u: Optional[Tensor]) -> Tensor:
        """Random nonlinear transition function f(x,u) -> x_next."""
        x_next = self.f_linear_weight * x
        
        # Add various nonlinear terms
        if self.f_tanh_weight > 0:
            x_next += self.f_tanh_weight * torch.tanh(x)
        
        if self.f_sin_weight > 0:
            x_next += self.f_sin_weight * torch.sin(x)
        
        if self.f_cos_weight > 0:
            x_next += self.f_cos_weight * torch.cos(x)
        
        if self.f_quad_weight > 0:
            x_next += self.f_quad_weight * torch.tanh(x * x)
        
        if self.f_cubic_weight > 0:
            x_next += self.f_cubic_weight * torch.tanh(x * x * x)
        
        # Cross terms
        if self.f_mix_weight > 0:
            mix = x @ self.f_mix_matrix.to(x.device)
            x_next += self.f_mix_weight * mix
        
        # Control input
        if u is not None and u.numel() > 0:
            if hasattr(self, 'f_control_matrix'):
                Bu = self.f_control_weight * (u @ self.f_control_matrix.to(x.device).T)
            else:
                Bu = self.f_control_weight * u
            
            # Pad/truncate control to state dimension if needed
            if Bu.size(-1) < x_next.size(-1):
                pad = torch.zeros(Bu.size(0), x_next.size(-1) - Bu.size(-1), device=x.device)
                Bu = torch.cat([Bu, pad], dim=-1)
            elif Bu.size(-1) > x_next.size(-1):
                Bu = Bu[..., :x_next.size(-1)]
            x_next = x_next + Bu
        
        return x_next
    
    @torch.no_grad()
    def g(self, x: Tensor) -> Tensor:
        """Random nonlinear observation function g(x) -> y."""
        features = []
        
        # Linear term
        if self.g_linear_weight > 0:
            features.append(self.g_linear_weight * x)
        
        # Nonlinear terms
        if self.g_sin_weight > 0:
            features.append(self.g_sin_weight * torch.sin(x))
        
        if self.g_cos_weight > 0:
            features.append(self.g_cos_weight * torch.cos(x))
        
        if self.g_tanh_weight > 0:
            features.append(self.g_tanh_weight * torch.tanh(x))
        
        if self.g_square_weight > 0:
            features.append(self.g_square_weight * torch.tanh(x * x))
        
        # Combine features and project to observation space
        if features:
            feats = torch.cat(features, dim=-1)
        else:
            feats = x
        
        # Apply random projection
        y = feats @ self.g_projection.to(x.device)
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
        seed: Optional[int] = None,
    ):
        self.n_state = n_state
        self.n_obs = n_obs
        self.n_ctrl = n_ctrl
        self.q = process_noise_std
        self.r = meas_noise_std
        self.device = device or get_device()
        
        # Generate random system functions
        self.system_generator = RandomSystemGenerator(
            n_state=n_state, 
            n_obs=n_obs, 
            n_ctrl=n_ctrl, 
            seed=seed
        )

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

            x = self.system_generator.f(x, u_t)
            if noisy and self.q > 0:
                x = x + torch.randn_like(x) * self.q
            y = self.system_generator.g(x)
            if noisy and self.r > 0:
                y = y + torch.randn_like(y) * self.r

            X[:, t] = x
            Y[:, t] = y

        return X, Y, U
    
    def regenerate_system(self, seed: Optional[int] = None):
        """Regenerate the system with new random parameters."""
        self.system_generator = RandomSystemGenerator(
            n_state=self.n_state,
            n_obs=self.n_obs, 
            n_ctrl=self.n_ctrl,
            seed=seed
        )
    
    def get_system_info(self) -> Dict[str, Any]:
        """Get information about the current random system parameters."""
        return {
            'f_params': {
                'linear_weight': self.system_generator.f_linear_weight,
                'tanh_weight': self.system_generator.f_tanh_weight,
                'sin_weight': self.system_generator.f_sin_weight,
                'cos_weight': self.system_generator.f_cos_weight,
                'quad_weight': self.system_generator.f_quad_weight,
                'cubic_weight': self.system_generator.f_cubic_weight,
                'mix_weight': self.system_generator.f_mix_weight,
                'control_weight': self.system_generator.f_control_weight,
            },
            'g_params': {
                'linear_weight': self.system_generator.g_linear_weight,
                'sin_weight': self.system_generator.g_sin_weight,
                'cos_weight': self.system_generator.g_cos_weight,
                'tanh_weight': self.system_generator.g_tanh_weight,
                'square_weight': self.system_generator.g_square_weight,
            }
        }


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
        
        # Generate all trajectories at once for efficiency
        print(f"Generating {num_traj} trajectories of length {horizon}...")
        if sys.n_ctrl > 0 and random_controls and control_std > 0:
            Uall = (
                torch.randn(num_traj, horizon, sys.n_ctrl, device=sys.device)
                * control_std
            )
        else:
            Uall = None
            
        # Generate all data in one batch
        Xall, Yall, Uall = sys.generate(
            horizon=horizon,
            batch_size=num_traj,
            noisy=noisy,
            u_seq=Uall,
        )
        
        # Store all data
        self.Xall = Xall.cpu()
        self.Yall = Yall.cpu()
        self.Uall = Uall.cpu() if Uall is not None else None
        print(f"Dataset generation complete!")

    def __len__(self) -> int:
        return self.num_traj

    def __getitem__(self, idx: int):
        result = {"x": self.Xall[idx], "y": self.Yall[idx]}
        if self.Uall is not None:
            result["u"] = self.Uall[idx]
        return result
