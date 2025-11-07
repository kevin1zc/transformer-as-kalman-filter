"""
Base class for dynamic system datasets.

Provides common functionality for generating and storing time-series data
from linear and nonlinear dynamical systems with optional controls.
"""

from __future__ import annotations

from typing import Optional
import torch
from torch import Tensor
from torch.utils.data import Dataset


def get_device(prefer: str = "cuda") -> torch.device:
    """Get the best available compute device.

    Args:
        prefer: Preferred device ("cuda" or "cpu")

    Returns:
        Device object (CUDA if available and preferred, otherwise CPU)
    """
    if prefer == "cuda" and torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def set_seed(seed: int = 1337) -> None:
    """Set random seeds for reproducibility across libraries.

    Args:
        seed: Random seed value
    """
    import random
    import numpy as np

    random.seed(seed)
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


class DynamicSystemDataset(Dataset):
    """Base class for dynamical system datasets.

    Generates multiple trajectories from a single system instance. Each trajectory
    contains states, observations, and optionally controls over a fixed horizon.

    Attributes:
        num_traj: Total number of trajectories
        horizon: Length of each trajectory
        n_state: State dimension
        n_obs: Observation dimension
        n_ctrl: Control dimension
        device: Compute device (CPU/CUDA)
        Xall: All true states (num_traj, horizon, n_state)
        Yall: All observations (num_traj, horizon, n_obs)
        Uall: All control inputs (num_traj, horizon, n_ctrl) or None
    """

    def __init__(
        self,
        num_traj: int,
        horizon: int,
        n_state: int,
        n_obs: int,
        n_ctrl: int,
        random_controls: bool,
        control_std: float,
        noisy: bool,
        device: Optional[torch.device],
        seed: Optional[int],
        data_name: str = "trajectories",
    ):
        """Initialize the dataset with a single system generating multiple trajectories.

        Args:
            num_traj: Total number of trajectories to generate
            horizon: Length (time steps) of each trajectory
            n_state: State dimension
            n_obs: Observation dimension
            n_ctrl: Control dimension (0 if uncontrolled)
            random_controls: Whether to generate random control inputs
            control_std: Standard deviation of control noise
            noisy: Whether to add process and measurement noise
            device: PyTorch device (None for auto-selection)
            seed: Random seed for reproducibility
            data_name: Descriptive name for logging
        """
        super().__init__()
        self.num_traj = num_traj
        self.horizon = horizon
        self.n_state = n_state
        self.n_obs = n_obs
        self.n_ctrl = n_ctrl
        self.random_controls = random_controls
        self.control_std = control_std
        self.noisy = noisy
        self.device = device or get_device()

        print(f"Generating {self.num_traj} {data_name} from a single system...")
        if seed is not None:
            set_seed(seed)

    def _store_data(self, X: Tensor, Y: Tensor, U: Optional[Tensor] = None) -> None:
        """Store generated trajectories on CPU for PyTorch Dataset compatibility.

        Args:
            X: True states (num_traj, horizon, n_state)
            Y: Observations (num_traj, horizon, n_obs)
            U: Control inputs (num_traj, horizon, n_ctrl) or None
        """
        self.Xall = X.cpu()
        self.Yall = Y.cpu()
        self.Uall = U.cpu() if U is not None else None
        print("Dataset generation complete!")

    def __len__(self) -> int:
        """Return the number of trajectories in the dataset."""
        return self.num_traj

    def __getitem__(self, idx: int) -> dict[str, Tensor]:
        """Get a single trajectory by index.

        Args:
            idx: Trajectory index

        Returns:
            Dictionary with keys:
                'x': True states (horizon, n_state)
                'y': Observations (horizon, n_obs)
                'u': Control inputs (horizon, n_ctrl) if available
        """
        result = {"x": self.Xall[idx], "y": self.Yall[idx]}
        if self.Uall is not None:
            result["u"] = self.Uall[idx]
        return result


def make_controls(
    num_traj: int,
    horizon: int,
    n_ctrl: int,
    random_controls: bool,
    control_std: float,
    device: torch.device,
) -> Optional[Tensor]:
    """Utility to generate control sequences or a dummy tensor for batching.

    Returns:
        - (num_traj, horizon, n_ctrl) if n_ctrl>0 and random_controls
        - zeros (num_traj, horizon, 1) as dummy if n_ctrl==0
        - zeros (num_traj, horizon, n_ctrl) if n_ctrl>0 but not randomized
    """
    if n_ctrl > 0 and random_controls and control_std > 0:
        return torch.randn(num_traj, horizon, n_ctrl, device=device) * control_std
    if n_ctrl > 0:
        return torch.zeros(num_traj, horizon, n_ctrl, device=device)
    # n_ctrl == 0 -> dummy last-dim for batch sizing
    return torch.zeros(num_traj, horizon, 1, device=device)
