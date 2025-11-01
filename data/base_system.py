"""
Base class for dynamic system datasets.
"""

from __future__ import annotations

from typing import Optional
import torch
from torch import Tensor
from torch.utils.data import Dataset


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


class DynamicSystemDataset(Dataset):
    """Base class for dynamic system datasets."""

    def __init__(
        self,
        num_distinct_systems: int,
        num_traj_per_system: int,
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
        """Initialize the base dataset.

        Args:
            num_distinct_systems: Number of distinct systems
            num_traj_per_system: Number of trajectories per system
            horizon: Length of each trajectory
            n_state: State dimension
            n_obs: Observation dimension
            n_ctrl: Control dimension
            random_controls: Whether to generate random controls
            control_std: Standard deviation of controls
            noisy: Whether to add noise
            device: Computation device
            seed: Random seed
            data_name: Name of data type for logging
        """
        super().__init__()
        self.num_distinct_systems = num_distinct_systems
        self.num_traj_per_system = num_traj_per_system
        self.num_traj = num_distinct_systems * num_traj_per_system
        self.horizon = horizon
        self.n_state = n_state
        self.n_obs = n_obs
        self.n_ctrl = n_ctrl
        self.random_controls = random_controls
        self.control_std = control_std
        self.noisy = noisy
        self.device = device or get_device()

        print(
            f"Generating {self.num_traj} {data_name} from {num_distinct_systems} distinct systems..."
        )
        if seed is not None:
            set_seed(seed)

    def _store_data(self, X: Tensor, Y: Tensor, U: Optional[Tensor] = None):
        """Store generated data on CPU."""
        self.Xall = X.cpu()
        self.Yall = Y.cpu()
        self.Uall = U.cpu() if U is not None else None
        print(f"Dataset generation complete!")

    def __len__(self) -> int:
        return self.num_traj

    def __getitem__(self, idx: int):
        result = {
            "x": self.Xall[idx],
            "y": self.Yall[idx],
        }
        if self.Uall is not None:
            result["u"] = self.Uall[idx]
        return result
