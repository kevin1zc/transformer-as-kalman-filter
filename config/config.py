"""
Configuration management for Transformer Kalman Filter experiment.
"""

from dataclasses import dataclass
from typing import Optional, Union
import torch


@dataclass
class LDSConfig:
    """Configuration for Linear Dynamical System."""

    n_state: int = 4
    n_ctrl: int = 0
    n_obs: int = 4
    horizon: int = 64
    batch_size: int = 128
    process_noise_std: float = 0.05
    meas_noise_std: float = 0.05
    control_std: float = 0.0
    stable_radius: float = 0.95
    device: Optional[torch.device] = None


@dataclass
class TransformerConfig:
    """Configuration for Transformer model."""

    d_model: int = 128
    n_heads: int = 4
    dropout: float = 0.1
    max_len: int = 1024
    mlp_identity: bool = True
    use_positional_encoding: bool = False


@dataclass
class GRUConfig:
    """Configuration for GRU model."""

    hidden_size: int = 128
    num_layers: int = 2
    dropout: float = 0.1
    bidirectional: bool = False


@dataclass
class TrainingConfig:
    """Configuration for training procedure."""

    num_train_traj: int = 5000
    num_val_traj: int = 1000
    max_epochs: int = 20
    patience: int = 5
    learning_rate: float = 2e-3
    weight_decay: float = 1e-2
    grad_clip: float = 1.0
    seed: int = 42
    enable_visualization: bool = True
    save_model: bool = True
    model_save_path: str = "best_lqs_transformer.pt"


@dataclass
class ExperimentConfig:
    """Complete experiment configuration."""

    lds: LDSConfig
    model: Union[TransformerConfig, GRUConfig]
    training: TrainingConfig

    @classmethod
    def default(
        cls,
        n_state: int = 4,
        n_ctrl: int = 0,
        n_obs: int = 4,
        horizon: int = 64,
        device: Optional[torch.device] = None,
        model_type: str = "transformer",
    ):
        """Create default configuration."""
        if model_type == "transformer":
            model_config = TransformerConfig()
        elif model_type == "gru":
            model_config = GRUConfig()
        else:
            raise ValueError(f"Unknown model type: {model_type}")

        return cls(
            lds=LDSConfig(
                n_state=n_state,
                n_ctrl=n_ctrl,
                n_obs=n_obs,
                horizon=horizon,
                device=device,
            ),
            model=model_config,
            training=TrainingConfig(),
        )
