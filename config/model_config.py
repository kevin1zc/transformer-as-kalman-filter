"""
Model configuration classes for neural network models.

These configs are designed for datasets with:
- 50,000 trajectories
- Horizon length: 100

Configs are tuned to have roughly 1:5 ratio of trainable parameters to training data.
With 50,000 × 100 = 5,000,000 training steps, target is ~1,000,000 parameters per model.
"""

from __future__ import annotations


class OneLayerTransformerConfig:
    """Configuration for OneLayerTransformer (arXiv:2312.06937).

    Key parameters:
    - horizon: attention history length (set large to approximate full history)
    - beta: kernel temperature; larger -> more concentrated weights
    - use_qr_kernel: if True, use Gaussian NW kernel with Q^{-1}, R^{-1}
    - use_time_varying_gain: if True, allow time-varying K_t (if provided)
    - full_history: convenience flag; if True, model may override to use all past
    """

    def __init__(
        self,
        horizon: int = 100,
        beta: float = 10.0,
        use_qr_kernel: bool = False,
        use_time_varying_gain: bool = False,
        full_history: bool = False,
    ):
        self.horizon = horizon
        self.beta = beta
        self.use_qr_kernel = use_qr_kernel
        self.use_time_varying_gain = use_time_varying_gain
        self.full_history = full_history


class FilterformerConfig:
    """Configuration for Filterformer model.

    Architecture: Filterformer as per paper (arXiv:2310.19603) (~54K params)
    """

    def __init__(
        self,
        n_ref_paths: int = 32,
        n_sim: int = 32,
        n_pos: int = 32,
        n_time: int = 20,
        encoding_dim: int = 96,
        mlp_hidden_dims: list[int] = None,
        n_geometric: int = 16,
        dropout: float = 0.2,
        horizon: int = 100,
    ):
        self.n_ref_paths = n_ref_paths
        self.n_sim = n_sim
        self.n_pos = n_pos
        self.n_time = n_time
        self.encoding_dim = encoding_dim
        self.mlp_hidden_dims = mlp_hidden_dims or [128, 128, 64]
        self.n_geometric = n_geometric
        self.dropout = dropout
        self.horizon = horizon


class GRUConfig:
    """Configuration for GRUStateEstimator model.

    Architecture: GRU with hidden_size=64, num_layers=2 (~54K params)
    """

    def __init__(
        self,
        hidden_size: int = 64,
        num_layers: int = 2,
        dropout: float = 0.1,
        bidirectional: bool = False,
    ):
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.dropout = dropout
        self.bidirectional = bidirectional


class MambaConfig:
    """Configuration for Mamba state estimator.

    Architecture: Selective state space model (~41K params)
    """

    def __init__(
        self,
        d_model: int = 48,
        d_state: int = 16,
        d_conv: int = 4,
        expand: int = 2,
        n_layers: int = 2,
    ):
        self.d_model = d_model
        self.d_state = d_state
        self.d_conv = d_conv
        self.expand = expand
        self.n_layers = n_layers


class Mamba2Config:
    """Configuration for Mamba2 state estimator.

    Architecture: Improved SSD formulation (~49K params)
    Note: d_ssm = expand * d_model must be divisible by headdim
    """

    def __init__(
        self,
        d_model: int = 32,
        d_state: int = 128,
        d_conv: int = 4,
        expand: int = 2,
        headdim: int = 16,
        n_layers: int = 3,
    ):
        self.d_model = d_model
        self.d_state = d_state
        self.d_conv = d_conv
        self.expand = expand
        self.headdim = headdim
        self.n_layers = n_layers
