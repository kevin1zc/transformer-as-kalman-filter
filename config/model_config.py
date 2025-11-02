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
    """Configuration for OneLayerTransformer model (arXiv:2312.06937).

    Implements the Transformer Filter construction from the paper.
    Uses quadratic embeddings and Nadaraya-Watson kernel smoothing.
    """

    def __init__(
        self,
        horizon: int = 10,
        beta: float = 10.0,
    ):
        self.horizon = horizon  # H: number of past states to consider
        self.beta = beta  # Temperature parameter for kernel


class FilterformerConfig:
    """Configuration for Filterformer model.

    Architecture: Filterformer as per paper (arXiv:2310.19603)
    """

    def __init__(
        self,
        n_ref_paths: int = 32,
        n_sim: int = 128,
        n_pos: int = 128,
        n_time: int = 16,
        encoding_dim: int = 360,
        mlp_hidden_dims: list[int] = None,
        n_geometric: int = 32,
        dropout: float = 0.1,
    ):
        self.n_ref_paths = n_ref_paths
        self.n_sim = n_sim
        self.n_pos = n_pos
        self.n_time = n_time
        self.encoding_dim = encoding_dim
        self.mlp_hidden_dims = mlp_hidden_dims or [720, 720]
        self.n_geometric = n_geometric
        self.dropout = dropout


class GRUConfig:
    """Configuration for GRUStateEstimator model.

    Architecture: GRU with hidden_size=256, num_layers=3
    """

    def __init__(
        self,
        hidden_size: int = 256,
        num_layers: int = 3,
        dropout: float = 0.1,
        bidirectional: bool = False,
    ):
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.dropout = dropout
        self.bidirectional = bidirectional


class LTCConfig:
    """Configuration for LiquidTimeConstantModel.

    Architecture: LTC with hidden_size=400, num_layers=3
    """

    def __init__(
        self, hidden_size: int = 400, num_layers: int = 3, dropout: float = 0.1
    ):
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.dropout = dropout
