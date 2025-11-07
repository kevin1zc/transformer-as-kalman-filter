"""
Models package for state estimation experiments.

This package contains implementations of:
- OneLayerTransformer: Trainable transformer for linear systems (arXiv:2312.06937)
- KalmanFilterExact: Oracle Kalman filter baselines
- Filterformer & FilterformerPractical: Nonlinear filtering architectures (arXiv:2310.19603)
- GRUStateEstimator: Recurrent neural network baseline
- MambaStateEstimator & Mamba2StateEstimator: Selective state space models
- particle_filter_sequence: Oracle particle filter for nonlinear systems
"""

from .one_layer_transformer import OneLayerTransformer, KalmanFilterExact
from .gru_model import GRUStateEstimator
from .kalman_filter import kalman_filter_sequence
from .filterformer import Filterformer, FilterformerPractical
from .particle_filter import particle_filter_sequence
from .mamba_model import MambaStateEstimator, Mamba2StateEstimator

__all__ = [
    "OneLayerTransformer",
    "KalmanFilterExact",
    "GRUStateEstimator",
    "kalman_filter_sequence",
    "Filterformer",
    "FilterformerPractical",
    "particle_filter_sequence",
    "MambaStateEstimator",
    "Mamba2StateEstimator",
]
