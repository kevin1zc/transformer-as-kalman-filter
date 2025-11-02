"""
Bootstrap Particle Filter (Sequential Importance Resampling).

This module provides a simple particle filter suitable for nonlinear state
space models with additive Gaussian process and measurement noise.
"""

from __future__ import annotations

from typing import Callable, Optional

import torch
from torch import Tensor


def multinomial_resample(weights: Tensor) -> Tensor:
    """Multinomial resampling.

    Args:
        weights: (B, N) normalized weights
    Returns:
        idx: (B, N) integer indices
    """
    return torch.multinomial(weights, num_samples=weights.size(1), replacement=True)


@torch.no_grad()
def particle_filter_sequence(
    f: Callable[[Tensor, Optional[Tensor]], Tensor],
    g: Callable[[Tensor], Tensor],
    q_std: float,
    r_std: float,
    Y: Tensor,
    *,
    num_particles: int = 1024,
    U: Optional[Tensor] = None,
    x0_mean: Optional[Tensor] = None,
    x0_std: float = 1.0,
) -> Tensor:
    """Run a bootstrap particle filter for the full sequence.

    Args:
        f: transition function, f(x_t, u_t) -> x_{t+1} with shapes (B*N, n_x)
        g: observation function, g(x_t) -> y_t with shapes (B*N, n_y)
        q_std: process noise std (assumed isotropic Gaussian)
        r_std: measurement noise std (assumed isotropic Gaussian on each dim)
        Y: (B, T, n_y) observations
        num_particles: number of particles
        U: (B, T, n_u) or None
        x0_mean: (B, n_x) or None, initial mean
        x0_std: initial std for particle sampling
    Returns:
        XH: (B, T, n_x) estimated state means
    """
    device = Y.device
    B, T, n_y = Y.shape

    if x0_mean is None:
        # Infer state dimension from observation dimension (rough heuristic)
        n_x = Y.shape[-1]  # Assume n_x == n_y if not specified
        x0_mean = torch.zeros(B, n_x, device=device)
    else:
        x0_mean = x0_mean.to(device)
        n_x = x0_mean.size(-1)
    x_particles = x0_mean.unsqueeze(1) + x0_std * torch.randn(
        B, num_particles, n_x, device=device
    )
    log_w = torch.zeros(B, num_particles, device=device)
    XH = torch.zeros(B, T, n_x, device=device)

    for t in range(T):
        # Propagate
        x_in = x_particles.reshape(B * num_particles, n_x)
        u_in = None
        if U is not None:
            u_in = (
                U[:, t]
                .unsqueeze(1)
                .expand(B, num_particles, -1)
                .reshape(B * num_particles, -1)
            )
        x_pred = f(x_in, u_in)
        x_pred = x_pred + q_std * torch.randn_like(x_pred)
        x_particles = x_pred.reshape(B, num_particles, n_x)

        # Update weights
        y_pred = g(x_particles.reshape(B * num_particles, n_x))
        y_pred = y_pred.reshape(B, num_particles, -1)
        resid = Y[:, t].unsqueeze(1) - y_pred
        ll = -0.5 * ((resid / max(r_std, 1e-6)) ** 2).sum(dim=-1)
        log_w = log_w + ll

        # Normalize and estimate
        log_w = log_w - torch.logsumexp(log_w, dim=1, keepdim=True)
        w = torch.exp(log_w)
        XH[:, t] = (w.unsqueeze(-1) * x_particles).sum(dim=1)

        # Resample
        idx = multinomial_resample(w)
        batch_idx = torch.arange(B, device=device).unsqueeze(1)
        x_particles = x_particles[batch_idx, idx]
        log_w.zero_()

    return XH
