"""
Bootstrap Particle Filter (Sequential Importance Resampling).

This module provides a simple particle filter suitable for nonlinear state
space models with additive Gaussian process and measurement noise.

API is designed to integrate with the project's nonlinear system utilities.
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
    B, N = weights.shape
    # Draw N categorical samples per batch
    # torch.multinomial expects probs sum to 1 per row
    idx = torch.multinomial(weights, num_samples=N, replacement=True)
    return idx


@torch.no_grad()
def particle_filter_sequence(
    f: Callable[[Tensor, Optional[Tensor]], Tensor],
    g: Callable[[Tensor, int], Tensor],
    q_std: float,
    r_std: float,
    Y: Tensor,
    *,
    num_particles: int = 1024,
    U: Optional[Tensor] = None,
    n_obs: Optional[int] = None,
    x0_mean: Optional[Tensor] = None,
    x0_std: float = 1.0,
) -> Tensor:
    """Run a bootstrap particle filter for the full sequence.

    Args:
        f: transition function, f(x_t, u_t) -> x_{t+1} with shapes (B*N, n_x)
        g: observation function, g(x_t, n_obs) -> y_t with shapes (B*N, n_y)
        q_std: process noise std (assumed isotropic Gaussian)
        r_std: measurement noise std (assumed isotropic Gaussian on each dim)
        Y: (B, T, n_y) observations
        num_particles: number of particles
        U: (B, T, n_u) or None
        n_obs: observation dimension override (if None, taken from Y)
        x0_mean: (B, n_x) initial mean; if None, zeros
        x0_std: initial std for particle sampling
    Returns:
        XH: (B, T, n_x) estimated state means
    """
    device = Y.device
    B, T, n_y = Y.shape
    n_y_eff = n_obs or n_y

    # Infer state dimension from x0_mean or from a single forward pass via g
    # We'll sample initial particles in n_x from x0_mean/std; we need n_x.
    # To get n_x, we sample dummy x from standard normal once g is not enough.
    if x0_mean is None:
        # default to 1D if cannot infer; but better try infer from g by probing
        # Try n_x=1 probe
        n_x = 1
        x0_mean = torch.zeros(B, n_x, device=device)
    else:
        n_x = x0_mean.size(-1)

    # Particles: (B, N, n_x)
    x_particles = x0_mean.unsqueeze(1) + x0_std * torch.randn(
        B, num_particles, n_x, device=device
    )
    # Weights: start uniform
    log_w = torch.zeros(B, num_particles, device=device)

    XH = torch.zeros(B, T, n_x, device=device)
    q = q_std
    r = r_std

    for t in range(T):
        # Propagate
        if U is not None:
            u_t = U[:, t]
            u_rep = u_t.unsqueeze(1).expand(B, num_particles, -1)
            x_in = x_particles.reshape(B * num_particles, n_x)
            u_in = u_rep.reshape(B * num_particles, -1)
            x_pred = f(x_in, u_in)
        else:
            x_in = x_particles.reshape(B * num_particles, n_x)
            x_pred = f(x_in, None)
        # Add process noise
        x_pred = x_pred + q * torch.randn_like(x_pred)
        x_particles = x_pred.reshape(B, num_particles, n_x)

        # Weights update via likelihood p(y_t | x_t)
        y_pred = g(x_particles.reshape(B * num_particles, n_x), n_y_eff)
        y_pred = y_pred.reshape(B, num_particles, n_y_eff)
        y_t = Y[:, t].unsqueeze(1)  # (B,1,n_y)
        resid = y_t - y_pred  # (B,N,n_y)
        # Gaussian independent dims: log-likelihood up to constant
        # log p = -0.5 * sum((resid/r)^2) - n_y*log(r) + const
        ll = -0.5 * ((resid / max(r, 1e-6)) ** 2).sum(dim=-1)
        log_w = log_w + ll

        # Normalize weights
        log_w = log_w - (torch.logsumexp(log_w, dim=1, keepdim=True))
        w = torch.exp(log_w)

        # Estimate mean
        XH[:, t] = (w.unsqueeze(-1) * x_particles).sum(dim=1)

        # Resample
        idx = multinomial_resample(w)
        # gather particles
        batch_idx = torch.arange(B, device=device).unsqueeze(1)
        x_particles = x_particles[batch_idx, idx]
        # reset weights to uniform in log space
        log_w = torch.zeros(B, num_particles, device=device)

    return XH


