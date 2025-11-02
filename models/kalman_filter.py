"""
Kalman Filter implementation for state estimation.

This module provides batched discrete-time Kalman filter implementations
for linear and nonlinear dynamical systems.
"""

from __future__ import annotations

from typing import Optional

import torch
from torch import Tensor


@torch.no_grad()
def kalman_filter_sequence(
    A: Tensor,
    H: Tensor,
    Q: Tensor,
    R: Tensor,
    Y: Tensor,
    U: Optional[Tensor] = None,
    Bmat: Optional[Tensor] = None,
    x0: Optional[Tensor] = None,
    P0: Optional[Tensor] = None,
) -> Tensor:
    """Batched discrete-time Kalman filter over a full sequence.

    Args:
        A: (n_x, n_x) state transition matrix
        H: (n_y, n_x) observation matrix
        Q: (n_x, n_x) process noise covariance
        R: (n_y, n_y) measurement noise covariance
        Y: (B, T, n_y) observations
        U: (B, T, n_u) or None, control inputs
        Bmat: (n_x, n_u) or None, control input matrix
        x0: (B, n_x) or None, initial state estimate
        P0: (B, n_x, n_x) or None, initial state covariance
    Returns:
        XH: (B, T, n_x) filtered state estimates
    """
    device = Y.device
    Bsz, T, n_y = Y.shape
    n_x = A.size(0)

    A, H, Q, R = A.to(device), H.to(device), Q.to(device), R.to(device)
    if Bmat is not None:
        Bmat = Bmat.to(device)

    if x0 is None:
        H_pinv = torch.linalg.pinv(H)
        x = (H_pinv @ Y[:, 0:1].transpose(-1, -2)).squeeze(-1)
    else:
        x = x0.to(device)

    P = (
        torch.eye(n_x, device=device).expand(Bsz, n_x, n_x).clone()
        if P0 is None
        else P0.to(device)
    )
    XH = torch.zeros(Bsz, T, n_x, device=device)
    I = torch.eye(n_x, device=device)

    for t in range(T):
        # Predict
        x_pred = x @ A.T
        if U is not None and Bmat is not None:
            x_pred = x_pred + U[:, t] @ Bmat.T
        P_pred = A.expand(Bsz, -1, -1) @ P @ A.T.expand(Bsz, -1, -1) + Q.expand(
            Bsz, -1, -1
        )

        # Update
        y_pred = x_pred @ H.T
        r = Y[:, t] - y_pred
        S = H.expand(Bsz, -1, -1) @ P_pred @ H.T.expand(Bsz, -1, -1) + R.expand(
            Bsz, -1, -1
        )

        HPt = P_pred @ H.T.expand(Bsz, -1, -1)
        K = torch.linalg.solve(S, HPt.transpose(1, 2)).transpose(1, 2)

        x = x_pred + torch.bmm(K, r.unsqueeze(-1)).squeeze(-1)
        P = (I.expand(Bsz, -1, -1) - K @ H.expand(Bsz, -1, -1)) @ P_pred

        XH[:, t] = x

    return XH
