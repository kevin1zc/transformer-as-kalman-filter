"""
Training and evaluation utilities for Transformer Kalman Filter experiment.
"""

from __future__ import annotations

from typing import Optional, Tuple
from datetime import datetime
import os
import json

import torch
from torch import nn
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import numpy as np

from config import ExperimentConfig
from data import LDS, LDSSequenceDataset, set_seed
from models import OneLayerTransformer, kalman_filter_sequence


def create_output_directory(model_type: str, n_state: int, n_obs: int) -> str:
    """Create a timestamped output directory for the current training run."""
    # Create output directory if it doesn't exist
    os.makedirs("output", exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    dir_name = f"{model_type}_{timestamp}"
    output_dir = os.path.join("output", dir_name)
    os.makedirs(output_dir, exist_ok=True)
    return output_dir


def save_training_parameters(
    cfg: ExperimentConfig, output_dir: str, training_history: dict
):
    """Save training parameters and configuration to a JSON file."""
    params = {
        "experiment_info": {
            "timestamp": datetime.now().isoformat(),
            "model_type": "transformer" if hasattr(cfg.model, "d_model") else "gru",
            "n_state": cfg.lds.n_state,
            "n_obs": cfg.lds.n_obs,
            "n_ctrl": cfg.lds.n_ctrl,
            "horizon": cfg.lds.horizon,
        },
        "lds_config": {
            "batch_size": cfg.lds.batch_size,
            "process_noise_std": cfg.lds.process_noise_std,
            "meas_noise_std": cfg.lds.meas_noise_std,
            "control_std": cfg.lds.control_std,
            "stable_radius": cfg.lds.stable_radius,
        },
        "model_config": {
            "d_model": getattr(cfg.model, "d_model", None),
            "n_heads": getattr(cfg.model, "n_heads", None),
            "hidden_size": getattr(cfg.model, "hidden_size", None),
            "num_layers": getattr(cfg.model, "num_layers", None),
            "dropout": cfg.model.dropout,
            "max_len": getattr(cfg.model, "max_len", None),
            "bidirectional": getattr(cfg.model, "bidirectional", None),
            "use_positional_encoding": getattr(
                cfg.model, "use_positional_encoding", None
            ),
        },
        "training_config": {
            "num_train_traj": cfg.training.num_train_traj,
            "num_val_traj": cfg.training.num_val_traj,
            "max_epochs": cfg.training.max_epochs,
            "patience": cfg.training.patience,
            "learning_rate": cfg.training.learning_rate,
            "weight_decay": cfg.training.weight_decay,
            "grad_clip": cfg.training.grad_clip,
            "seed": cfg.training.seed,
        },
        "training_results": {
            "best_val_loss": training_history.get("best_val", None),
            "final_train_loss": (
                training_history["train_loss"][-1]
                if training_history["train_loss"]
                else None
            ),
            "final_val_loss": (
                training_history["val_loss"][-1]
                if training_history["val_loss"]
                else None
            ),
            "total_epochs": len(training_history["train_loss"]),
        },
    }

    params_path = os.path.join(output_dir, "training_parameters.json")
    with open(params_path, "w") as f:
        json.dump(params, f, indent=2)

    print(f"Training parameters saved to: {params_path}")


def train_epoch(
    model: nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    scaler: Optional[torch.amp.GradScaler] = None,
    *,
    device: torch.device,
    grad_clip: float = 1.0,
) -> float:
    """Train for one epoch."""
    model.train()
    total_loss = 0.0
    n = 0
    use_cuda = device.type == "cuda"
    for batch in loader:
        y = batch["y"].to(device, non_blocking=use_cuda)
        x = batch["x"].to(device, non_blocking=use_cuda)
        u = batch["u"].to(device, non_blocking=use_cuda) if "u" in batch else None
        optimizer.zero_grad(set_to_none=True)
        with torch.amp.autocast("cuda", enabled=use_cuda):
            x_hat = model(y, u)
            loss = nn.functional.mse_loss(x_hat, x)
        if scaler is not None and use_cuda:
            scaler.scale(loss).backward()
            if grad_clip is not None:
                scaler.unscale_(optimizer)
                nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            if grad_clip is not None:
                nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            optimizer.step()
        total_loss += loss.item() * y.size(0)
        n += y.size(0)
    return total_loss / max(1, n)


@torch.no_grad()
def eval_epoch(model: nn.Module, loader: DataLoader, *, device: torch.device) -> float:
    """Evaluate for one epoch."""
    model.eval()
    total_loss = 0.0
    n = 0
    use_cuda = device.type == "cuda"
    for batch in loader:
        y = batch["y"].to(device, non_blocking=use_cuda)
        x = batch["x"].to(device, non_blocking=use_cuda)
        u = batch["u"].to(device, non_blocking=use_cuda) if "u" in batch else None
        with torch.amp.autocast("cuda", enabled=use_cuda):
            x_hat = model(y, u)
            loss = nn.functional.mse_loss(x_hat, x)
        total_loss += loss.item() * y.size(0)
        n += y.size(0)
    return total_loss / max(1, n)


@torch.no_grad()
def eval_with_kf(
    model: nn.Module, loader: DataLoader, lds: LDS, *, device: torch.device
) -> Tuple[float, float]:
    """Compare Transformer vs. ground truth and Kalman Filter vs. ground truth.

    Returns (mse_transformer, mse_kf) averaged over all batches/timesteps.
    """
    model.eval()
    use_cuda = device.type == "cuda"
    A, H, Bmat = lds.A, lds.H, lds.B
    # Use correct noise covariance matrices (q and r are already standard deviations)
    Q = (lds.q**2) * torch.eye(A.size(0), device=device)
    R = (lds.r**2) * torch.eye(H.size(0), device=device)

    mse_tr_total = 0.0
    mse_kf_total = 0.0
    count = 0

    for batch in loader:
        y = batch["y"].to(device, non_blocking=use_cuda)
        x_true = batch["x"].to(device, non_blocking=use_cuda)
        u = batch["u"].to(device, non_blocking=use_cuda) if "u" in batch else None
        with torch.amp.autocast("cuda", enabled=use_cuda):
            x_tr = model(y, u)
        x_kf = kalman_filter_sequence(A, H, Q, R, y, U=u, Bmat=Bmat)
        mse_tr_total += nn.functional.mse_loss(x_tr, x_true, reduction="sum").item()
        mse_kf_total += nn.functional.mse_loss(x_kf, x_true, reduction="sum").item()
        count += x_true.numel()

    return mse_tr_total / count, mse_kf_total / count


def visualize_1d_results(
    lds: LDS,
    model: nn.Module,
    device: torch.device,
    save_path: Optional[str] = None,
    model_type: str = "transformer",
):
    """Visualize results for 1D state space."""
    if lds.A.size(0) != 1:
        print(f"Visualization only supported for 1D state space, got {lds.A.size(0)}D")
        return

    model.eval()
    # Generate a single trajectory
    X_true, Y, U = lds.generate(horizon=50, batch_size=1, noisy=True)
    X_true = X_true.squeeze(0)  # (T, 1)
    Y = Y.squeeze(0)  # (T, n_obs)
    U = U.squeeze(0) if U is not None else None  # (T, n_ctrl) or None

    # Get transformer predictions
    with torch.no_grad():
        Y_batch = Y.unsqueeze(0).to(device)  # (1, T, n_obs)
        U_batch = U.unsqueeze(0).to(device) if U is not None else None
        X_tr = model(Y_batch, U_batch).squeeze(0)  # (T, 1)

    # Get Kalman filter predictions
    A, H, Bmat = lds.A, lds.H, lds.B
    Q = (lds.q**2) * torch.eye(A.size(0), device=device)
    R = (lds.r**2) * torch.eye(H.size(0), device=device)
    X_kf = kalman_filter_sequence(A, H, Q, R, Y_batch, U=U_batch, Bmat=Bmat).squeeze(0)

    # Convert to numpy for plotting
    X_true_np = X_true.cpu().numpy().flatten()
    X_tr_np = X_tr.cpu().numpy().flatten()
    X_kf_np = X_kf.cpu().numpy().flatten()
    Y_np = Y.cpu().numpy()

    # Create the plot
    plt.figure(figsize=(12, 8))

    # Plot state estimates
    plt.subplot(2, 1, 1)
    steps = np.arange(len(X_true_np))
    plt.plot(steps, X_true_np, "k-", label="Ground Truth", linewidth=2)
    plt.plot(steps, X_tr_np, "b--", label=model_type.title(), linewidth=2)
    plt.plot(steps, X_kf_np, "r:", label="Kalman Filter", linewidth=2)
    plt.xlabel("Time Step")
    plt.ylabel("State Value")
    plt.title("State Estimation Comparison (1D)")
    plt.legend()
    plt.grid(True, alpha=0.3)

    # Plot observations
    plt.subplot(2, 1, 2)
    for i in range(Y_np.shape[1]):
        plt.plot(steps, Y_np[:, i], label=f"Measurement Channel {i+1}", alpha=0.7)
    plt.xlabel("Time Step")
    plt.ylabel("Measurement Value")
    plt.title("Noisy Measurements (Observations)")
    plt.legend()
    plt.grid(True, alpha=0.3)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        print(f"Visualization saved to {save_path}")
    else:
        plt.show()


def train_model(cfg: ExperimentConfig, model_class=None) -> Tuple[nn.Module, LDS, dict]:
    """Train the transformer model with the given configuration."""
    # Set random seed
    set_seed(cfg.training.seed)

    # Create output directory
    model_type = "transformer" if hasattr(cfg.model, "d_model") else "gru"
    output_dir = create_output_directory(model_type, cfg.lds.n_state, cfg.lds.n_obs)

    # Update model save path to use output directory
    if cfg.training.save_model:
        cfg.training.model_save_path = os.path.join(output_dir, "model.pt")

    # Get device
    device = cfg.lds.device or torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )
    cfg.lds.device = device

    # Create LDS
    lds = LDS.random(cfg.lds, seed=cfg.training.seed)

    # Create datasets
    train_ds = LDSSequenceDataset(
        lds,
        num_traj=cfg.training.num_train_traj,
        horizon=cfg.lds.horizon,
        random_controls=True,
        control_std=cfg.lds.control_std,
        noisy=True,
    )
    val_ds = LDSSequenceDataset(
        lds,
        num_traj=cfg.training.num_val_traj,
        horizon=cfg.lds.horizon,
        random_controls=True,
        control_std=cfg.lds.control_std,
        noisy=True,
    )

    # Create data loaders
    pin = device.type == "cuda"
    train_loader = DataLoader(
        train_ds, batch_size=cfg.lds.batch_size, shuffle=True, pin_memory=pin
    )
    val_loader = DataLoader(
        val_ds, batch_size=cfg.lds.batch_size, shuffle=False, pin_memory=pin
    )

    # Create model
    if model_class is None:
        model_class = OneLayerTransformer

    model = model_class(
        n_obs=cfg.lds.n_obs,
        n_state=cfg.lds.n_state,
        n_ctrl=cfg.lds.n_ctrl,
        cfg=cfg.model,
    ).to(device)

    # Create optimizer and scaler
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=cfg.training.learning_rate,
        weight_decay=cfg.training.weight_decay,
    )
    scaler = torch.amp.GradScaler("cuda", enabled=(device.type == "cuda"))

    # Training loop
    best_val = float("inf")
    patience_counter = 0
    training_history = {"train_loss": [], "val_loss": [], "best_val": float("inf")}

    print(
        f"Training on {device} with {cfg.training.num_train_traj} training trajectories"
    )
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
    print()
    print("=" * 80)
    print("TRAINING PROGRESS")
    print("=" * 80)
    print(
        f"{'Epoch':<6} | {'Train MSE':<10} | {'Val MSE':<10} | {'Best Val MSE':<12} | {'Status':<15}"
    )
    print("-" * 80)

    for epoch in range(1, cfg.training.max_epochs + 1):
        tr_loss = train_epoch(
            model,
            train_loader,
            optimizer,
            scaler,
            device=device,
            grad_clip=cfg.training.grad_clip,
        )
        val_loss = eval_epoch(model, val_loader, device=device)

        training_history["train_loss"].append(tr_loss)
        training_history["val_loss"].append(val_loss)

        if val_loss < best_val:
            best_val = val_loss
            patience_counter = 0
            training_history["best_val"] = best_val
            status = "New best!"
            if cfg.training.save_model:
                torch.save(
                    {
                        "model": model.state_dict(),
                        "cfg": cfg,
                        "training_history": training_history,
                    },
                    cfg.training.model_save_path,
                )
        else:
            patience_counter += 1
            status = f"Patience: {patience_counter}/{cfg.training.patience}"

        print(
            f"{epoch:6d} | {tr_loss:10.6f} | {val_loss:10.6f} | {best_val:12.6f} | {status:<15}"
        )

        # Early stopping
        if patience_counter >= cfg.training.patience:
            print("-" * 80)
            print(f"Early stopping at epoch {epoch} (patience exceeded)")
            break

    # Final evaluation
    print("-" * 80)
    print("FINAL EVALUATION")
    print("-" * 80)
    tr_mse, kf_mse = eval_with_kf(model, val_loader, lds, device=device)
    print(f"Validation MSE vs Ground Truth:")
    print(f"  Transformer: {tr_mse:.6f}")
    print(f"  Kalman Filter (oracle): {kf_mse:.6f}")
    print(f"  Performance ratio: {tr_mse/kf_mse:.3f}x")
    print("=" * 80)

    # Save training parameters
    save_training_parameters(cfg, output_dir, training_history)

    # Visualization for 1D case
    if cfg.training.enable_visualization and cfg.lds.n_state == 1:
        viz_path = os.path.join(output_dir, "state_estimation_comparison.png")
        visualize_1d_results(
            lds, model, device, save_path=viz_path, model_type=model_type
        )

    return model, lds, training_history


def load_trained_model(
    model_path: str, device: torch.device
) -> Tuple[nn.Module, LDS, dict]:
    """Load a trained model from checkpoint."""
    checkpoint = torch.load(model_path, map_location=device, weights_only=False)
    cfg = checkpoint["cfg"]
    training_history = checkpoint.get("training_history", {})

    # Recreate LDS
    lds = LDS.random(cfg.lds, seed=cfg.training.seed)

    # Recreate model
    model = OneLayerTransformer(
        n_obs=cfg.lds.n_obs,
        n_state=cfg.lds.n_state,
        n_ctrl=cfg.lds.n_ctrl,
        cfg=cfg.model,
    ).to(device)

    # Load weights
    model.load_state_dict(checkpoint["model"])
    model.eval()

    return model, lds, training_history
