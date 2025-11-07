"""
Linear system experiment: Compare multiple state estimation models.

Generates linear trajectories from a single system and compares:
- OneLayerTransformer (trainable transformer)
- KalmanFilterExact (oracle baselines: steady-state & time-varying)
- GRUStateEstimator (RNN baseline)
- MambaStateEstimator & Mamba2StateEstimator (selective SSMs)

Outputs: test MSE rankings plot and optional 1D time-series visualization.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Subset
import matplotlib.pyplot as plt
from tqdm import tqdm

# Repo root on path
sys.path.insert(0, str(Path(__file__).parent.parent))

from data import (
    get_device,
    set_seed,
    SimpleLinearSystemDataset,
)
from models import (
    OneLayerTransformer,
    GRUStateEstimator,
    MambaStateEstimator,
    Mamba2StateEstimator,
    kalman_filter_sequence,
)
from models.one_layer_transformer import KalmanFilterExact
from config import OneLayerTransformerConfig, GRUConfig, MambaConfig, Mamba2Config


def train_model(
    model: nn.Module,
    train_loader,
    val_loader,
    device,
    epochs: int,
    lr: float,
    patience: int,
):
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=2
    )

    best_val, patience_ctr = float("inf"), 0
    history = {"train": [], "val": []}

    for epoch in tqdm(range(epochs), desc="Training", leave=True):
        model.train()
        train_loss, train_count = 0.0, 0
        for batch in tqdm(
            train_loader, desc=f"  Epoch {epoch+1}/{epochs} - Train", leave=False
        ):
            y, x, u = batch["y"].to(device), batch["x"].to(device), batch.get("u")
            u = u.to(device) if u is not None else None
            optimizer.zero_grad()
            loss = nn.functional.mse_loss(model(y, u), x)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            train_loss += loss.item() * y.size(0)
            train_count += y.size(0)

        model.eval()
        val_loss, val_count = 0.0, 0
        with torch.no_grad():
            for batch in tqdm(
                val_loader, desc=f"  Epoch {epoch+1}/{epochs} - Val", leave=False
            ):
                y, x, u = batch["y"].to(device), batch["x"].to(device), batch.get("u")
                u = u.to(device) if u is not None else None
                val_loss += nn.functional.mse_loss(
                    model(y, u), x, reduction="sum"
                ).item()
                val_count += x.numel()

        train_mse = train_loss / max(1, train_count)
        val_mse = val_loss / max(1, val_count)
        history["train"].append(train_mse)
        history["val"].append(val_mse)
        scheduler.step(val_mse)

        if val_mse < best_val:
            best_val, patience_ctr = val_mse, 0
        else:
            patience_ctr += 1

        if patience_ctr >= patience:
            break

    return history, best_val


def evaluate_models(
    models_dict: dict[str, nn.Module], test_loader, test_dataset, device, cfg
):
    results = {}

    # Evaluate all neural models
    for name, model in models_dict.items():
        model.eval()
        total, count = 0.0, 0
        with torch.no_grad():
            for batch in tqdm(test_loader, desc=f"Eval {name}", leave=False):
                y, x, u = batch["y"].to(device), batch["x"].to(device), batch.get("u")
                u = u.to(device) if u is not None else None
                x_hat = model(y, u)[:, cfg.warmup_ignore :, :]
                x_eff = x[:, cfg.warmup_ignore :, :]
                total += nn.functional.mse_loss(x_hat, x_eff, reduction="sum").item()
                count += x_eff.numel()
        results[name] = total / max(1, count)

    # Kalman Filter (oracle)
    Q = torch.eye(cfg.n_state, device=device) * (cfg.process_noise_std**2)
    R = torch.eye(cfg.n_obs, device=device) * (cfg.meas_noise_std**2)
    A, H = test_dataset.A.to(device), test_dataset.H.to(device)
    B = test_dataset.B.to(device) if test_dataset.B is not None else None

    total, count = 0.0, 0
    with torch.no_grad():
        for batch in tqdm(test_loader, desc="Eval Kalman Filter", leave=False):
            y, x, u = batch["y"].to(device), batch["x"].to(device), batch.get("u")
            u = u.to(device) if u is not None else None
            x_hat = kalman_filter_sequence(A, H, Q, R, y, U=u, Bmat=B)[
                :, cfg.warmup_ignore :, :
            ]
            x_eff = x[:, cfg.warmup_ignore :, :]
            total += nn.functional.mse_loss(x_hat, x_eff, reduction="sum").item()
            count += x_eff.numel()
    results["Kalman Filter"] = total / max(1, count)

    return results


def visualize_rankings(results: dict[str, float], cfg) -> str:
    sorted_items = sorted(results.items(), key=lambda kv: kv[1])
    names = [k for k, _ in sorted_items]
    mses = [v for _, v in sorted_items]

    colors = plt.cm.viridis(np.linspace(0, 1, len(names)))
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.barh(range(len(names)), mses, color=colors)
    ax.set_xlabel("Mean Squared Error", fontsize=12, fontweight="bold")
    ax.set_yticks(range(len(names)))
    ax.set_yticklabels(names)
    ax.set_title("Model Rankings - Linear System", fontsize=14, fontweight="bold")
    ax.invert_yaxis()
    for i, mse in enumerate(mses):
        ax.text(mse, i, f"  {mse:.6f}", va="center", fontsize=10, fontweight="bold")
    plt.tight_layout()
    Path("images").mkdir(exist_ok=True)
    out_path = f"images/results_linear_{cfg.n_state}d.png"
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
    return out_path


def visualize_1d_trajectories(
    models_dict: dict[str, nn.Module], test_loader, test_dataset, device, cfg
) -> str | None:
    if cfg.n_state != 1:
        return None
    batch = next(iter(test_loader))
    y, x, u = batch["y"].to(device), batch["x"].to(device), batch.get("u")
    u = u.to(device) if u is not None else None

    preds = {}
    with torch.no_grad():
        for name, model in models_dict.items():
            model.eval()
            preds[name] = (
                model(y[0:1], u[0:1] if u is not None else None)[0, :, 0].cpu().numpy()
            )

        Q = torch.eye(cfg.n_state, device=device) * (cfg.process_noise_std**2)
        R = torch.eye(cfg.n_obs, device=device) * (cfg.meas_noise_std**2)
        A, H = test_dataset.A.to(device), test_dataset.H.to(device)
        B = test_dataset.B.to(device) if test_dataset.B is not None else None
        preds["Kalman Filter"] = (
            kalman_filter_sequence(
                A, H, Q, R, y[0:1], U=(u[0:1] if u is not None else None), Bmat=B
            )[0, :, 0]
            .cpu()
            .numpy()
        )

    fig, ax = plt.subplots(figsize=(12, 4))
    ax.plot(
        np.arange(cfg.horizon),
        x[0, :, 0].cpu().numpy(),
        "k-",
        label="Ground Truth",
        linewidth=2.5,
        alpha=0.7,
    )
    for i, (name, pred) in enumerate(preds.items()):
        ax.plot(
            np.arange(cfg.horizon),
            pred,
            "-",
            label=name,
            color=plt.cm.tab10.colors[i % 10],
            linewidth=2,
        )
    ax.set_xlabel("Time Step", fontsize=11, fontweight="bold")
    ax.set_ylabel("State", fontsize=11, fontweight="bold")
    ax.legend(loc="best", fontsize=9)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    Path("images").mkdir(exist_ok=True)
    out_path = f"images/state_1d_linear_{cfg.n_state}d.png"
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
    return out_path


def main():
    parser = argparse.ArgumentParser(
        description="Linear system: compare transformer, GRU, Mamba vs KF oracle"
    )
    parser.add_argument("--n_state", type=int, default=4)
    parser.add_argument("--n_obs", type=int, default=2)
    parser.add_argument("--n_ctrl", type=int, default=0)
    parser.add_argument("--horizon", type=int, default=100)
    parser.add_argument("--num_traj_train", type=int, default=20000)
    parser.add_argument("--process_noise_std", type=float, default=0.1)
    parser.add_argument("--meas_noise_std", type=float, default=0.1)
    parser.add_argument("--control_std", type=float, default=1.0)
    parser.add_argument("--epochs", type=int, default=15)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--learning_rate", type=float, default=2e-3)
    parser.add_argument("--patience", type=int, default=5)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--warmup_ignore", type=int, default=0)
    parser.add_argument(
        "--device", type=str, default="auto", choices=["auto", "cpu", "cuda"]
    )

    args = parser.parse_args()

    class Cfg:
        pass

    cfg = Cfg()
    for k, v in vars(args).items():
        setattr(cfg, k, v)

    set_seed(cfg.seed)
    device = get_device() if cfg.device == "auto" else torch.device(cfg.device)

    # Build dataset from a single system, then split 5:2:2 (train:val:test)
    num_val = int(cfg.num_traj_train * 2 / 5)
    num_test = int(cfg.num_traj_train * 2 / 5)
    total_traj = cfg.num_traj_train + num_val + num_test

    full_ds = SimpleLinearSystemDataset(
        n_state=cfg.n_state,
        n_obs=cfg.n_obs,
        n_ctrl=cfg.n_ctrl,
        num_traj=total_traj,
        horizon=cfg.horizon,
        process_noise_std=cfg.process_noise_std,
        meas_noise_std=cfg.meas_noise_std,
        random_controls=cfg.n_ctrl > 0,
        control_std=cfg.control_std,
        noisy=True,
        device=device,
        seed=None,
    )

    train_ds = Subset(full_ds, range(cfg.num_traj_train))
    val_ds = Subset(full_ds, range(cfg.num_traj_train, cfg.num_traj_train + num_val))
    test_ds = Subset(full_ds, range(cfg.num_traj_train + num_val, total_traj))
    # attach system matrices for KF
    test_ds.A, test_ds.H, test_ds.B = full_ds.A, full_ds.H, full_ds.B

    train_loader = DataLoader(
        train_ds,
        batch_size=cfg.batch_size,
        shuffle=True,
        pin_memory=(device.type == "cuda"),
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=cfg.batch_size,
        shuffle=False,
        pin_memory=(device.type == "cuda"),
    )
    test_loader = DataLoader(
        test_ds,
        batch_size=cfg.batch_size,
        shuffle=False,
        pin_memory=(device.type == "cuda"),
    )

    # Print experiment metadata
    print("\n" + "=" * 80)
    print("EXPERIMENT SETUP")
    print("=" * 80)
    print(f"\nDataset: SimpleLinearSystemDataset (single system, 5:2:2 split)")
    print(f"  State dimension: {cfg.n_state}")
    print(f"  Observation dimension: {cfg.n_obs}")
    print(f"  Control dimension: {cfg.n_ctrl}")
    print(f"  Horizon length: {cfg.horizon}")
    print(f"  Training trajectories: {cfg.num_traj_train}")
    print(f"  Validation trajectories: {num_val}")
    print(f"  Test trajectories: {num_test}")
    print(f"  Process noise std: {cfg.process_noise_std:.4f}")
    print(f"  Measurement noise std: {cfg.meas_noise_std:.4f}")
    print(f"\nTraining configuration:")
    print(f"  Device: {device}")
    print(f"  Epochs: {cfg.epochs}")
    print(f"  Batch size: {cfg.batch_size}")
    print(f"  Learning rate: {cfg.learning_rate}")
    print(f"  Patience: {cfg.patience}")
    print(f"  Warmup ignore: {cfg.warmup_ignore}")
    print(f"  Seed: {cfg.seed}")
    print("=" * 80)

    # Instantiate and train models
    models_dict = {}
    print(f"\n{'='*80}")
    print("MODELS TO TRAIN")
    print("=" * 80)

    model_configs = [
        (
            "OneLayerTransformer",
            OneLayerTransformer,
            OneLayerTransformerConfig(horizon=cfg.horizon),
        ),
        ("GRUStateEstimator", GRUStateEstimator, GRUConfig()),
        ("Mamba", MambaStateEstimator, MambaConfig()),
        ("Mamba2", Mamba2StateEstimator, Mamba2Config()),
    ]

    # Print model info
    for i, (name, cls, cfg_cls) in enumerate(model_configs, 1):
        temp_model = cls(
            n_obs=cfg.n_obs, n_state=cfg.n_state, n_ctrl=cfg.n_ctrl, cfg=cfg_cls
        ).to(device)
        n_params = sum(p.numel() for p in temp_model.parameters() if p.requires_grad)
        print(f"  {i}. {name}: {n_params:,} trainable parameters")
        del temp_model

    print("=" * 80)

    # Train models
    for name, cls, cfg_cls in model_configs:
        model = cls(
            n_obs=cfg.n_obs, n_state=cfg.n_state, n_ctrl=cfg.n_ctrl, cfg=cfg_cls
        ).to(device)
        print(f"\n=== Training {name} ===")
        _, best_val = train_model(
            model,
            train_loader,
            val_loader,
            device,
            cfg.epochs,
            cfg.learning_rate,
            cfg.patience,
        )
        print(f"Best Val MSE ({name}): {best_val:.6f}")
        models_dict[name] = model

    # Add oracle KF baselines (what OneLayerTransformer aims to approximate)
    # These directly implement KF equations, NOT transformer architectures
    Q = torch.eye(cfg.n_state, device=device) * (cfg.process_noise_std**2)
    R = torch.eye(cfg.n_obs, device=device) * (cfg.meas_noise_std**2)
    A_mat, H_mat = full_ds.A.to(device), full_ds.H.to(device)

    # Oracle baseline: Steady-state Kalman gain (what OLT aims to learn)
    kf_oracle_steady = KalmanFilterExact(
        cfg.n_obs, cfg.n_state, A_mat, H_mat, Q, R, use_time_varying=False
    ).to(device)
    models_dict["OLT-Oracle-Steady"] = kf_oracle_steady

    # Oracle baseline: Time-varying Kalman gain (theoretical upper bound)
    kf_oracle_tv = KalmanFilterExact(
        cfg.n_obs,
        cfg.n_state,
        A_mat,
        H_mat,
        Q,
        R,
        use_time_varying=True,
        horizon=cfg.horizon,
    ).to(device)
    models_dict["OLT-Oracle-TimeVarying"] = kf_oracle_tv

    results = evaluate_models(models_dict, test_loader, test_ds, device, cfg)

    print("\n" + "=" * 80)
    print("TEST RESULTS")
    print("=" * 80)
    print(f"\nDataset: {cfg.n_state}D state, {cfg.n_obs}D obs, horizon={cfg.horizon}")
    print(
        f"Noise: process={cfg.process_noise_std:.3f}, measurement={cfg.meas_noise_std:.3f}"
    )
    print(f"Test set: {len(test_ds)} trajectories")
    if cfg.warmup_ignore > 0:
        print(f"Evaluation: ignoring first {cfg.warmup_ignore} timesteps")

    print(f"\n{'Model':<30} | {'Test MSE':<12} | {'Notes':<30}")
    print("-" * 80)

    sorted_results = sorted(results.items(), key=lambda kv: kv[1])
    for rank, (name, mse) in enumerate(sorted_results, 1):
        if "OLT-Oracle" in name:
            note = "Oracle KF (OLT target)"
        elif "Kalman" in name:
            note = "Oracle (batched KF)"
        elif "OneLayerTransformer" in name:
            note = "Learned (approximates oracle)"
        elif "GRU" in name:
            note = "Baseline RNN"
        elif "Mamba2" in name:
            note = "Selective SSM (SSD)"
        elif "Mamba" in name:
            note = "Selective SSM"
        else:
            note = ""
        print(f"{rank}. {name:<30} | {mse:<12.6f} | {note}")

    print("=" * 80)

    rank_path = visualize_rankings(results, cfg)
    ts1d_path = visualize_1d_trajectories(
        models_dict, test_loader, test_ds, device, cfg
    )

    print(f"\nSaved: {rank_path}")
    if ts1d_path:
        print(f"Saved: {ts1d_path}")


if __name__ == "__main__":
    main()
