"""
Training script for Transformer Kalman Filter experiment.

This script automatically discovers all neural network models and filters,
trains the models, and compares their performance with visualization.
"""

import argparse
import inspect
import sys
from pathlib import Path
import warnings

warnings.filterwarnings("ignore", category=UserWarning)

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm

from data import (
    get_device,
    set_seed,
    SimpleLinearSystemDataset,
    SimpleNonlinearSystemDataset,
)
from models import __all__ as models_all
from config import OneLayerTransformerConfig, FilterformerConfig, GRUConfig, LTCConfig


def discover_neural_models():
    """Automatically discover all neural network model classes."""
    import models

    neural_models = {}
    for name in models_all:
        obj = getattr(models, name)
        if inspect.isclass(obj) and issubclass(obj, nn.Module) and obj != nn.Module:
            neural_models[name] = obj

    return neural_models


def print_metadata(dataset_train, dataset_val, models, device, cfg):
    """Print metadata information about dataset and models."""
    print("\n" + "=" * 80)
    print("EXPERIMENT METADATA")
    print("=" * 80)
    print(f"\nDataset: {dataset_train.__class__.__name__}")
    print(
        f"  Dimensions: state={cfg.n_state}, obs={cfg.n_obs}, ctrl={cfg.n_ctrl}, horizon={cfg.horizon}"
    )
    print(
        f"  Training: {len(dataset_train)} ({cfg.num_distinct_systems_train}×{cfg.num_traj_per_system})"
    )
    print(
        f"  Validation: {len(dataset_val)} ({cfg.num_distinct_systems_val}×{cfg.num_traj_per_system})"
    )
    print(
        f"  Noise: process={cfg.process_noise_std:.4f}, meas={cfg.meas_noise_std:.4f}"
    )
    print(f"\nModels ({len(models)}):")
    for i, (name, model) in enumerate(models.items(), 1):
        n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print(f"  {i}. {name}: {n_params:,} params")
    print(
        f"\nTraining: device={device}, epochs={cfg.epochs}, batch={cfg.batch_size}, lr={cfg.learning_rate}"
    )
    print(f"  Early stop patience={cfg.patience}, seed={cfg.seed}")
    print("\nFilters: Kalman Filter (Extended Kalman/Particle for nonlinear)")
    print("=" * 80 + "\n")


def train_single_model(model, train_loader, val_loader, device, cfg):
    """Train a single neural network model."""
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=cfg.learning_rate, weight_decay=1e-4
    )
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=2
    )
    best_val_loss, patience_counter, history = float("inf"), 0, {"train": [], "val": []}
    epoch_pbar = tqdm(range(cfg.epochs), desc="Training", leave=True)

    for epoch in epoch_pbar:
        # Training
        model.train()
        train_loss, train_count = 0.0, 0
        for batch in tqdm(
            train_loader, desc=f"  Epoch {epoch+1}/{cfg.epochs} - Train", leave=False
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

        # Validation
        model.eval()
        val_loss, val_count = 0.0, 0
        with torch.no_grad():
            for batch in tqdm(
                val_loader, desc=f"  Epoch {epoch+1}/{cfg.epochs} - Val", leave=False
            ):
                y, x, u = batch["y"].to(device), batch["x"].to(device), batch.get("u")
                u = u.to(device) if u is not None else None
                val_loss += nn.functional.mse_loss(model(y, u), x).item() * y.size(0)
                val_count += y.size(0)

        train_mse, val_mse = train_loss / max(1, train_count), val_loss / max(
            1, val_count
        )
        history["train"].append(train_mse)
        history["val"].append(val_mse)
        scheduler.step(val_mse)

        if val_mse < best_val_loss:
            best_val_loss, patience_counter = val_mse, 0
        else:
            patience_counter += 1

        epoch_pbar.set_postfix(
            {
                "Train MSE": f"{train_mse:.6f}",
                "Val MSE": f"{val_mse:.6f}",
                "Best": f"{best_val_loss:.6f}",
                "Status": "Early Stop" if patience_counter >= cfg.patience else "",
            }
        )

        if patience_counter >= cfg.patience:
            break

    epoch_pbar.close()
    return history, best_val_loss


def evaluate_all_models(
    models_dict, test_loader, test_dataset, dataset_type, device, cfg
):
    """Evaluate all models (neural networks + filters) on test set."""
    results = {}

    print("\nEvaluating models on test set...")

    # Evaluate neural network models
    for name, model in models_dict.items():
        print(f"  Evaluating {name}...")
        model.eval()
        total_mse, count = 0.0, 0

        with torch.no_grad():
            for batch in tqdm(test_loader, desc=f"    {name}", leave=False):
                y, x, u = batch["y"].to(device), batch["x"].to(device), batch.get("u")
                u = u.to(device) if u is not None else None
                x_hat = model(y, u)
                total_mse += nn.functional.mse_loss(x_hat, x, reduction="sum").item()
                count += x.numel()

        results[name] = total_mse / max(1, count)

    # Evaluate Kalman Filter for linear systems
    if dataset_type == "linear":
        print("  Evaluating Kalman Filter...")
        from models import kalman_filter_sequence

        total_mse_kf = 0.0
        count = 0
        Q = torch.eye(cfg.n_state) * (cfg.process_noise_std**2)
        R = torch.eye(cfg.n_obs) * (cfg.meas_noise_std**2)

        with torch.no_grad():
            for batch_idx, batch in enumerate(
                tqdm(test_loader, desc="    Kalman Filter", leave=False)
            ):
                y = batch["y"].to(device)
                x = batch["x"].to(device)
                u = batch.get("u")
                batch_size = y.shape[0]

                start_idx = batch_idx * cfg.batch_size
                systems_in_batch = set()
                for i in range(batch_size):
                    systems_in_batch.add(test_dataset.get_system_idx(start_idx + i))

                for system_idx in systems_in_batch:
                    A_sys = test_dataset.A_all[system_idx].to(device)
                    H_sys = test_dataset.H_all[system_idx].to(device)
                    B_sys = (
                        test_dataset.B_all[system_idx].to(device)
                        if test_dataset.B_all is not None
                        else None
                    )

                    traj_mask = [
                        i
                        for i in range(batch_size)
                        if test_dataset.get_system_idx(start_idx + i) == system_idx
                    ]
                    if not traj_mask:
                        continue

                    y_sys, x_sys = y[traj_mask], x[traj_mask]
                    u_sys = u[traj_mask] if u is not None else None

                    x_hat_kf = kalman_filter_sequence(
                        A_sys, H_sys, Q, R, y_sys, u_sys, Bmat=B_sys
                    )
                    total_mse_kf += nn.functional.mse_loss(
                        x_hat_kf, x_sys, reduction="sum"
                    ).item()
                    count += x_sys.numel()

        results["Kalman Filter"] = total_mse_kf / max(1, count)

    # Evaluate Particle Filter for nonlinear systems
    if dataset_type == "nonlinear":
        print("  Evaluating Particle Filter...")
        from models import particle_filter_sequence

        # Helper function to create f and g from system parameters
        def make_f_g_system(
            sys_idx,
            f_params_all,
            g_params_all,
            g_proj_all,
            n_state,
            n_obs,
            n_ctrl,
            device,
        ):
            f_params = {
                k: v[sys_idx] if v.dim() == 1 else v[sys_idx]
                for k, v in f_params_all.items()
            }
            f_params["mix_matrix"] = f_params_all["mix_matrix"][sys_idx]
            if "control_matrix" in f_params_all:
                f_params["control_matrix"] = f_params_all["control_matrix"][sys_idx]
            g_params = {k: v[sys_idx] for k, v in g_params_all.items()}
            g_proj = g_proj_all[sys_idx]

            def f(x, u):
                # x: (B*N, n_state), u: (B*N, n_ctrl) or None
                x_next = f_params["linear"] * x
                if f_params["tanh"] > 0:
                    x_next += f_params["tanh"] * torch.tanh(x)
                if f_params["sin"] > 0:
                    x_next += f_params["sin"] * torch.sin(x)
                if f_params["cos"] > 0:
                    x_next += f_params["cos"] * torch.cos(x)
                if f_params["quad"] > 0:
                    x_next += f_params["quad"] * torch.tanh(x * x)
                if f_params["cubic"] > 0:
                    x_next += f_params["cubic"] * torch.tanh(x * x * x)
                if f_params["mix"] > 0:
                    x_next += f_params["mix"] * (x @ f_params["mix_matrix"].T)
                if u is not None and n_ctrl > 0:
                    if "control_matrix" in f_params:
                        Bu = f_params["control"] * (u @ f_params["control_matrix"].T)
                    else:
                        Bu = f_params["control"] * u
                    if Bu.size(-1) < n_state:
                        pad = torch.zeros(
                            Bu.size(0), n_state - Bu.size(-1), device=device
                        )
                        Bu = torch.cat([Bu, pad], dim=-1)
                    elif Bu.size(-1) > n_state:
                        Bu = Bu[:, :n_state]
                    x_next = x_next + Bu
                return x_next

            def g(x):
                # x: (B*N, n_state) -> (B*N, n_obs)
                g_active = {
                    k: g_params[k] > 0
                    for k in ["linear", "sin", "cos", "tanh", "square"]
                }
                features = []
                if g_active["linear"]:
                    features.append(g_params["linear"] * x)
                if g_active["sin"]:
                    features.append(g_params["sin"] * torch.sin(x))
                if g_active["cos"]:
                    features.append(g_params["cos"] * torch.cos(x))
                if g_active["tanh"]:
                    features.append(g_params["tanh"] * torch.tanh(x))
                if g_active["square"]:
                    features.append(g_params["square"] * torch.tanh(x * x))
                feats = torch.cat(features, dim=-1)
                return feats @ g_proj

            return f, g

        total_mse_pf = 0.0
        count = 0

        with torch.no_grad():
            for batch_idx, batch in enumerate(
                tqdm(test_loader, desc="    Particle Filter", leave=False)
            ):
                y = batch["y"].to(device)
                x = batch["x"].to(device)
                u = batch.get("u")
                batch_size = y.shape[0]

                start_idx = batch_idx * cfg.batch_size
                systems_in_batch = set()
                for i in range(batch_size):
                    systems_in_batch.add(test_dataset.get_system_idx(start_idx + i))

                for system_idx in systems_in_batch:
                    f_sys, g_sys = make_f_g_system(
                        system_idx,
                        test_dataset.f_params_all,
                        test_dataset.g_params_all,
                        test_dataset.g_proj_all,
                        cfg.n_state,
                        cfg.n_obs,
                        cfg.n_ctrl,
                        device,
                    )

                    traj_mask = [
                        i
                        for i in range(batch_size)
                        if test_dataset.get_system_idx(start_idx + i) == system_idx
                    ]
                    if not traj_mask:
                        continue

                    y_sys, x_sys = y[traj_mask], x[traj_mask]
                    u_sys = u[traj_mask] if u is not None else None

                    x_hat_pf = particle_filter_sequence(
                        f_sys,
                        g_sys,
                        cfg.process_noise_std,
                        cfg.meas_noise_std,
                        y_sys,
                        num_particles=1024,
                        U=u_sys,
                    )
                    total_mse_pf += nn.functional.mse_loss(
                        x_hat_pf, x_sys, reduction="sum"
                    ).item()
                    count += x_sys.numel()

        results["Particle Filter"] = total_mse_pf / max(1, count)

    return results


def visualize_1d_states(models_dict, test_loader, test_dataset, device, cfg):
    """Visualize state estimation over time for 1D systems."""
    print("\nGenerating 1D state estimation visualization...")

    test_batch = next(iter(test_loader))
    y, x_true, u = (
        test_batch["y"].to(device),
        test_batch["x"].to(device),
        test_batch.get("u"),
    )
    u = u.to(device) if u is not None else None

    y_sample, x_true_sample, u_sample = (
        y[0],
        x_true[0],
        (u[0] if u is not None else None),
    )

    predictions = {}
    with torch.no_grad():
        for name, model in models_dict.items():
            model.eval()
            y_input = y_sample.unsqueeze(0)
            u_input = u_sample.unsqueeze(0) if u_sample is not None else None
            predictions[name] = model(y_input, u_input)[0, :, 0].cpu().numpy()

    if cfg.system_type == "linear":
        from models import kalman_filter_sequence

        with torch.no_grad():
            system_idx = test_dataset.get_system_idx(0)
            A, H = test_dataset.A_all[system_idx].to(device), test_dataset.H_all[
                system_idx
            ].to(device)
            Q, R = torch.eye(cfg.n_state) * (cfg.process_noise_std**2), torch.eye(
                cfg.n_obs
            ) * (cfg.meas_noise_std**2)
            B = (
                test_dataset.B_all[system_idx].to(device)
                if test_dataset.B_all is not None
                else None
            )
            y_input, u_input = y_sample.unsqueeze(0), (
                u_sample.unsqueeze(0) if u_sample is not None else None
            )
            predictions["Kalman Filter"] = (
                kalman_filter_sequence(A, H, Q, R, y_input, u_input, Bmat=B)[0, :, 0]
                .cpu()
                .numpy()
            )

    # Create visualization
    time_steps = np.arange(cfg.horizon)
    x_true_plot = x_true_sample[:, 0].cpu().numpy()

    num_models = len(predictions)
    n_cols = 3
    n_rows = (num_models + n_cols - 1) // n_cols

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(18, 5 * n_rows))
    if n_rows == 1:
        axes = axes.reshape(1, -1)
    axes = axes.flatten()

    colors_list = plt.cm.tab10.colors

    for idx, (name, x_pred) in enumerate(predictions.items()):
        ax, color = axes[idx], colors_list[idx % len(colors_list)]
        ax.plot(
            time_steps,
            x_true_plot,
            "k-",
            label="Ground Truth",
            linewidth=2.5,
            alpha=0.7,
        )
        ax.plot(
            time_steps, x_pred, "-", label=name, color=color, linewidth=2, alpha=0.8
        )
        ax.set_xlabel("Time Step", fontsize=11, fontweight="bold")
        ax.set_ylabel("State Value", fontsize=11, fontweight="bold")
        ax.set_title(name, fontsize=13, fontweight="bold")
        ax.legend(loc="best", fontsize=9)
        ax.grid(True, alpha=0.3)

    # Hide empty subplots
    for idx in range(len(predictions), len(axes)):
        axes[idx].axis("off")

    plt.tight_layout()

    # Save figure
    filename = f"state_estimation_1d_{cfg.system_type}.png"
    plt.savefig(filename, dpi=150, bbox_inches="tight")
    print(f"  Saved to: {filename}")
    plt.close()


def visualize_results(results_dict, training_histories, cfg):
    """Visualize state estimation errors for all models."""
    print("\nGenerating visualization...")

    # Sort results by MSE
    sorted_results = sorted(results_dict.items(), key=lambda x: x[1])

    models = [name for name, _ in sorted_results]
    mses = [mse for _, mse in sorted_results]

    # Create figure with subplots: training curves on top row, ranking on bottom
    fig = plt.figure(figsize=(16, 10))
    gs = fig.add_gridspec(2, 2, hspace=0.35, wspace=0.3)

    colors_list = plt.cm.tab10.colors
    for axis_type, ax_idx in [("train", 0), ("val", 1)]:
        ax = fig.add_subplot(gs[0, ax_idx])
        for i, (name, history_dict) in enumerate(training_histories.items()):
            if "history" in history_dict:
                epochs = range(1, len(history_dict["history"][axis_type]) + 1)
                ax.plot(
                    epochs,
                    history_dict["history"][axis_type],
                    label=name,
                    color=colors_list[i % len(colors_list)],
                    linestyle="-",
                    linewidth=2.5,
                    alpha=0.8,
                )
        ax.set_xlabel("Epoch", fontsize=12, fontweight="bold")
        ax.set_ylabel("Mean Squared Error", fontsize=12, fontweight="bold")
        ax.set_title(
            f"{axis_type.capitalize()} MSE Over Time", fontsize=14, fontweight="bold"
        )
        ax.legend(loc="best", fontsize=9)
        ax.grid(True, alpha=0.3)
        ax.set_yscale("log")

    # Model Rankings (bottom, spanning both columns)
    colors = plt.cm.viridis(np.linspace(0, 1, len(models)))
    rankings = range(1, len(models) + 1)
    ax_rank = fig.add_subplot(gs[1, :])
    ax_rank.barh(range(len(models)), mses, color=colors)
    ax_rank.set_xscale("log")
    ax_rank.set_xlabel("Mean Squared Error (log scale)", fontsize=12, fontweight="bold")
    ax_rank.set_ylabel("Model", fontsize=12, fontweight="bold")
    ax_rank.set_title(
        "Model Rankings - Test Performance", fontsize=14, fontweight="bold"
    )
    ax_rank.set_yticks(range(len(models)))
    ax_rank.set_yticklabels(models)
    ax_rank.invert_yaxis()  # Best model at top
    ax_rank.grid(True, alpha=0.3, axis="x")

    # Add value labels
    for i, (ranking, mse) in enumerate(zip(rankings, mses)):
        ax_rank.text(
            mse, i, f"  {mse:.2e}", va="center", fontsize=10, fontweight="bold"
        )

    plt.tight_layout()

    # Save figure
    filename = f"results_{cfg.system_type}_{cfg.n_state}d.png"
    plt.savefig(filename, dpi=150, bbox_inches="tight")
    print(f"  Saved to: {filename}")
    plt.close()


def print_results(training_histories, test_results):
    """Print final results summary."""
    print("\n" + "=" * 80)
    print("TRAINING RESULTS")
    print("=" * 80)

    print(f"\n{'Model':<30} | {'Best Val MSE':<15} | {'Test MSE':<15}")
    print("-" * 80)

    for name in training_histories.keys():
        best_val = training_histories[name]["best_val"]
        test_mse = test_results.get(name, "N/A")
        if isinstance(test_mse, float):
            print(f"{name:<30} | {best_val:<15.6f} | {test_mse:<15.6f}")
        else:
            print(f"{name:<30} | {best_val:<15.6f} | {test_mse:<15}")

    # Add filter results
    if "Kalman Filter" in test_results:
        print(
            f"{'Kalman Filter':<30} | {'N/A (oracle)':<15} | {test_results['Kalman Filter']:<15.6f}"
        )
    if "Particle Filter" in test_results:
        print(
            f"{'Particle Filter':<30} | {'N/A (oracle)':<15} | {test_results['Particle Filter']:<15.6f}"
        )

    print("=" * 80)


def create_model_instance(model_class, cfg, device):
    """Create an instance of a model class with appropriate configuration."""
    # Get signature to see what parameters it needs
    sig = inspect.signature(model_class.__init__)
    params = {}

    # Common parameters
    params["n_state"] = cfg.n_state
    params["n_obs"] = cfg.n_obs
    params["n_ctrl"] = cfg.n_ctrl

    # Use appropriate config from config.model_config
    model_name = model_class.__name__
    if "OneLayerTransformer" in model_name:
        if "cfg" in sig.parameters:
            params["cfg"] = OneLayerTransformerConfig()
    elif "Filterformer" in model_name:
        if "cfg" in sig.parameters:
            params["cfg"] = FilterformerConfig()
    elif "GRUStateEstimator" in model_name:
        if "cfg" in sig.parameters:
            params["cfg"] = GRUConfig()
    elif "LiquidTimeConstantModel" in model_name:
        config = LTCConfig()
        if "hidden_size" in sig.parameters:
            params["hidden_size"] = config.hidden_size
        if "num_layers" in sig.parameters:
            params["num_layers"] = config.num_layers
        if "dropout" in sig.parameters:
            params["dropout"] = config.dropout

    # Additional parameters
    if "output_covariance" in sig.parameters:
        params["output_covariance"] = False

    try:
        model = model_class(**params)
        return model.to(device)
    except Exception as e:
        print(f"Warning: Could not instantiate {model_class.__name__}: {e}")
        return None


def main():
    parser = argparse.ArgumentParser(
        description="Train and Compare State Estimation Models"
    )
    parser.add_argument(
        "--system_type", type=str, default="linear", choices=["linear", "nonlinear"]
    )
    parser.add_argument("--n_state", type=int, default=3)
    parser.add_argument("--n_obs", type=int, default=2)
    parser.add_argument("--n_ctrl", type=int, default=0)
    parser.add_argument("--horizon", type=int, default=64)
    parser.add_argument("--num_distinct_systems_train", type=int, default=50)
    parser.add_argument("--num_traj_per_system", type=int, default=10)
    parser.add_argument("--process_noise_std", type=float, default=0.1)
    parser.add_argument("--meas_noise_std", type=float, default=0.1)
    parser.add_argument("--control_std", type=float, default=1.0)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--learning_rate", type=float, default=2e-3)
    parser.add_argument("--patience", type=int, default=5)
    parser.add_argument("--models", type=str, nargs="+", default=None)
    parser.add_argument("--exclude_models", type=str, nargs="+", default=None)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--device", type=str, default="auto", choices=["auto", "cpu", "cuda"]
    )

    args = parser.parse_args()

    # Create config object
    class Config:
        pass

    cfg = Config()
    for key, value in vars(args).items():
        setattr(cfg, key, value)

    # Set seed
    set_seed(cfg.seed)

    # Get device
    if cfg.device == "auto":
        device = get_device()
    else:
        device = torch.device(cfg.device)

    print("=" * 80)
    print("TRANSFORMER KALMAN FILTER - COMPARATIVE TRAINING")
    print("=" * 80)

    # Discover models
    print("\nDiscovering neural network models...")
    all_neural_models = discover_neural_models()
    print(
        f"Found {len(all_neural_models)} neural network models: {list(all_neural_models.keys())}"
    )

    # Filter models
    if cfg.models:
        neural_models = {
            name: all_neural_models[name]
            for name in cfg.models
            if name in all_neural_models
        }
        print(f"Training selected models: {list(neural_models.keys())}")
    else:
        neural_models = all_neural_models.copy()

    if cfg.exclude_models:
        neural_models = {
            name: cls
            for name, cls in neural_models.items()
            if name not in cfg.exclude_models
        }
        print(f"After exclusions: {list(neural_models.keys())}")

    # Create datasets
    print(f"\nCreating {cfg.system_type} datasets...")
    dataset_class = (
        SimpleLinearSystemDataset
        if cfg.system_type == "linear"
        else SimpleNonlinearSystemDataset
    )
    dataset_kwargs = dict(
        n_state=cfg.n_state,
        n_obs=cfg.n_obs,
        n_ctrl=cfg.n_ctrl,
        num_traj_per_system=cfg.num_traj_per_system,
        horizon=cfg.horizon,
        process_noise_std=cfg.process_noise_std,
        meas_noise_std=cfg.meas_noise_std,
        random_controls=cfg.n_ctrl > 0,
        control_std=cfg.control_std,
        noisy=True,
        device=device,
    )

    # Create separate datasets with 5:2:2 ratio (train:val:test)
    # Auto-calculate val and test sizes based on train size
    cfg.num_distinct_systems_val = int((cfg.num_distinct_systems_train * 2) / 5)
    cfg.num_distinct_systems_test = int((cfg.num_distinct_systems_train * 2) / 5)

    print(f"Creating datasets with 5:2:2 ratio (train:val:test)")
    print(
        f"  Train: {cfg.num_distinct_systems_train} systems with {cfg.num_traj_per_system} trajectories each"
    )
    print(
        f"  Val: {cfg.num_distinct_systems_val} systems with {cfg.num_traj_per_system} trajectories each"
    )
    print(
        f"  Test: {cfg.num_distinct_systems_test} systems with {cfg.num_traj_per_system} trajectories each"
    )

    train_ds = dataset_class(
        num_distinct_systems=cfg.num_distinct_systems_train,
        seed=cfg.seed,
        **dataset_kwargs,
    )
    val_ds = dataset_class(
        num_distinct_systems=cfg.num_distinct_systems_val,
        seed=cfg.seed + 1000,
        **dataset_kwargs,
    )
    test_ds = dataset_class(
        num_distinct_systems=cfg.num_distinct_systems_test,
        seed=cfg.seed + 2000,
        **dataset_kwargs,
    )

    loaders = [
        DataLoader(
            ds,
            batch_size=cfg.batch_size,
            shuffle=(i == 0),
            pin_memory=(device.type == "cuda"),
        )
        for i, ds in enumerate([train_ds, val_ds, test_ds])
    ]
    train_loader, val_loader, test_loader = loaders

    # Instantiate models first for metadata printing
    print(f"\nInstantiating models...")
    instantiated_models = {}
    for name, model_class in neural_models.items():
        model = create_model_instance(model_class, cfg, device)
        if model is None:
            print(f"  Warning: Could not instantiate {name}, will skip")
            continue
        instantiated_models[name] = model

    # Print metadata
    print_metadata(train_ds, val_ds, instantiated_models, device, cfg)

    # Train models
    trained_models = {}
    training_histories = {}

    for name, model in instantiated_models.items():
        print(f"\n{'='*80}")
        print(f"Training {name}")
        print("=" * 80)

        history, best_val = train_single_model(
            model, train_loader, val_loader, device, cfg
        )
        trained_models[name] = model
        training_histories[name] = {"best_val": best_val, "history": history}

        print(f"  {name} training complete! Best validation MSE: {best_val:.6f}")

    # Evaluate all models on test set
    print(f"\n{'='*80}")
    print("TEST EVALUATION")
    print("=" * 80)
    test_results = evaluate_all_models(
        trained_models, test_loader, test_ds, cfg.system_type, device, cfg
    )

    # Print results
    print_results(training_histories, test_results)

    # Visualize results
    visualize_results(test_results, training_histories, cfg)

    # For 1D systems, also create state estimation over time visualization
    if cfg.n_state == 1:
        visualize_1d_states(trained_models, test_loader, test_ds, device, cfg)

    print("\nExperiment completed successfully!")
    print(f"\nVisualization saved to: results_{cfg.system_type}_{cfg.n_state}d.png")
    if cfg.n_state == 1:
        print(
            f"1D state estimation saved to: state_estimation_1d_{cfg.system_type}.png"
        )


if __name__ == "__main__":
    main()
