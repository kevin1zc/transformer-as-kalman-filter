"""
Main script for Transformer Kalman Filter experiment.

This script demonstrates how a single-layer transformer can approximate a Kalman filter
for linear dynamical systems, as described in the paper:
"Can a Transformer Represent a Kalman Filter?" (arXiv:2312.06937)
"""

import argparse
import torch

from config import ExperimentConfig
from data import get_device
from training import train_model, load_trained_model, visualize_1d_results
from models import OneLayerTransformer, GRUStateEstimator


def main():
    """Main function to run the experiment."""
    parser = argparse.ArgumentParser(description="Transformer Kalman Filter Experiment")
    parser.add_argument(
        "--n_state", type=int, default=1, help="State dimension (default: 1)"
    )
    parser.add_argument(
        "--n_ctrl", type=int, default=0, help="Control dimension (default: 0)"
    )
    parser.add_argument(
        "--n_obs", type=int, default=2, help="Observation dimension (default: 2)"
    )
    parser.add_argument(
        "--horizon", type=int, default=64, help="Sequence length (default: 64)"
    )
    parser.add_argument(
        "--max_epochs",
        type=int,
        default=20,
        help="Maximum training epochs (default: 20)",
    )
    parser.add_argument(
        "--patience", type=int, default=5, help="Early stopping patience (default: 5)"
    )
    parser.add_argument(
        "--learning_rate",
        type=float,
        default=2e-3,
        help="Learning rate (default: 2e-3)",
    )
    parser.add_argument(
        "--num_train_traj",
        type=int,
        default=5000,
        help="Number of training trajectories (default: 5000)",
    )
    parser.add_argument(
        "--num_val_traj",
        type=int,
        default=1000,
        help="Number of validation trajectories (default: 1000)",
    )
    parser.add_argument(
        "--batch_size", type=int, default=128, help="Batch size (default: 128)"
    )
    parser.add_argument(
        "--device",
        type=str,
        default="auto",
        choices=["auto", "cpu", "cuda"],
        help="Device to use (default: auto)",
    )
    parser.add_argument(
        "--seed", type=int, default=42, help="Random seed (default: 42)"
    )
    parser.add_argument(
        "--load_model", type=str, default=None, help="Path to load pre-trained model"
    )
    parser.add_argument(
        "--save_model", action="store_true", help="Save the trained model"
    )
    parser.add_argument(
        "--enable_viz",
        action="store_true",
        help="Enable visualization (only for 1D state)",
    )
    parser.add_argument(
        "--noise_std",
        type=float,
        default=0.05,
        help="Noise standard deviation (default: 0.05)",
    )
    parser.add_argument(
        "--model_type",
        type=str,
        default="transformer",
        choices=["transformer", "gru"],
        help="Model architecture to use (default: transformer)",
    )
    parser.add_argument(
        "--use_pos_encoding",
        action="store_true",
        help="Enable positional encoding for transformer (default: disabled)",
    )

    args = parser.parse_args()

    # Determine device
    if args.device == "auto":
        device = get_device()
    else:
        device = torch.device(args.device)

    print(f"Using device: {device}")
    print(f"Experiment configuration:")
    print(f"  State dimension: {args.n_state}")
    print(f"  Control dimension: {args.n_ctrl}")
    print(f"  Observation dimension: {args.n_obs}")
    print(f"  Sequence length: {args.horizon}")
    print(f"  Training trajectories: {args.num_train_traj}")
    print(f"  Validation trajectories: {args.num_val_traj}")
    print(f"  Max epochs: {args.max_epochs}")
    print(f"  Learning rate: {args.learning_rate}")
    print(f"  Random seed: {args.seed}")
    print(f"  Model type: {args.model_type}")
    print()

    # Create configuration
    cfg = ExperimentConfig.default(
        n_state=args.n_state,
        n_ctrl=args.n_ctrl,
        n_obs=args.n_obs,
        horizon=args.horizon,
        device=device,
    )

    # Update training configuration
    cfg.training.max_epochs = args.max_epochs
    cfg.training.patience = args.patience
    cfg.training.learning_rate = args.learning_rate
    cfg.training.num_train_traj = args.num_train_traj
    cfg.training.num_val_traj = args.num_val_traj
    cfg.training.seed = args.seed
    cfg.training.save_model = args.save_model
    cfg.training.enable_visualization = args.enable_viz and args.n_state == 1

    # Update model configuration for positional encoding
    if args.model_type == "transformer":
        cfg.model.use_positional_encoding = args.use_pos_encoding

    # Update LDS configuration
    cfg.lds.batch_size = args.batch_size
    cfg.lds.process_noise_std = args.noise_std
    cfg.lds.meas_noise_std = args.noise_std

    # Select model class
    model_class_map = {"transformer": OneLayerTransformer, "gru": GRUStateEstimator}
    model_class = model_class_map[args.model_type]

    if args.load_model:
        print(f"Loading pre-trained model from {args.load_model}")
        model, lds, training_history = load_trained_model(args.load_model, device)
        print(f"Model loaded successfully!")
        print(f"Training history: {training_history}")

        # Show final performance
        from training import eval_with_kf
        from data import LDSSequenceDataset
        from torch.utils.data import DataLoader

        val_ds = LDSSequenceDataset(
            lds,
            num_traj=args.num_val_traj,
            horizon=args.horizon,
            random_controls=True,
            control_std=cfg.lds.control_std,
            noisy=True,
        )
        val_loader = DataLoader(
            val_ds,
            batch_size=args.batch_size,
            shuffle=False,
            pin_memory=(device.type == "cuda"),
        )

        tr_mse, kf_mse = eval_with_kf(model, val_loader, lds, device=device)
        print(
            f"Val-set MSE vs truth | {args.model_type.title()}: {tr_mse:.6f} | Kalman (oracle): {kf_mse:.6f}"
        )

        # Visualization for 1D case
        if args.enable_viz and args.n_state == 1:
            visualize_1d_results(
                lds,
                model,
                device,
                save_path="loaded_model_visualization.png",
                model_type=args.model_type,
            )
    else:
        print(f"Training new {args.model_type} model...")
        model, lds, training_history = train_model(cfg, model_class=model_class)
        print("Training completed!")

    print("\nExperiment completed successfully!")


if __name__ == "__main__":
    main()
