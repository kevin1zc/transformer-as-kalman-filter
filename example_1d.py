"""
Example script for 1D state space with visualization.

This demonstrates different models' ability to approximate a Kalman filter
for a simple 1D linear dynamical system.
"""

import argparse

from config import ExperimentConfig
from data import get_device
from training import train_model
from models import OneLayerTransformer, GRUStateEstimator


def analyze_state_measurement_relationship(lds):
    """Analyze the relationship between state and measurements."""
    import numpy as np

    print("Generating sample data for analysis...")

    # Generate a single trajectory
    X_true, Y, U = lds.generate(horizon=20, batch_size=1, noisy=True)
    X_true = X_true.squeeze(0).cpu()  # (T, 1)
    Y = Y.squeeze(0).cpu()  # (T, 2)

    # Get the observation matrix H
    H = lds.H.cpu().numpy()  # (2, 1)

    print(f"\nObservation matrix H (2x1):")
    print(f"  H[0,0] = {H[0,0]:.4f}")
    print(f"  H[1,0] = {H[1,0]:.4f}")

    print(f"\nNoise parameters:")
    print(f"  Process noise std: {lds.q:.4f}")
    print(f"  Measurement noise std: {lds.r:.4f}")

    # Calculate expected measurements (without noise)
    Y_expected = X_true.numpy() @ H.T  # (T, 2)

    # Convert to numpy for analysis
    X_np = X_true.numpy().flatten()
    Y_np = Y.numpy()

    print(f"\nSample data (first 10 time steps):")
    print(
        "Time | True State | Meas 1 (noisy) | Meas 1 (expected) | Meas 2 (noisy) | Meas 2 (expected)"
    )
    print("-" * 85)
    for t in range(min(10, len(X_np))):
        print(
            f"{t:4d} | {X_np[t]:10.4f} | {Y_np[t,0]:13.4f} | {Y_expected[t,0]:16.4f} | {Y_np[t,1]:13.4f} | {Y_expected[t,1]:16.4f}"
        )

    # Calculate correlation between state and measurements
    corr_1 = np.corrcoef(X_np, Y_np[:, 0])[0, 1]
    corr_2 = np.corrcoef(X_np, Y_np[:, 1])[0, 1]

    print(f"\nCorrelation analysis:")
    print(f"  State vs Measurement 1: {corr_1:.4f}")
    print(f"  State vs Measurement 2: {corr_2:.4f}")

    # Calculate measurement noise
    noise_1 = Y_np[:, 0] - Y_expected[:, 0]
    noise_2 = Y_np[:, 1] - Y_expected[:, 1]

    print(f"\nNoise analysis:")
    print(f"  Measurement 1 noise std: {np.std(noise_1):.4f} (expected: {lds.r:.4f})")
    print(f"  Measurement 2 noise std: {np.std(noise_2):.4f} (expected: {lds.r:.4f})")

    # Calculate signal-to-noise ratio
    signal_power_1 = np.var(Y_expected[:, 0])
    noise_power_1 = np.var(noise_1)
    snr_1 = (
        10 * np.log10(signal_power_1 / noise_power_1)
        if noise_power_1 > 0
        else float("inf")
    )

    signal_power_2 = np.var(Y_expected[:, 1])
    noise_power_2 = np.var(noise_2)
    snr_2 = (
        10 * np.log10(signal_power_2 / noise_power_2)
        if noise_power_2 > 0
        else float("inf")
    )

    print(f"\nSignal-to-Noise Ratio (SNR):")
    print(f"  Measurement 1 SNR: {snr_1:.2f} dB")
    print(f"  Measurement 2 SNR: {snr_2:.2f} dB")

    print(f"\nAnalysis complete!")


def main():
    """Run 1D example with visualization."""
    parser = argparse.ArgumentParser(description="1D State Estimation Example")
    parser.add_argument(
        "--model_type",
        type=str,
        default="transformer",
        choices=["transformer", "gru"],
        help="Model architecture to use (default: transformer)",
    )
    parser.add_argument(
        "--max_epochs",
        type=int,
        default=15,
        help="Maximum training epochs (default: 15)",
    )
    parser.add_argument(
        "--noise_std",
        type=float,
        default=0.1,
        help="Noise standard deviation (default: 0.1)",
    )
    parser.add_argument(
        "--use_pos_encoding",
        action="store_true",
        help="Enable positional encoding for transformer (default: disabled)",
    )

    args = parser.parse_args()

    # Model class mapping
    model_class_map = {"transformer": OneLayerTransformer, "gru": GRUStateEstimator}
    model_class = model_class_map[args.model_type]

    print(f"=== Transformer Kalman Filter - 1D {args.model_type.title()} Example ===")
    print(f"This example demonstrates the {args.model_type}'s ability to approximate")
    print("a Kalman filter for a 1D linear dynamical system.\n")

    # Get device
    device = get_device()
    print(f"Using device: {device}\n")

    # Create configuration for 1D case
    cfg = ExperimentConfig.default(
        n_state=1,  # 1D state space
        n_ctrl=0,  # No control
        n_obs=2,  # 2D observations
        horizon=50,  # Shorter sequences for visualization
        device=device,
        model_type=args.model_type,
    )

    # Update training configuration for faster training
    cfg.training.num_train_traj = 2000
    cfg.training.num_val_traj = 500
    cfg.training.max_epochs = args.max_epochs
    cfg.training.patience = 3
    cfg.training.enable_visualization = True
    cfg.training.save_model = True

    # Update model configuration for positional encoding
    if args.model_type == "transformer":
        cfg.model.use_positional_encoding = args.use_pos_encoding

    # Update LDS configuration
    cfg.lds.batch_size = 64
    cfg.lds.process_noise_std = args.noise_std
    cfg.lds.meas_noise_std = args.noise_std

    print("Configuration:")
    print(f"  Model type: {args.model_type}")
    print(f"  State dimension: {cfg.lds.n_state}")
    print(f"  Observation dimension: {cfg.lds.n_obs}")
    print(f"  Sequence length: {cfg.lds.horizon}")
    print(f"  Training trajectories: {cfg.training.num_train_traj}")
    print(f"  Max epochs: {cfg.training.max_epochs}")
    print(f"  Noise std: {cfg.lds.process_noise_std}")
    print()

    # Train the model
    print(f"Training {args.model_type} model...")
    model, lds, training_history = train_model(cfg, model_class=model_class)

    # Analyze the relationship between state and measurements
    print("\nAnalyzing state-measurement relationship...")
    analyze_state_measurement_relationship(lds)

    print("\nTraining completed!")
    print(f"Best validation loss: {training_history['best_val']:.6f}")
    print(f"Model saved to: {cfg.training.model_save_path}")
    if cfg.training.enable_visualization:
        print(f"Visualization saved to: {cfg.training.model_save_path.replace('model.pt', 'state_estimation_comparison.png')}")

    print("\nThe visualization shows:")
    print("- Ground truth state (black line)")
    print(f"- {args.model_type.title()} prediction (blue dashed line)")
    print("- Kalman filter prediction (red dotted line)")
    print("- Observations (bottom plot)")

    print(f"\nIf the {args.model_type} is working correctly, the blue and red lines")
    print("should be very close to each other and both close to the black line.")


if __name__ == "__main__":
    main()
