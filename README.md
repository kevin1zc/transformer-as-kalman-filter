# Transformer Kalman Filter

This project implements and verifies the concept from the paper "Can a Transformer Represent a Kalman Filter?" (arXiv:2312.06937), which demonstrates that a single-layer transformer can approximate a Kalman filter to arbitrarily small error.

## Project Structure

```
├── config/                      # Configuration management
│   ├── __init__.py              # Config package initialization
│   └── config.py                # Configuration classes
├── data/                        # Data generation and dataset classes
│   ├── __init__.py              # Data package initialization
│   └── data_generator.py        # LDS data generation utilities
├── training/                    # Training utilities and visualization
│   ├── __init__.py              # Training package initialization
│   └── training.py              # Training and evaluation utilities
├── models/                      # Model architectures
│   ├── __init__.py              # Model package initialization
│   ├── one_layer_transformer.py # Single-layer transformer
│   ├── gru_model.py             # GRU-based state estimator
│   └── kalman_filter.py         # Kalman filter implementation
├── main.py                      # Main script with command-line interface
├── example_1d.py                # 1D example with model selection
├── requirements.txt             # Python dependencies
└── README.md                    # This file
```

## Quick Start

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

### 2. Run 1D Example (with visualization)

```bash
# Transformer model (default)
python example_1d.py

# GRU model
python example_1d.py --model_type gru

# Custom parameters
python example_1d.py --model_type gru --max_epochs 20 --noise_std 0.05
```

This will:
- Train the selected model on a 1D linear dynamical system
- Generate a visualization comparing model vs Kalman filter
- Save the model and visualization plots

### 3. Run Custom Experiment

```bash
# Transformer model
python main.py --n_state 4 --n_obs 3 --max_epochs 30 --enable_viz --model_type transformer

# GRU model
python main.py --n_state 4 --n_obs 3 --max_epochs 30 --enable_viz --model_type gru
```

## Command Line Options

- `--n_state`: State dimension (default: 1)
- `--n_ctrl`: Control dimension (default: 0) 
- `--n_obs`: Observation dimension (default: 2)
- `--horizon`: Sequence length (default: 64)
- `--max_epochs`: Maximum training epochs (default: 20)
- `--patience`: Early stopping patience (default: 5)
- `--learning_rate`: Learning rate (default: 2e-3)
- `--num_train_traj`: Number of training trajectories (default: 5000)
- `--num_val_traj`: Number of validation trajectories (default: 1000)
- `--batch_size`: Batch size (default: 128)
- `--device`: Device to use (auto/cpu/cuda, default: auto)
- `--seed`: Random seed (default: 42)
- `--load_model`: Path to load pre-trained model
- `--save_model`: Save the trained model
- `--enable_viz`: Enable visualization (only for 1D state)
- `--noise_std`: Noise standard deviation (default: 0.05)
- `--model_type`: Model architecture to use (transformer/gru, default: transformer)
- `--use_pos_encoding`: Enable positional encoding for transformer (default: disabled)

## Examples

### Train a 4D system with transformer
```bash
python main.py --n_state 4 --n_obs 3 --max_epochs 30 --save_model --model_type transformer
```

### Train a 4D system with GRU
```bash
python main.py --n_state 4 --n_obs 3 --max_epochs 30 --save_model --model_type gru
```

### Load and evaluate a model
```bash
python main.py --load_model best_lqs_transformer.pt --n_state 4 --n_obs 3
```

### Quick 1D test with visualization
```bash
python example_1d.py  # Transformer (default)
python example_1d.py --model_type gru  # GRU
```

### Transformer with positional encoding
```bash
# 1D example with positional encoding
python example_1d.py --model_type transformer --use_pos_encoding

# 4D system with positional encoding
python main.py --n_state 4 --n_obs 3 --model_type transformer --use_pos_encoding --save_model
```

## Positional Encoding

The transformer model supports optional positional encoding, which can be enabled using the `--use_pos_encoding` flag. By default, positional encoding is disabled as recent research suggests that transformers can learn positional information implicitly through their attention mechanisms.

**When to use positional encoding:**
- **Disabled (default)**: For most cases, especially when following the paper's approach
- **Enabled**: When you want to experiment with explicit positional information or when working with very long sequences

**Note**: Positional encoding only applies to transformer models and is ignored for GRU models.

## Visualization

For 1D state spaces, the system automatically generates visualization plots showing:
- Ground truth state (black line)
- Transformer prediction (blue dashed line) 
- Kalman filter prediction (red dotted line)
- Observations (bottom plot)

The visualization helps verify that the transformer successfully approximates the Kalman filter behavior.

