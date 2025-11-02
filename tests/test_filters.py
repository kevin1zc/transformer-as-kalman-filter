"""Unit tests for Kalman and Particle filters."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import torch
from models.kalman_filter import kalman_filter_sequence
from models.particle_filter import particle_filter_sequence


# Common systems for testing
def make_1d_kf_sys():
    """Create 1D Kalman filter system."""
    return (
        torch.tensor([[0.9]], dtype=torch.float32),  # A
        torch.tensor([[1.0]], dtype=torch.float32),  # H
        torch.tensor([[0.01]], dtype=torch.float32),  # Q
        torch.tensor([[0.1]], dtype=torch.float32),  # R
    )


def make_2d_kf_sys():
    """Create 2D Kalman filter system."""
    A = torch.tensor([[0.9, 0.1], [-0.1, 0.9]], dtype=torch.float32)
    H = torch.eye(2)
    return A, H, torch.eye(2) * 0.01, torch.eye(2) * 0.1


def make_pf_sys():
    """Create particle filter system functions."""
    return lambda x, u: x * 0.9, lambda x: x * 2.0


def test_kalman_basic():
    """Basic 1D Kalman filter test with known ground truth."""
    A, H, Q, R = make_1d_kf_sys()
    T = 20
    true_x = torch.zeros(1, T, 1)
    true_x[:, 0, 0] = 1.0
    for t in range(1, T):
        true_x[:, t] = 0.9 * true_x[:, t - 1]
    Y = (true_x @ H.T) + torch.randn(1, T, 1) * (R[0, 0] ** 0.5)

    x_hat = kalman_filter_sequence(A, H, Q, R, Y)
    assert x_hat.shape == (1, T, 1)
    assert torch.abs(x_hat - true_x).max() < 1.0


def test_kalman_2d():
    """2D Kalman filter test."""
    A, H, Q, R = make_2d_kf_sys()
    Y = torch.randn(2, 20, 2) * 0.5
    x_hat = kalman_filter_sequence(A, H, Q, R, Y)
    assert x_hat.shape == (2, 20, 2) and torch.isfinite(x_hat).all()


def test_kalman_controls():
    """Kalman filter with control inputs."""
    A, H, Q, R = make_1d_kf_sys()
    B = torch.tensor([[0.5]], dtype=torch.float32)
    U, Y = torch.ones(2, 10, 1) * 0.1, torch.randn(2, 10, 1) * 0.5
    x_hat = kalman_filter_sequence(A, H, Q, R, Y, U=U, Bmat=B)
    assert x_hat.shape == (2, 10, 1) and torch.isfinite(x_hat).all()


def test_kalman_init():
    """Kalman filter with custom initial conditions."""
    A, H, Q, R = make_1d_kf_sys()
    x_hat = kalman_filter_sequence(
        A,
        H,
        Q,
        R,
        torch.randn(1, 10, 1),
        x0=torch.tensor([[2.0]]),
        P0=torch.tensor([[[1.0]]]),
    )
    assert x_hat.shape == (1, 10, 1) and torch.isfinite(x_hat).all()


def test_particle_basic():
    """Basic 1D particle filter test with known ground truth."""
    f, g = make_pf_sys()
    T = 20
    true_x = torch.zeros(1, T, 1)
    true_x[:, 0, 0] = 1.0
    for t in range(1, T):
        true_x[:, t] = 0.9 * true_x[:, t - 1]
    Y = 2.0 * true_x + torch.randn(1, T, 1) * 0.2

    x_hat = particle_filter_sequence(
        f, g, q_std=0.05, r_std=0.2, Y=Y, num_particles=500
    )
    assert x_hat.shape == (1, T, 1) and torch.abs(x_hat - true_x).max() < 1.0


def test_particle_2d():
    """2D particle filter test."""
    f, g = lambda x, u: x * 0.9, lambda x: x
    x_hat = particle_filter_sequence(
        f,
        g,
        q_std=0.1,
        r_std=0.2,
        Y=torch.randn(2, 10, 2) * 0.5,
        num_particles=200,
        x0_mean=torch.zeros(2, 2),
    )
    assert x_hat.shape == (2, 10, 2) and torch.isfinite(x_hat).all()


def test_particle_controls():
    """Particle filter with control inputs."""
    f, g = lambda x, u: x * 0.9 + (u if u is not None else 0), lambda x: x
    x_hat = particle_filter_sequence(
        f,
        g,
        q_std=0.1,
        r_std=0.2,
        Y=torch.randn(1, 10, 1) * 0.5,
        U=torch.ones(1, 10, 1) * 0.1,
        num_particles=200,
    )
    assert x_hat.shape == (1, 10, 1) and torch.isfinite(x_hat).all()


def test_particle_init():
    """Particle filter with custom initial conditions."""
    f, g = make_pf_sys()
    x_hat = particle_filter_sequence(
        f,
        g,
        q_std=0.1,
        r_std=0.2,
        Y=torch.randn(1, 10, 1),
        num_particles=500,
        x0_mean=torch.tensor([[5.0]]),
        x0_std=0.5,
    )
    assert x_hat.shape == (1, 10, 1) and torch.isfinite(x_hat).all()


def test_particle_vs_kalman():
    """Compare particle and Kalman filters on linear system."""
    A, H, Q, R = make_2d_kf_sys()
    Y = torch.randn(1, 20, 2) * 0.5

    x_kf = kalman_filter_sequence(A, H, Q, R, Y)
    f, g = lambda x, u: torch.mm(x, A.T), lambda x: torch.mm(x, H.T)
    x_pf = particle_filter_sequence(
        f,
        g,
        q_std=0.1,
        r_std=0.3162,
        Y=Y,
        num_particles=1000,
        x0_mean=torch.zeros(1, 2),
    )

    assert torch.isfinite(x_kf).all() and torch.isfinite(x_pf).all()
    assert torch.abs(x_kf - x_pf).mean() < 0.5


def test_edge_cases():
    """Edge cases: single timestep, large batches."""
    A, H, Q, R = make_1d_kf_sys()

    # Single timestep
    Y = torch.randn(1, 1, 1)
    assert kalman_filter_sequence(A, H, Q, R, Y).shape == (1, 1, 1)
    f, g = lambda x, u: x * 0.9, lambda x: x
    assert particle_filter_sequence(
        f, g, q_std=0.1, r_std=0.2, Y=Y, num_particles=10
    ).shape == (1, 1, 1)

    # Large batch
    Y = torch.randn(10, 20, 1)
    assert kalman_filter_sequence(A, H, Q, R, Y).shape == (10, 20, 1)
    assert particle_filter_sequence(
        f, g, q_std=0.1, r_std=0.2, Y=Y, num_particles=100
    ).shape == (10, 20, 1)


if __name__ == "__main__":
    tests = [
        test_kalman_basic,
        test_kalman_2d,
        test_kalman_controls,
        test_kalman_init,
        test_particle_basic,
        test_particle_2d,
        test_particle_controls,
        test_particle_init,
        test_particle_vs_kalman,
        test_edge_cases,
    ]

    print("Running filter tests...\n")
    passed = failed = 0
    for test in tests:
        try:
            test()
            print(f"✓ {test.__name__}")
            passed += 1
        except Exception as e:
            print(f"✗ {test.__name__}: {e}")
            failed += 1

    print(f"\n{'='*60}\nResults: {passed} passed, {failed} failed\n{'='*60}")
    sys.exit(1 if failed > 0 else 0)
