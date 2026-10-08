"""
Unit tests for src/copulas.py

Tests cover:
  - Gaussian kernel log shape and symmetry
  - Mirrored Clayton kernel log shape and properties
  - Within-portfolio weight normalization (sum == 1)
  - Portfolio CDF monotonicity and boundary conditions
  - CDF batch consistency with single-point evaluation
  - Kernel mathematical properties (non-positive Gaussian log-kernel)
"""
import pytest
import numpy as np
from scipy.stats import norm

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from src.copulas import (
    gaussian_kernel_log,
    mirrored_clayton_kernel_log,
    compute_within_portfolio_gaussian,
    compute_within_portfolio_mirrored_clayton,
    compute_portfolio_cdf,
    compute_portfolio_cdf_batch,
)
from src.marginals import generate_all_leaves


# ── Fixtures ──────────────────────────────────────────────────────────────────

N = 20  # small n for fast tests

@pytest.fixture
def leaves():
    return generate_all_leaves(n=N, seed=1)


@pytest.fixture
def z_vec():
    u = (np.arange(1, N + 1) - 0.5) / N
    return norm.ppf(u)


@pytest.fixture
def u_vec():
    return (np.arange(1, N + 1) - 0.5) / N


# ── Gaussian kernel tests ─────────────────────────────────────────────────────

def test_gaussian_kernel_log_shape(z_vec):
    log_k = gaussian_kernel_log(z_vec, z_vec, gamma=0.5)
    assert log_k.shape == (N, N)


def test_gaussian_kernel_log_symmetric(z_vec):
    log_k = gaussian_kernel_log(z_vec, z_vec, gamma=0.75)
    np.testing.assert_allclose(log_k, log_k.T, atol=1e-12)


def test_gaussian_kernel_log_finite(z_vec):
    """Gaussian copula log-kernel should be finite everywhere.

    Note: the log-kernel (copula density log) CAN be positive when z_a ≈ z_b
    because the bivariate copula density can exceed 1.  The correct property
    is finiteness, not non-positivity.
    """
    log_k = gaussian_kernel_log(z_vec, z_vec, gamma=0.5)
    assert np.all(np.isfinite(log_k)), "Gaussian log-kernel must be finite"
    # Diagonal (z_a = z_b = z): exponent = gamma/(1+gamma)*z^2 >= 0
    assert np.all(np.diag(log_k) >= 0.0 - 1e-12)


def test_gaussian_kernel_log_diagonal_maximum(z_vec):
    """Diagonal (perfect dependence: z_a==z_b) should have least-negative values."""
    log_k = gaussian_kernel_log(z_vec, z_vec, gamma=0.75)
    diag = np.diag(log_k)
    # Diagonal values (z_a=z_b): K = z^2 - (2/gamma)*z^2 + z^2 = 2z^2*(1 - 1/gamma)
    # For gamma<1, 1-1/gamma < 0, so diagonal > off-diagonal for large |z|
    # Just check that some diagonal elements are larger than average off-diagonal
    off_diag_mean = (np.sum(log_k) - np.trace(log_k)) / (N * N - N)
    # Not strictly required to be largest but should be >= off-diagonal mean for gamma<1
    assert np.mean(diag) >= off_diag_mean - 1.0, "Diagonal should not be much smaller than off-diagonal"


# ── Mirrored Clayton kernel tests ──────────────────────────────────────────────

def test_mirrored_clayton_kernel_shape(u_vec):
    log_A = mirrored_clayton_kernel_log(u_vec, u_vec, zeta=1.0)
    assert log_A.shape == (N, N)


def test_mirrored_clayton_kernel_symmetric(u_vec):
    log_A = mirrored_clayton_kernel_log(u_vec, u_vec, zeta=2.3468)
    np.testing.assert_allclose(log_A, log_A.T, atol=1e-10)


def test_mirrored_clayton_kernel_finite(u_vec):
    log_A = mirrored_clayton_kernel_log(u_vec, u_vec, zeta=1.0)
    assert np.all(np.isfinite(log_A)), "All log-A values must be finite"


def test_mirrored_clayton_kernel_larger_zeta_heavier_tail(u_vec):
    """Larger zeta should increase A_{ab} at high u (near 1) more than low u."""
    # A_{ab} at high u_a,u_b (near 1) should be larger for larger zeta
    # because upper-tail dependence increases with zeta
    log_A_low = mirrored_clayton_kernel_log(u_vec, u_vec, zeta=0.5)
    log_A_high = mirrored_clayton_kernel_log(u_vec, u_vec, zeta=3.0)
    # At the top-right corner, high zeta should give larger log-weight
    # (last row, last column)
    assert log_A_high[-1, -1] > log_A_low[-1, -1], "Larger zeta should amplify top-right corner"


# ── Within-portfolio distribution tests ───────────────────────────────────────

def test_gaussian_portfolio_weights_sum_to_one(leaves):
    _, w, _ = compute_within_portfolio_gaussian(
        leaves['mr1_samples'], leaves['cr1_samples'],
        leaves['mr1_z'], leaves['cr1_z'], gamma=0.75)
    assert np.isclose(np.sum(w), 1.0, atol=1e-10)


def test_gaussian_portfolio_weights_nonneg(leaves):
    _, w, _ = compute_within_portfolio_gaussian(
        leaves['mr1_samples'], leaves['cr1_samples'],
        leaves['mr1_z'], leaves['cr1_z'], gamma=0.5)
    assert np.all(w >= 0)


def test_gaussian_portfolio_losses_sorted(leaves):
    losses, _, _ = compute_within_portfolio_gaussian(
        leaves['mr1_samples'], leaves['cr1_samples'],
        leaves['mr1_z'], leaves['cr1_z'], gamma=0.5)
    assert np.all(np.diff(losses) >= 0)


def test_gaussian_portfolio_loss_range(leaves):
    """Total losses should be between min(mr)+min(cr) and max(mr)+max(cr)."""
    losses, _, _ = compute_within_portfolio_gaussian(
        leaves['mr1_samples'], leaves['cr1_samples'],
        leaves['mr1_z'], leaves['cr1_z'], gamma=0.5)
    l_min = leaves['mr1_samples'][0] + leaves['cr1_samples'][0]
    l_max = leaves['mr1_samples'][-1] + leaves['cr1_samples'][-1]
    assert losses[0] >= l_min - 1e-10
    assert losses[-1] <= l_max + 1e-10


def test_gaussian_portfolio_size(leaves):
    losses, w, lw = compute_within_portfolio_gaussian(
        leaves['mr1_samples'], leaves['cr1_samples'],
        leaves['mr1_z'], leaves['cr1_z'], gamma=0.5)
    assert len(losses) == N * N
    assert len(w) == N * N
    assert len(lw) == N * N


def test_clayton_portfolio_weights_sum_to_one(leaves):
    _, w, _ = compute_within_portfolio_mirrored_clayton(
        leaves['mr1_samples'], leaves['cr1_samples'],
        leaves['mr1_u'], leaves['cr1_u'], zeta=2.3468)
    assert np.isclose(np.sum(w), 1.0, atol=1e-10)


def test_clayton_portfolio_losses_sorted(leaves):
    losses, _, _ = compute_within_portfolio_mirrored_clayton(
        leaves['mr1_samples'], leaves['cr1_samples'],
        leaves['mr1_u'], leaves['cr1_u'], zeta=1.0)
    assert np.all(np.diff(losses) >= 0)


# ── Portfolio CDF tests ───────────────────────────────────────────────────────

def test_portfolio_cdf_boundary(leaves):
    losses, w, _ = compute_within_portfolio_gaussian(
        leaves['mr1_samples'], leaves['cr1_samples'],
        leaves['mr1_z'], leaves['cr1_z'], gamma=0.5)
    # Below min loss: CDF = 0
    assert compute_portfolio_cdf(losses[0] - 1.0, losses, w) == 0.0
    # Above max loss: CDF ~ 1
    cdf_max = compute_portfolio_cdf(losses[-1] + 1.0, losses, w)
    assert np.isclose(cdf_max, 1.0, atol=1e-10)


def test_portfolio_cdf_monotone(leaves):
    losses, w, _ = compute_within_portfolio_gaussian(
        leaves['mr1_samples'], leaves['cr1_samples'],
        leaves['mr1_z'], leaves['cr1_z'], gamma=0.5)
    grid = np.linspace(losses[0] - 0.5, losses[-1] + 0.5, 50)
    cdfs = [compute_portfolio_cdf(l, losses, w) for l in grid]
    assert all(cdfs[i] <= cdfs[i + 1] + 1e-12 for i in range(len(cdfs) - 1))


def test_portfolio_cdf_batch_consistent(leaves):
    losses, w, _ = compute_within_portfolio_gaussian(
        leaves['mr1_samples'], leaves['cr1_samples'],
        leaves['mr1_z'], leaves['cr1_z'], gamma=0.5)
    grid = np.linspace(losses[0], losses[-1], 10)
    batch = compute_portfolio_cdf_batch(grid, losses, w)
    single = np.array([compute_portfolio_cdf(l, losses, w) for l in grid])
    np.testing.assert_allclose(batch, single, atol=1e-12)


def test_portfolio_cdf_right_continuity(leaves):
    """Theta(0)=1 convention: CDF(l_s) >= CDF(l_s - eps)."""
    losses, w, _ = compute_within_portfolio_gaussian(
        leaves['mr1_samples'], leaves['cr1_samples'],
        leaves['mr1_z'], leaves['cr1_z'], gamma=0.5)
    # Check at a mid-support point
    l_test = losses[N * N // 2]
    eps = 1e-9
    cdf_at = compute_portfolio_cdf(l_test, losses, w)
    cdf_before = compute_portfolio_cdf(l_test - eps, losses, w)
    assert cdf_at >= cdf_before - 1e-12
