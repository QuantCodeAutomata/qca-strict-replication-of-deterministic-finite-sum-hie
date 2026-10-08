"""
Unit tests for src/marginals.py

Tests cover:
  - Sample generation shapes and sorting
  - Lognormal distribution properties
  - Shifted Pareto parameterization
  - Midpoint plotting positions
  - Gaussian z-score transforms
  - generate_all_leaves interface
  - Edge cases: n=1, extreme parameters
"""
import pytest
import numpy as np
from scipy.stats import norm, kstest, lognorm

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from src.marginals import (
    generate_market_leaf,
    generate_credit_leaf,
    generate_all_leaves,
    MU_M, SIGMA_M, X0_CR, C_CR, BETA_TAIL,
)


# ── Fixtures ──────────────────────────────────────────────────────────────────

@pytest.fixture
def rng():
    return np.random.default_rng(0)


@pytest.fixture
def market_leaf(rng):
    return generate_market_leaf(200, MU_M, SIGMA_M, rng)


@pytest.fixture
def credit_leaf(rng):
    return generate_credit_leaf(200, X0_CR, C_CR, BETA_TAIL, rng)


# ── Market leaf tests ─────────────────────────────────────────────────────────

def test_market_leaf_shape(market_leaf):
    samples, u, z = market_leaf
    assert samples.shape == (200,)
    assert u.shape == (200,)
    assert z.shape == (200,)


def test_market_leaf_sorted(market_leaf):
    samples, _, _ = market_leaf
    assert np.all(np.diff(samples) >= 0), "Samples must be sorted ascending"


def test_market_leaf_positive(market_leaf):
    samples, _, _ = market_leaf
    assert np.all(samples > 0), "Lognormal samples must be positive"


def test_market_leaf_midpoint_positions(market_leaf):
    _, u, _ = market_leaf
    n = len(u)
    expected = (np.arange(1, n + 1) - 0.5) / n
    np.testing.assert_allclose(u, expected, rtol=1e-12)


def test_market_leaf_z_in_range(market_leaf):
    _, _, z = market_leaf
    assert np.all(np.isfinite(z)), "Z scores must be finite"
    # Midpoint positions never hit 0 or 1, so z must be finite
    assert z[0] > -10 and z[-1] < 10


def test_market_leaf_z_consistent_with_u(market_leaf):
    _, u, z = market_leaf
    # z = Phi^{-1}(u), so Phi(z) = u
    np.testing.assert_allclose(norm.cdf(z), u, rtol=1e-10)


def test_market_leaf_lognormal_properties():
    """Large sample should pass KS test for lognormal."""
    rng_large = np.random.default_rng(12345)
    samples, _, _ = generate_market_leaf(5000, MU_M, SIGMA_M, rng_large)
    # lognormal with s=sigma_M, scale=exp(mu_M)
    stat, pval = kstest(samples, 'lognorm', args=(SIGMA_M, 0, np.exp(MU_M)))
    assert pval > 0.01, f"KS test p-value too low: {pval:.4f}"


# ── Credit leaf tests ─────────────────────────────────────────────────────────

def test_credit_leaf_shape(credit_leaf):
    samples, u, z = credit_leaf
    assert samples.shape == (200,)
    assert u.shape == (200,)
    assert z.shape == (200,)


def test_credit_leaf_sorted(credit_leaf):
    samples, _, _ = credit_leaf
    assert np.all(np.diff(samples) >= 0)


def test_credit_leaf_above_x0(credit_leaf):
    samples, _, _ = credit_leaf
    assert np.all(samples >= X0_CR), "Shifted Pareto must be >= x0"


def test_credit_leaf_pareto_cdf():
    """Test that CDF formula F_X(x) = 1 - ((x-x0)/c + 1)^(-beta) holds."""
    rng_test = np.random.default_rng(99)
    samples, u_plot, _ = generate_credit_leaf(2000, X0_CR, C_CR, BETA_TAIL, rng_test)
    # Compute empirical CDF at midpoint positions
    # True CDF at sorted sample values
    true_cdf = 1.0 - ((samples - X0_CR) / C_CR + 1.0) ** (-BETA_TAIL)
    # Empirical CDF at sorted sample is approximately u_plot for large n
    assert np.corrcoef(true_cdf, u_plot)[0, 1] > 0.99, "CDF shape mismatch"


def test_credit_leaf_inverse_cdf():
    """Check that inverse CDF gives back correct quantile values."""
    # For u uniform on (0,1): X = x0 + c*((1-u)^(-1/beta) - 1)
    u_test = np.array([0.1, 0.5, 0.9, 0.99])
    expected = X0_CR + C_CR * ((1.0 - u_test) ** (-1.0 / BETA_TAIL) - 1.0)
    # Verify these satisfy F_X(X) = u
    cdf_vals = 1.0 - ((expected - X0_CR) / C_CR + 1.0) ** (-BETA_TAIL)
    np.testing.assert_allclose(cdf_vals, u_test, rtol=1e-10)


# ── generate_all_leaves tests ─────────────────────────────────────────────────

def test_generate_all_leaves_keys():
    leaves = generate_all_leaves(n=50, seed=7)
    expected_keys = {
        'mr1_samples', 'mr1_u', 'mr1_z',
        'cr1_samples', 'cr1_u', 'cr1_z',
        'mr2_samples', 'mr2_u', 'mr2_z',
        'cr2_samples', 'cr2_u', 'cr2_z',
        'seed', 'n',
    }
    assert expected_keys.issubset(set(leaves.keys()))


def test_generate_all_leaves_reproducible():
    l1 = generate_all_leaves(n=100, seed=42)
    l2 = generate_all_leaves(n=100, seed=42)
    np.testing.assert_array_equal(l1['mr1_samples'], l2['mr1_samples'])
    np.testing.assert_array_equal(l1['cr1_samples'], l2['cr1_samples'])


def test_generate_all_leaves_independent():
    """Four leaves should not be identical to each other."""
    leaves = generate_all_leaves(n=200, seed=0)
    assert not np.allclose(leaves['mr1_samples'], leaves['mr2_samples'])
    assert not np.allclose(leaves['cr1_samples'], leaves['cr2_samples'])


def test_generate_all_leaves_n1():
    """Edge case: n=1 should still work."""
    leaves = generate_all_leaves(n=1, seed=0)
    assert leaves['mr1_samples'].shape == (1,)
    assert leaves['cr1_samples'].shape == (1,)
    assert np.isclose(leaves['mr1_u'][0], 0.5)


# ── Constants ─────────────────────────────────────────────────────────────────

def test_module_constants():
    assert MU_M == 0.0
    assert SIGMA_M == 0.25
    assert X0_CR == 1.0
    assert C_CR == 1.0
    assert BETA_TAIL == 2.5
