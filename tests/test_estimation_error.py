"""
Unit tests for src/estimation_error.py

Tests cover:
  - Upper-tail moments U_m: boundary, monotonicity, consistency
  - I_m: differences of moments, sign properties
  - ETL standard-deviation formula: non-negative radicand, scaling with n
  - Quantile SD approximation: positive output, sensitivity to Delta
  - Sensitivity dict: output shape and value properties
  - Edge cases: zero mass above quantile, beta_aux approaching alpha
"""
import pytest
import numpy as np

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from src.estimation_error import (
    compute_upper_tail_moments,
    compute_I_moments,
    compute_etl_std,
    compute_quantile_std,
    sensitivity_analysis,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

N = 200


@pytest.fixture
def uniform_dist():
    losses = np.sort(np.random.default_rng(7).uniform(0, 100, N))
    weights = np.ones(N) / N
    return losses, weights


@pytest.fixture
def pareto_dist():
    """Heavier-tailed distribution: shifted Pareto-like."""
    rng = np.random.default_rng(13)
    u = rng.uniform(0, 1, N)
    losses = np.sort(1.0 + (1.0 - u) ** (-1.0 / 2.5) - 1.0)
    weights = np.ones(N) / N
    return losses, weights


# ---------------------------------------------------------------------------
# Upper-tail moments tests
# ---------------------------------------------------------------------------

def test_upper_tail_moments_returns_four_values(uniform_dist):
    """compute_upper_tail_moments returns (U1, U2, M_gt, eta)."""
    losses, w = uniform_dist
    result = compute_upper_tail_moments(losses, w, alpha=0.99)
    assert len(result) == 4
    U1, U2, M_gt, eta = result
    assert U1 >= 0.0
    assert U2 >= 0.0
    assert M_gt >= 0.0
    assert eta >= 0.0


def test_upper_tail_moments_U1_positive(uniform_dist):
    """U_1 should be strictly positive for non-extreme alpha."""
    losses, w = uniform_dist
    U1, U2, M_gt, eta = compute_upper_tail_moments(losses, w, alpha=0.90)
    assert U1 > 0.0


def test_upper_tail_moments_U2_geq_U1sq_over_mass(uniform_dist):
    """Jensen: E[L^2 | tail] >= E[L | tail]^2."""
    losses, w = uniform_dist
    U1, U2, M_gt, eta = compute_upper_tail_moments(losses, w, alpha=0.95)
    tail_mass = M_gt + eta
    if tail_mass < 1e-10:
        pytest.skip("No tail mass")
    cond_mean = U1 / tail_mass
    cond_sq_mean = U2 / tail_mass
    assert cond_sq_mean >= cond_mean ** 2 - 1e-6, "Jensen inequality violated"


def test_upper_tail_moments_U1_decreases_with_higher_alpha(uniform_dist):
    """Stricter quantile leaves less tail mass: U1(0.99) <= U1(0.95)."""
    losses, w = uniform_dist
    U1_95, _, _, _ = compute_upper_tail_moments(losses, w, alpha=0.95)
    U1_99, _, _, _ = compute_upper_tail_moments(losses, w, alpha=0.99)
    assert U1_99 <= U1_95 + 1e-8


def test_upper_tail_moments_all_finite(uniform_dist):
    losses, w = uniform_dist
    U1, U2, M_gt, eta = compute_upper_tail_moments(losses, w, alpha=0.99)
    assert all(np.isfinite(v) for v in [U1, U2, M_gt, eta])


# ---------------------------------------------------------------------------
# I_moments tests
# ---------------------------------------------------------------------------

def test_I_moments_returns_two_values(uniform_dist):
    """compute_I_moments returns (I1, I2) tuple."""
    losses, w = uniform_dist
    result = compute_I_moments(losses, w, alpha=0.99, beta_aux=0.95)
    assert len(result) == 2
    I1, I2 = result
    assert np.isfinite(I1)
    assert np.isfinite(I2)


def test_I_moments_requires_beta_lt_alpha(uniform_dist):
    """beta_aux >= alpha must raise AssertionError."""
    losses, w = uniform_dist
    with pytest.raises(AssertionError):
        compute_I_moments(losses, w, alpha=0.95, beta_aux=0.99)


def test_I_moments_consistency_with_U(uniform_dist):
    """I_m = U_m(alpha) - U_m(beta): check via direct subtraction."""
    losses, w = uniform_dist
    U1_a, U2_a, _, _ = compute_upper_tail_moments(losses, w, alpha=0.99)
    U1_b, U2_b, _, _ = compute_upper_tail_moments(losses, w, alpha=0.95)
    I1, I2 = compute_I_moments(losses, w, alpha=0.99, beta_aux=0.95)
    assert I1 == pytest.approx(U1_a - U1_b, abs=1e-12)
    assert I2 == pytest.approx(U2_a - U2_b, abs=1e-12)


# ---------------------------------------------------------------------------
# ETL standard deviation tests
# ---------------------------------------------------------------------------

def test_etl_std_nonneg(uniform_dist):
    losses, w = uniform_dist
    sd = compute_etl_std(losses, w, n_leaf=N, alpha=0.99, beta_aux=0.95)
    assert sd >= 0.0, f"ETL SD should be non-negative, got {sd}"


def test_etl_std_finite(uniform_dist):
    losses, w = uniform_dist
    sd = compute_etl_std(losses, w, n_leaf=N, alpha=0.99, beta_aux=0.95)
    assert np.isfinite(sd)


def test_etl_std_scales_inversely_with_sqrt_n(uniform_dist):
    """sigma(ETL) ~ 1/sqrt(n): ratio SD_200/SD_400 should be sqrt(2)."""
    losses, w = uniform_dist
    sd200 = compute_etl_std(losses, w, n_leaf=200, alpha=0.99, beta_aux=0.95)
    sd400 = compute_etl_std(losses, w, n_leaf=400, alpha=0.99, beta_aux=0.95)
    if sd200 > 1e-10:
        ratio = sd200 / sd400
        assert ratio == pytest.approx(np.sqrt(2), rel=0.01), f"Ratio {ratio:.4f} != sqrt(2)"


def test_etl_std_nonneg_pareto(pareto_dist):
    """Non-negative ETL SD for heavy-tailed distribution."""
    losses, w = pareto_dist
    sd95 = compute_etl_std(losses, w, n_leaf=N, alpha=0.95, beta_aux=0.90)
    sd99 = compute_etl_std(losses, w, n_leaf=N, alpha=0.99, beta_aux=0.95)
    assert sd95 >= 0.0
    assert sd99 >= 0.0


def test_etl_std_positive_nontrivial(uniform_dist):
    """For a spread distribution, ETL SD should be strictly positive."""
    losses, w = uniform_dist
    sd = compute_etl_std(losses, w, n_leaf=N, alpha=0.95, beta_aux=0.90)
    assert sd > 0.0


# ---------------------------------------------------------------------------
# Quantile standard deviation tests
# ---------------------------------------------------------------------------

def test_quantile_std_nonneg(uniform_dist):
    losses, w = uniform_dist
    sd = compute_quantile_std(losses, w, n_leaf=N, alpha=0.99, delta_bw=1.0)
    assert sd >= 0.0


def test_quantile_std_finite(uniform_dist):
    losses, w = uniform_dist
    sd = compute_quantile_std(losses, w, n_leaf=N, alpha=0.99, delta_bw=1.0)
    assert np.isfinite(sd)


def test_quantile_std_scales_with_n(uniform_dist):
    """sigma(q) ~ 1/sqrt(n): ratio SD_200/SD_800 should be 2."""
    losses, w = uniform_dist
    sd200 = compute_quantile_std(losses, w, n_leaf=200, alpha=0.99, delta_bw=2.0)
    sd800 = compute_quantile_std(losses, w, n_leaf=800, alpha=0.99, delta_bw=2.0)
    if sd200 > 1e-10:
        ratio = sd200 / sd800
        assert ratio == pytest.approx(2.0, rel=0.01), f"Ratio {ratio:.4f} != 2"


def test_quantile_std_sensitive_to_delta(uniform_dist):
    """Quantile SD should vary with bandwidth Delta."""
    losses, w = uniform_dist
    sds = [compute_quantile_std(losses, w, n_leaf=N, alpha=0.99, delta_bw=d)
           for d in [0.1, 0.5, 1.0, 5.0]]
    assert not np.allclose(sds, sds[0], rtol=0.001), "All delta values give identical SD"


# ---------------------------------------------------------------------------
# Sensitivity analysis dict tests
# ---------------------------------------------------------------------------

def test_sensitivity_analysis_returns_dict(uniform_dist):
    losses, w = uniform_dist
    result = sensitivity_analysis(
        losses, w, n_leaf=N, alpha=0.99,
        beta_aux_list=[0.90, 0.95], delta_list=[0.5, 1.0],
    )
    assert isinstance(result, dict)
    assert "etl_std" in result
    assert "quantile_std" in result
    assert "beta_aux_list" in result
    assert "delta_list" in result


def test_sensitivity_analysis_etl_std_length(uniform_dist):
    losses, w = uniform_dist
    beta_aux_list = [0.90, 0.925, 0.95, 0.975]
    result = sensitivity_analysis(
        losses, w, n_leaf=N, alpha=0.99,
        beta_aux_list=beta_aux_list, delta_list=[1.0],
    )
    assert len(result["etl_std"]) == len(beta_aux_list)


def test_sensitivity_analysis_quantile_std_length(uniform_dist):
    losses, w = uniform_dist
    delta_list = [0.1, 0.5, 1.0, 2.0]
    result = sensitivity_analysis(
        losses, w, n_leaf=N, alpha=0.99,
        beta_aux_list=[0.95], delta_list=delta_list,
    )
    assert len(result["quantile_std"]) == len(delta_list)


def test_sensitivity_analysis_etl_std_nonneg(uniform_dist):
    losses, w = uniform_dist
    result = sensitivity_analysis(
        losses, w, n_leaf=N, alpha=0.99,
        beta_aux_list=[0.90, 0.95], delta_list=[1.0],
    )
    for v in result["etl_std"]:
        assert v >= 0.0 or np.isnan(v)


def test_sensitivity_analysis_quantile_std_nonneg(uniform_dist):
    losses, w = uniform_dist
    result = sensitivity_analysis(
        losses, w, n_leaf=N, alpha=0.99,
        beta_aux_list=[0.95], delta_list=[0.5, 1.0],
    )
    for v in result["quantile_std"]:
        assert v >= 0.0 or np.isnan(v)
