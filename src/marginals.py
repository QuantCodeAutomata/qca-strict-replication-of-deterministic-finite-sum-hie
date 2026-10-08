"""
Marginal distribution generation for four-leaf hierarchical loss model.

Implements:
  - Market-risk leaf: Lognormal(mu_M=0, sigma_M=0.25)
  - Credit-risk leaf: Shifted Pareto with x0=1, c=1, beta_tail=2.5
    Parameterization: X = x0 + Y, F_Y(y) = 1 - (1+y/c)^(-beta_tail) for y>=0
    Inverse CDF: X = x0 + c*((1-u)^(-1/beta_tail) - 1)

Plotting positions: u_m = (m - 0.5) / n  (midpoint rule)
"""
from __future__ import annotations

import numpy as np
from scipy.stats import norm

# Module-level constants for marginal parameters used across modules
MU_M = 0.0        # Lognormal log-mean for market-risk leaf
SIGMA_M = 0.25    # Lognormal log-std for market-risk leaf
X0_CR = 1.0       # Shifted Pareto shift parameter
C_CR = 1.0        # Shifted Pareto scale parameter
BETA_TAIL = 2.5   # Shifted Pareto tail exponent


def generate_market_leaf(
    n: int,
    mu_M: float,
    sigma_M: float,
    rng: np.random.Generator,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Generate, sort, and rank a market-risk leaf sample (lognormal).

    Parameters
    ----------
    n : sample size
    mu_M : log-mean of the lognormal distribution
    sigma_M : log-std of the lognormal distribution
    rng : numpy random Generator for reproducibility

    Returns
    -------
    samples : sorted ascending array of shape (n,)
    u : midpoint plotting positions u_m = (m-0.5)/n
    z : Gaussian quantiles z_m = Phi^{-1}(u_m)
    """
    # Using scipy.stats.norm.rvs — Context7 confirmed
    raw = rng.standard_normal(n)
    samples = np.exp(mu_M + sigma_M * raw)
    samples = np.sort(samples)
    u = (np.arange(1, n + 1) - 0.5) / n
    z = norm.ppf(u)  # Using scipy.stats.norm.ppf — Context7 confirmed
    return samples, u, z


def generate_credit_leaf(
    n: int,
    x0: float,
    c: float,
    beta_tail: float,
    rng: np.random.Generator,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Generate, sort, and rank a credit-risk leaf sample (shifted Pareto).

    Parameterization (paper-fixed):
      F_X(x) = 1 - ((x - x0)/c + 1)^(-beta_tail)  for x >= x0
      Inverse CDF: X = x0 + c * ((1-u)^(-1/beta_tail) - 1)

    Parameters
    ----------
    n : sample size
    x0 : shift parameter (lower bound of support)
    c : scale parameter
    beta_tail : tail exponent (> 0)
    rng : numpy random Generator

    Returns
    -------
    samples, u, z  (same convention as generate_market_leaf)
    """
    # Custom — Context7 found no library equivalent (paper Eq. shifted Pareto)
    u_uniform = rng.uniform(0.0, 1.0, n)
    # Inverse CDF via probability integral transform
    samples = x0 + c * ((1.0 - u_uniform) ** (-1.0 / beta_tail) - 1.0)
    samples = np.sort(samples)
    u = (np.arange(1, n + 1) - 0.5) / n
    z = norm.ppf(u)
    return samples, u, z


def generate_all_leaves(n: int, seed: int = 42) -> dict:
    """Generate all four leaf samples with fixed seed.

    Parameters
    ----------
    n : leaf sample size
    seed : random seed for full reproducibility

    Returns
    -------
    dict with keys:
      mr1_samples, mr1_u, mr1_z  (portfolio-1 market risk)
      mr2_samples, mr2_u, mr2_z  (portfolio-2 market risk)
      cr1_samples, cr1_u, cr1_z  (portfolio-1 credit risk)
      cr2_samples, cr2_u, cr2_z  (portfolio-2 credit risk)
      seed, n
    """
    rng = np.random.default_rng(seed)

    mr1_samples, mr1_u, mr1_z = generate_market_leaf(n, mu_M=0.0, sigma_M=0.25, rng=rng)
    cr1_samples, cr1_u, cr1_z = generate_credit_leaf(n, x0=1.0, c=1.0, beta_tail=2.5, rng=rng)
    mr2_samples, mr2_u, mr2_z = generate_market_leaf(n, mu_M=0.0, sigma_M=0.25, rng=rng)
    cr2_samples, cr2_u, cr2_z = generate_credit_leaf(n, x0=1.0, c=1.0, beta_tail=2.5, rng=rng)

    return {
        "mr1_samples": mr1_samples, "mr1_u": mr1_u, "mr1_z": mr1_z,
        "cr1_samples": cr1_samples, "cr1_u": cr1_u, "cr1_z": cr1_z,
        "mr2_samples": mr2_samples, "mr2_u": mr2_u, "mr2_z": mr2_z,
        "cr2_samples": cr2_samples, "cr2_u": cr2_u, "cr2_z": cr2_z,
        "seed": seed, "n": n,
    }
