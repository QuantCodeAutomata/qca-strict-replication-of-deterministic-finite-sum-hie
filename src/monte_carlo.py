"""
Monte Carlo benchmark for hierarchical loss aggregation.

Implements 10^6 simulation draws for Gaussian and mirrored Clayton
top-level hierarchies. Within-portfolio marginals are mapped through
the deterministic weighted CDFs built from the leaf samples.

# Using scipy.stats.multivariate_normal — Context7 confirmed for Gaussian copula simulation
# Custom — Context7 found no library equivalent for mirrored Clayton conditional inversion
"""
from __future__ import annotations

import numpy as np
from scipy.stats import norm, multivariate_normal

from src.copulas import compute_portfolio_cdf_batch


# ---------------------------------------------------------------------------
# Copula samplers
# ---------------------------------------------------------------------------

def sample_gaussian_copula_2d(
    rho: float,
    n_samples: int,
    rng: np.random.Generator,
) -> tuple[np.ndarray, np.ndarray]:
    """Sample (U1, U2) from a bivariate Gaussian copula with correlation rho.

    Parameters
    ----------
    rho : correlation parameter
    n_samples : number of samples
    rng : numpy random Generator

    Returns
    -------
    u1, u2 : uniform marginals on (0, 1), shape (n_samples,) each
    """
    # Using scipy.stats.multivariate_normal — Context7 confirmed
    cov = np.array([[1.0, rho], [rho, 1.0]])
    z = rng.multivariate_normal(mean=[0.0, 0.0], cov=cov, size=n_samples)
    # Map standard normal to uniform via CDF
    u1 = norm.cdf(z[:, 0])
    u2 = norm.cdf(z[:, 1])
    return u1, u2


def sample_mirrored_clayton_copula_2d(
    theta: float,
    n_samples: int,
    rng: np.random.Generator,
) -> tuple[np.ndarray, np.ndarray]:
    """Sample (U1, U2) from a bivariate mirrored Clayton copula.

    Uses conditional inversion of the Clayton copula then reflects via
    survival transform (u -> 1-u) to obtain the mirrored (upper-tail) version.

    Clayton conditional inverse (Custom — Context7 found no library equivalent):
      Given u1 ~ Uniform(0,1) and v ~ Uniform(0,1):
      u2 = [u1^(-theta) * (v^(-theta/(1+theta)) - 1) + 1]^(-1/theta)
    Then apply mirrored transform: (u1, u2) -> (1-u1, 1-u2).

    Parameters
    ----------
    theta : Clayton parameter (> 0)
    n_samples : number of samples
    rng : numpy random Generator

    Returns
    -------
    u1, u2 : uniform marginals on (0, 1), shape (n_samples,) each
    """
    u1_raw = rng.uniform(0.0, 1.0, n_samples)
    v = rng.uniform(0.0, 1.0, n_samples)

    # Conditional Clayton inverse
    # u2 = [u1^(-theta) * (v^(-theta/(theta+1)) - 1) + 1]^(-1/theta)
    exp_v = v ** (-theta / (theta + 1.0))
    u2_raw = (u1_raw ** (-theta) * (exp_v - 1.0) + 1.0) ** (-1.0 / theta)

    # Clip to valid range
    u1_raw = np.clip(u1_raw, 1e-12, 1.0 - 1e-12)
    u2_raw = np.clip(u2_raw, 1e-12, 1.0 - 1e-12)

    # Mirror: survival transform
    u1 = 1.0 - u1_raw
    u2 = 1.0 - u2_raw

    return u1, u2


# ---------------------------------------------------------------------------
# Inverse CDF for within-portfolio distributions
# ---------------------------------------------------------------------------

def build_inverse_cdf(
    sorted_losses: np.ndarray,
    weights: np.ndarray,
) -> callable:
    """Build an interpolating inverse CDF function from a weighted discrete distribution.

    Parameters
    ----------
    sorted_losses : sorted ascending support points, shape (N,)
    weights : normalized weights, shape (N,)

    Returns
    -------
    inv_cdf : callable(u) -> loss value via linear interpolation
    """
    cum_weights = np.cumsum(weights)

    def inv_cdf(u: np.ndarray) -> np.ndarray:
        """Map uniform quantiles u to loss values via linear interpolation."""
        return np.interp(u, cum_weights, sorted_losses)

    return inv_cdf


# ---------------------------------------------------------------------------
# Monte Carlo benchmark functions
# ---------------------------------------------------------------------------

def _compute_mc_metrics(
    l_total: np.ndarray,
    grid_points: list,
) -> dict:
    """Compute empirical CDF, quantile, and ETL from MC sample."""
    n_sim = len(l_total)
    l_sorted = np.sort(l_total)

    # Empirical CDF function
    def emp_cdf(l_val):
        return float(np.mean(l_total <= l_val))

    # CDF at grid points
    cdf_at_grid = {l: emp_cdf(l) for l in grid_points}

    # q_99
    q99 = float(np.quantile(l_sorted, 0.99, method="inverted_cdf"))

    # ETL_99 (empirical, no atom correction needed for MC)
    tail_mask = l_total > q99
    if np.mean(tail_mask) > 0:
        etl99 = float(np.mean(l_total[tail_mask]))
    else:
        etl99 = q99

    return {
        "cdf_at_grid": cdf_at_grid,
        "q99": q99,
        "etl99": etl99,
        "emp_cdf": emp_cdf,
        "l_sorted": l_sorted,
    }


def run_monte_carlo_gaussian(
    leaves: dict,
    params: dict,
    n_sim: int = 1_000_000,
    seed: int = 42,
    grid_points: list = None,
) -> dict:
    """Monte Carlo benchmark for Gaussian top-level hierarchy.

    Algorithm:
    1. Build deterministic within-portfolio CDFs using Gaussian copula weights.
    2. Sample 10^6 bivariate Gaussian top-level copula draws (rho_top).
    3. Map to portfolio losses via deterministic inverse CDFs.
    4. Compute L = L_1 + L_2 and extract empirical risk measures.

    Parameters
    ----------
    leaves : output of generate_all_leaves()
    params : dict with keys gamma_1, gamma_2, rho_top
    n_sim : number of MC draws
    seed : random seed
    grid_points : loss values at which to evaluate empirical CDF

    Returns
    -------
    dict with q99, etl99, cdf_at_grid, l_sorted, emp_cdf
    """
    from src.copulas import compute_within_portfolio_gaussian

    if grid_points is None:
        grid_points = [25.0, 28.0, 30.0]

    gamma_1 = params["gamma_1"]
    gamma_2 = params["gamma_2"]
    rho_top = params["rho_top"]

    # Build within-portfolio weighted distributions
    losses1, w1, _ = compute_within_portfolio_gaussian(
        leaves["mr1_samples"], leaves["cr1_samples"],
        leaves["mr1_z"], leaves["cr1_z"], gamma=gamma_1,
    )
    losses2, w2, _ = compute_within_portfolio_gaussian(
        leaves["mr2_samples"], leaves["cr2_samples"],
        leaves["mr2_z"], leaves["cr2_z"], gamma=gamma_2,
    )

    inv_cdf1 = build_inverse_cdf(losses1, w1)
    inv_cdf2 = build_inverse_cdf(losses2, w2)

    # Sample top-level Gaussian copula
    rng = np.random.default_rng(seed)
    u1, u2 = sample_gaussian_copula_2d(rho_top, n_sim, rng)

    # Map through inverse CDFs
    L1 = inv_cdf1(u1)
    L2 = inv_cdf2(u2)
    L = L1 + L2

    return _compute_mc_metrics(L, grid_points)


def run_monte_carlo_clayton(
    leaves: dict,
    params: dict,
    n_sim: int = 1_000_000,
    seed: int = 42,
    grid_points: list = None,
) -> dict:
    """Monte Carlo benchmark for mirrored Clayton top-level hierarchy.

    Algorithm:
    1. Build within-portfolio CDFs using mirrored Clayton copula weights.
    2. Sample 10^6 mirrored Clayton top-level copula draws (theta).
    3. Map to portfolio losses via inverse CDFs.
    4. Compute L = L_1 + L_2 and extract empirical risk measures.

    Parameters
    ----------
    leaves : output of generate_all_leaves()
    params : dict with keys zeta_1, zeta_2, theta
    n_sim : number of MC draws
    seed : random seed

    Returns
    -------
    dict with q99, etl99, cdf_at_grid, l_sorted, emp_cdf
    """
    from src.copulas import compute_within_portfolio_mirrored_clayton

    if grid_points is None:
        grid_points = [25.0, 28.0, 30.0]

    zeta_1 = params["zeta_1"]
    zeta_2 = params["zeta_2"]
    theta = params["theta"]

    # Build within-portfolio weighted distributions (mirrored Clayton)
    losses1, w1, _ = compute_within_portfolio_mirrored_clayton(
        leaves["mr1_samples"], leaves["cr1_samples"],
        leaves["mr1_u"], leaves["cr1_u"], zeta=zeta_1,
    )
    losses2, w2, _ = compute_within_portfolio_mirrored_clayton(
        leaves["mr2_samples"], leaves["cr2_samples"],
        leaves["mr2_u"], leaves["cr2_u"], zeta=zeta_2,
    )

    inv_cdf1 = build_inverse_cdf(losses1, w1)
    inv_cdf2 = build_inverse_cdf(losses2, w2)

    # Sample top-level mirrored Clayton copula
    rng = np.random.default_rng(seed)
    u1, u2 = sample_mirrored_clayton_copula_2d(theta, n_sim, rng)

    L1 = inv_cdf1(u1)
    L2 = inv_cdf2(u2)
    L = L1 + L2

    return _compute_mc_metrics(L, grid_points)
