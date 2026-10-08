"""
Finite-sum estimation-error diagnostics for ETL and quantile.

Implements paper's standard-deviation approximations (Equations 27-32) for
finite weighted empirical distributions.

# Custom — Context7 found no library equivalent for paper's custom
# finite-sum estimation-error formulas (paper Eq. 27-32)
"""
from __future__ import annotations

import numpy as np
from src.risk_measures import compute_quantile, compute_etl


def compute_upper_tail_moments(
    sorted_losses: np.ndarray,
    weights: np.ndarray,
    alpha: float,
) -> tuple[float, float, float, float]:
    """Compute atom-corrected upper-tail moments U_m(alpha) for m=1,2.

    U_m(alpha) = sum_r p_r * l_r^m * 1{l_r > q_alpha}
                 + eta_alpha * q_alpha^m
    where eta_alpha = (1-alpha) - M_>(q_alpha).

    Parameters
    ----------
    sorted_losses : sorted ascending losses, shape (N,)
    weights : normalized weights, shape (N,)
    alpha : confidence level

    Returns
    -------
    (U1, U2, M_gt, eta)
    """
    cum_weights = np.cumsum(weights)
    etl_info = compute_etl(sorted_losses, weights, cum_weights, alpha)
    q = etl_info["q"]
    M_gt = etl_info["M_gt"]
    eta = etl_info["eta"]

    mask_gt = sorted_losses > q
    U1 = float(np.sum(weights[mask_gt] * sorted_losses[mask_gt])) + eta * q
    U2 = float(np.sum(weights[mask_gt] * sorted_losses[mask_gt] ** 2)) + eta * q ** 2

    return U1, U2, M_gt, eta


def compute_I_moments(
    sorted_losses: np.ndarray,
    weights: np.ndarray,
    alpha: float,
    beta_aux: float,
) -> tuple[float, float]:
    """Compute I_m(alpha, beta_aux) = U_m(alpha) - U_m(beta_aux) for m=1,2.

    Parameters
    ----------
    alpha : primary confidence level (alpha > beta_aux)
    beta_aux : auxiliary confidence level (must satisfy beta_aux < alpha)

    Returns
    -------
    (I1, I2)
    """
    assert beta_aux < alpha, "beta_aux must be strictly less than alpha"
    U1_alpha, U2_alpha, _, _ = compute_upper_tail_moments(sorted_losses, weights, alpha)
    U1_beta, U2_beta, _, _ = compute_upper_tail_moments(sorted_losses, weights, beta_aux)
    return U1_alpha - U1_beta, U2_alpha - U2_beta


def compute_etl_std(
    sorted_losses: np.ndarray,
    weights: np.ndarray,
    n_leaf: int,
    alpha: float,
    beta_aux: float,
) -> float:
    """Estimate the standard deviation of the ETL estimator.

    Paper Eq. (30):
      sigma(ETL_hat_alpha) = (1/sqrt(n)) * sqrt(max(0, V))
    where
      V = [alpha*q_alpha^2 + (1-beta_aux)*q_beta_aux^2] / (alpha-beta_aux)^2
          + I2 / (alpha-beta_aux)^2
          - [(1-beta_aux)*q_beta_aux + alpha*q_alpha + I1]^2 / (alpha-beta_aux)^2

    Note: n_leaf is the leaf sample size (n=200 in the paper).

    Parameters
    ----------
    sorted_losses : sorted losses, shape (N,)
    weights : normalized weights
    n_leaf : leaf sample size
    alpha : primary confidence level
    beta_aux : auxiliary confidence level (< alpha)

    Returns
    -------
    sigma_etl : standard deviation estimate (>= 0)
    """
    cum_weights = np.cumsum(weights)
    q_alpha = compute_quantile(sorted_losses, cum_weights, alpha)
    q_beta = compute_quantile(sorted_losses, cum_weights, beta_aux)
    I1, I2 = compute_I_moments(sorted_losses, weights, alpha, beta_aux)

    d = alpha - beta_aux
    d2 = d ** 2

    # Paper Eq. (30) numerator
    term1 = (alpha * q_alpha ** 2 + (1.0 - beta_aux) * q_beta ** 2) / d2
    term2 = I2 / d2
    term3_num = (1.0 - beta_aux) * q_beta + alpha * q_alpha + I1
    term3 = term3_num ** 2 / d2

    V = term1 + term2 - term3
    # Guard against small negative values due to floating-point error
    V = max(V, 0.0)

    sigma_etl = (1.0 / np.sqrt(n_leaf)) * np.sqrt(V)
    return float(sigma_etl)


def compute_quantile_std(
    sorted_losses: np.ndarray,
    weights: np.ndarray,
    n_leaf: int,
    alpha: float,
    delta_bw: float,
) -> float:
    """Estimate the standard deviation of the quantile estimator.

    Paper Eq. (31)-(32):
      f_Delta(q_alpha) = [F_L(q_alpha + Delta) - F_L(q_alpha - Delta)] / (2*Delta)
      sigma(q_hat_alpha) ~ sqrt(alpha*(1-alpha)/n) / f_Delta(q_alpha)

    Parameters
    ----------
    sorted_losses : sorted losses
    weights : normalized weights
    n_leaf : leaf sample size
    alpha : confidence level
    delta_bw : bandwidth Delta for local density estimation

    Returns
    -------
    sigma_q : quantile standard deviation estimate
    """
    cum_weights = np.cumsum(weights)
    q_alpha = compute_quantile(sorted_losses, cum_weights, alpha)

    # Evaluate CDF at q_alpha ± delta using searchsorted on the weighted CDF
    def cdf_at(l_val: float) -> float:
        idx = np.searchsorted(sorted_losses, l_val, side="right")
        if idx == 0:
            return 0.0
        if idx >= len(cum_weights):
            return float(cum_weights[-1])
        return float(cum_weights[idx - 1])

    f_upper = cdf_at(q_alpha + delta_bw)
    f_lower = cdf_at(q_alpha - delta_bw)
    f_delta = (f_upper - f_lower) / (2.0 * delta_bw)
    f_delta = max(f_delta, 1e-10)  # guard against zero density

    sigma_q = np.sqrt(alpha * (1.0 - alpha) / n_leaf) / f_delta
    return float(sigma_q)


def sensitivity_analysis(
    sorted_losses: np.ndarray,
    weights: np.ndarray,
    n_leaf: int,
    alpha: float,
    beta_aux_list: list,
    delta_list: list,
) -> dict:
    """Run sensitivity analysis of estimation-error formulas.

    Computes ETL standard deviation for each beta_aux and quantile
    standard deviation for each delta bandwidth.

    Parameters
    ----------
    sorted_losses : sorted losses
    weights : normalized weights
    n_leaf : leaf sample size
    alpha : primary confidence level
    beta_aux_list : list of auxiliary confidence levels to sweep
    delta_list : list of bandwidths to sweep for quantile std

    Returns
    -------
    dict with:
      etl_std: list of sigma_etl values (one per beta_aux)
      quantile_std: list of sigma_q values (one per delta)
      beta_aux_list, delta_list: echoed inputs
    """
    etl_stds = []
    for beta_aux in beta_aux_list:
        try:
            s = compute_etl_std(sorted_losses, weights, n_leaf, alpha, beta_aux)
        except Exception:
            s = np.nan
        etl_stds.append(s)

    q_stds = []
    for delta in delta_list:
        try:
            s = compute_quantile_std(sorted_losses, weights, n_leaf, alpha, delta)
        except Exception:
            s = np.nan
        q_stds.append(s)

    return {
        "etl_std": etl_stds,
        "quantile_std": q_stds,
        "beta_aux_list": list(beta_aux_list),
        "delta_list": list(delta_list),
    }
