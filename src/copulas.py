"""
Within-portfolio copula kernel computations for the hierarchical loss model.

Implements:
  - Gaussian copula density kernel (paper equations 10-11)
  - Mirrored Clayton copula density kernel (paper equations 12-13)
  - Within-portfolio weighted distributions
  - Portfolio CDF evaluation

All formulas are custom implementations.
# Custom — Context7 found no library equivalent for paper's hierarchical finite-sum formulas
"""
from __future__ import annotations

import numpy as np
from scipy.special import logsumexp
from scipy.stats import norm


# ---------------------------------------------------------------------------
# Log-kernel functions
# ---------------------------------------------------------------------------

def gaussian_kernel_log(z_a: np.ndarray, z_b: np.ndarray, gamma: float) -> np.ndarray:
    """Compute log of Gaussian copula density kernel for all (a,b) pairs.

    The Gaussian copula density at (u_a, u_b) with correlation gamma is:
      c(u_a, u_b) = (1/sqrt(1-gamma^2)) * exp[-gamma^2/(2(1-gamma^2)) * K(z_a, z_b)]
    where K(z_a, z_b) = z_a^2 - (2/gamma)*z_a*z_b + z_b^2.

    This function returns the log of the kernel part only (excluding the 1/sqrt(...) constant):
      log_kernel(a, b) = -gamma^2 / (2*(1-gamma^2)) * (z_a^2 - (2/gamma)*z_a*z_b + z_b^2)

    Parameters
    ----------
    z_a : 1D array of length n (Gaussian quantiles for first leaf)
    z_b : 1D array of length n (Gaussian quantiles for second leaf)
    gamma : within-portfolio Gaussian correlation parameter

    Returns
    -------
    log_kernel : n x n matrix where entry (a,b) = log-weight for pair (a,b)
    """
    # Custom — Context7 found no library equivalent (paper Eq. 10-11)
    factor = -gamma ** 2 / (2.0 * (1.0 - gamma ** 2))
    # Kernel: z_a^2 - (2/gamma)*z_a*z_b + z_b^2
    # Vectorized outer computation
    z_a2 = z_a ** 2  # (n,)
    z_b2 = z_b ** 2  # (n,)
    cross = np.outer(z_a, z_b)  # (n, n): z_a[i]*z_b[j]
    kernel = z_a2[:, None] - (2.0 / gamma) * cross + z_b2[None, :]  # (n, n)
    return factor * kernel


def mirrored_clayton_kernel_log(u_a: np.ndarray, u_b: np.ndarray, zeta: float) -> np.ndarray:
    """Compute log of mirrored Clayton copula density kernel for all (a,b) pairs.

    The mirrored (survival) Clayton copula density is:
      c_tilde(u_a, u_b; zeta) = (1+zeta) * A_{ab}^zeta
    where:
      A_{ab}^zeta = [(1-u_a)^(-zeta) + (1-u_b)^(-zeta) - 1]^(-1/zeta - 2)
                    * [(1-u_a)*(1-u_b)]^(-zeta-1)

    This returns log(A_{ab}^zeta), computed in log-domain for stability.

    Parameters
    ----------
    u_a : 1D array of length n (plotting positions for first leaf)
    u_b : 1D array of length n (plotting positions for second leaf)
    zeta : mirrored Clayton parameter (> 0)

    Returns
    -------
    log_A : n x n matrix of log(A_{ab}^zeta)
    """
    # Custom — Context7 found no library equivalent (paper Eq. 12-13)
    v_a = 1.0 - u_a  # (n,)  = 1 - u_a
    v_b = 1.0 - u_b  # (n,)

    log_v_a = np.log(v_a)  # (n,)
    log_v_b = np.log(v_b)  # (n,)

    # log[(1-u_a)^(-zeta)] = -zeta * log(v_a)
    log_term_a = -zeta * log_v_a  # (n,)
    log_term_b = -zeta * log_v_b  # (n,)

    # Bracket: (1-u_a)^(-zeta) + (1-u_b)^(-zeta) - 1  for each (a,b) pair
    # Use numerically stable log-sum: log(e^x + e^y - 1)
    # For large x,y: log(e^x + e^y - 1) ≈ log(e^max(x,y)*(1 + e^(min-max) - e^(-max)))
    # But since zeta>0 and v<1, terms are > 1, so bracket > 1 and > 0

    term_a_mat = log_term_a[:, None] * np.ones((1, len(u_b)))  # (n,n)
    term_b_mat = log_term_b[None, :] * np.ones((len(u_a), 1))  # (n,n)

    # bracket = exp(log_term_a) + exp(log_term_b) - 1
    bracket = np.exp(log_term_a[:, None]) + np.exp(log_term_b[None, :]) - 1.0
    # Clip to avoid log(0) if bracket is near 0 (shouldn't happen with midpoint u)
    bracket = np.maximum(bracket, 1e-300)
    log_bracket = np.log(bracket)  # (n, n)

    # log(A_{ab}) = (-1/zeta - 2) * log_bracket + (-zeta - 1) * (log_v_a + log_v_b)
    log_A = (-1.0 / zeta - 2.0) * log_bracket + (-zeta - 1.0) * (
        log_v_a[:, None] + log_v_b[None, :]
    )
    return log_A


# ---------------------------------------------------------------------------
# Within-portfolio weighted distributions
# ---------------------------------------------------------------------------

def compute_within_portfolio_gaussian(
    mr_samples: np.ndarray,
    cr_samples: np.ndarray,
    z_mr: np.ndarray,
    z_cr: np.ndarray,
    gamma: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Build the within-portfolio Gaussian copula weighted loss distribution.

    For each pair (a, b) with a in {market scenarios}, b in {credit scenarios}:
      local loss: l_{ab} = mr_samples[a] + cr_samples[b]
      log-weight: log_w_{ab} = gaussian_kernel_log(z_a, z_b, gamma)

    Normalize weights to sum exactly to 1.

    Parameters
    ----------
    mr_samples : sorted market-risk samples, shape (n,)
    cr_samples : sorted credit-risk samples, shape (n,)
    z_mr : Gaussian quantiles for market leaf, shape (n,)
    z_cr : Gaussian quantiles for credit leaf, shape (n,)
    gamma : within-portfolio Gaussian correlation

    Returns
    -------
    sorted_losses : loss values sorted ascending, shape (n^2,)
    weights : normalized probability weights, shape (n^2,)
    log_unnorm : unnormalized log-weights (before softmax), shape (n^2,)
    """
    # Custom — Context7 found no library equivalent (paper Eq. 10)
    n = len(mr_samples)
    # All pairwise losses: l_{ab} = mr[a] + cr[b]
    loss_matrix = mr_samples[:, None] + cr_samples[None, :]  # (n, n)
    log_w_matrix = gaussian_kernel_log(z_mr, z_cr, gamma)  # (n, n)

    # Flatten
    losses_flat = loss_matrix.ravel()  # (n^2,)
    log_w_flat = log_w_matrix.ravel()  # (n^2,)

    # Sort by loss ascending
    sort_idx = np.argsort(losses_flat, kind="stable")
    sorted_losses = losses_flat[sort_idx]
    log_unnorm = log_w_flat[sort_idx]

    # Normalize via log-sum-exp for numerical stability
    log_sum = logsumexp(log_unnorm)
    log_norm = log_unnorm - log_sum
    weights = np.exp(log_norm)

    # Numerical safety: ensure weights sum to exactly 1
    weights = weights / weights.sum()

    return sorted_losses, weights, log_unnorm


def compute_within_portfolio_mirrored_clayton(
    mr_samples: np.ndarray,
    cr_samples: np.ndarray,
    u_mr: np.ndarray,
    u_cr: np.ndarray,
    zeta: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Build the within-portfolio mirrored Clayton copula weighted loss distribution.

    For each pair (a, b):
      local loss: l_{ab} = mr_samples[a] + cr_samples[b]
      log-weight: log_w_{ab} = mirrored_clayton_kernel_log(u_a, u_b, zeta) + log(1+zeta)
                               (the (1+zeta)/n^2 factor; n^2 cancels in normalization)

    Parameters
    ----------
    mr_samples : sorted market-risk samples, shape (n,)
    cr_samples : sorted credit-risk samples, shape (n,)
    u_mr : plotting positions for market leaf
    u_cr : plotting positions for credit leaf
    zeta : mirrored Clayton within-portfolio parameter

    Returns
    -------
    sorted_losses, weights, log_unnorm  (same convention as Gaussian)
    """
    # Custom — Context7 found no library equivalent (paper Eq. 12)
    loss_matrix = mr_samples[:, None] + cr_samples[None, :]  # (n, n)
    log_A_matrix = mirrored_clayton_kernel_log(u_mr, u_cr, zeta)  # (n, n)

    # Raw log-weight includes log(1+zeta); (1/n^2) cancels in normalization
    log_w_matrix = np.log(1.0 + zeta) + log_A_matrix  # (n, n)

    losses_flat = loss_matrix.ravel()
    log_w_flat = log_w_matrix.ravel()

    sort_idx = np.argsort(losses_flat, kind="stable")
    sorted_losses = losses_flat[sort_idx]
    log_unnorm = log_w_flat[sort_idx]

    log_sum = logsumexp(log_unnorm)
    log_norm = log_unnorm - log_sum
    weights = np.exp(log_norm)
    weights = weights / weights.sum()

    return sorted_losses, weights, log_unnorm


# ---------------------------------------------------------------------------
# CDF evaluation
# ---------------------------------------------------------------------------

def compute_portfolio_cdf(
    loss_val: float,
    sorted_losses: np.ndarray,
    weights: np.ndarray,
) -> float:
    """Evaluate the within-portfolio CDF F_i(loss_val).

    Uses right-continuity convention Theta(0)=1, so F_i(l) counts
    scenarios where l_{ab} <= l.

    Parameters
    ----------
    loss_val : evaluation point
    sorted_losses : sorted ascending array of scenario losses
    weights : normalized probability weights (same order as sorted_losses)

    Returns
    -------
    F_i(loss_val) in [0, 1]
    """
    # searchsorted with 'right' gives index of first element > loss_val
    # so all elements at indices < that index satisfy loss <= loss_val
    # Theta(0)=1 means l_{ab}=loss_val is included (<=)
    idx = np.searchsorted(sorted_losses, loss_val, side="right")
    if idx == 0:
        return 0.0
    return float(np.sum(weights[:idx]))


def compute_portfolio_cdf_batch(
    loss_vals: np.ndarray,
    sorted_losses: np.ndarray,
    weights: np.ndarray,
) -> np.ndarray:
    """Vectorized batch CDF evaluation.

    Parameters
    ----------
    loss_vals : array of evaluation points, shape (m,)
    sorted_losses : sorted scenario losses
    weights : normalized probability weights

    Returns
    -------
    cdf_vals : array of F_i values, shape (m,)
    """
    cum_weights = np.cumsum(weights)
    # searchsorted 'right': index is where to insert to keep sorted, i.e., number of <= entries
    indices = np.searchsorted(sorted_losses, loss_vals, side="right")
    cdf_vals = np.where(indices == 0, 0.0, cum_weights[np.minimum(indices, len(cum_weights)) - 1])
    # Handle indices beyond array length
    cdf_vals = np.where(indices >= len(cum_weights), cum_weights[-1], cdf_vals)
    return cdf_vals
