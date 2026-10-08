"""
Deterministic tail risk measures for finite weighted loss distributions.

Implements:
  - Alpha-quantile extraction (paper Eq. 17)
  - Atom-corrected ETL (paper Eq. 16)
  - Market and credit ETL contributions (paper Eq. 19-23)

All computations are from the deterministic finite weighted support.
# Custom — Context7 found no library equivalent for paper's atom-corrected ETL formulas
"""
from __future__ import annotations

from typing import List, Optional
import numpy as np
import pandas as pd


def compute_quantile(
    sorted_losses: np.ndarray,
    cum_weights: np.ndarray,
    alpha: float = 0.99,
) -> float:
    """Extract the alpha-quantile from a finite weighted loss distribution.

    Uses paper's convention (Eq. 17):
      q_alpha(L) = l_(m_alpha) where m_alpha = first index with cumulative weight >= alpha.
    Right-continuity (Theta(0)=1) is assumed already embedded in cum_weights.

    Parameters
    ----------
    sorted_losses : sorted ascending scenario losses, shape (N,)
    cum_weights : cumulative probability weights (sum to 1), shape (N,)
    alpha : confidence level in (0, 1)

    Returns
    -------
    q_alpha : scalar quantile value
    """
    idx = np.searchsorted(cum_weights, alpha, side="left")
    idx = min(idx, len(sorted_losses) - 1)
    return float(sorted_losses[idx])


def compute_etl(
    sorted_losses: np.ndarray,
    weights: np.ndarray,
    cum_weights: np.ndarray,
    alpha: float = 0.99,
) -> dict:
    """Compute atom-corrected ETL from finite weighted loss distribution.

    Paper Eq. (16):
      q = q_alpha(L)
      M_>(q) = sum_r p_r * 1{l_r > q}
      M_=(q) = sum_r p_r * 1{l_r == q}
      eta(q) = (1-alpha) - M_>(q)
      ETL_alpha = (1/(1-alpha)) * [sum_{l_r>q} p_r*l_r + eta*q]

    Parameters
    ----------
    sorted_losses : sorted ascending losses, shape (N,)
    weights : normalized probability weights, shape (N,)
    cum_weights : cumulative weights, shape (N,)
    alpha : confidence level

    Returns
    -------
    dict with keys: etl, M_gt, M_eq, eta, q
    """
    q = compute_quantile(sorted_losses, cum_weights, alpha)

    mask_gt = sorted_losses > q
    mask_eq = sorted_losses == q

    M_gt = float(np.sum(weights[mask_gt]))
    M_eq = float(np.sum(weights[mask_eq]))
    eta = max((1.0 - alpha) - M_gt, 0.0)

    wl_gt = float(np.sum(weights[mask_gt] * sorted_losses[mask_gt]))
    etl = (wl_gt + eta * q) / (1.0 - alpha)

    return {
        "etl": etl,
        "M_gt": M_gt,
        "M_eq": M_eq,
        "eta": eta,
        "q": q,
    }


def compute_etl_contributions(
    sorted_losses: np.ndarray,
    weights: np.ndarray,
    cum_weights: np.ndarray,
    mr_losses: np.ndarray,
    cr_losses: np.ndarray,
    alpha: float = 0.99,
) -> dict:
    """Compute market and credit ETL contributions with atom correction.

    Paper Eq. (19)-(23):
      Let A_q = {r : l_r == q} be the atom at the quantile.
      ETLCont_MR = (1/(1-alpha)) * [sum_{l_r>q} p_r*l_r^MR + eta * E[L^MR | L=q]]
      ETLCont_CR = (1/(1-alpha)) * [sum_{l_r>q} p_r*l_r^CR + eta * E[L^CR | L=q]]
      where E[L^MR | L=q] = sum_{r in A_q} (p_r / M_=(q)) * l_r^MR

    Verification: ETL = ETLCont_MR + ETLCont_CR (decomposition identity)

    Parameters
    ----------
    sorted_losses : total sorted losses, shape (N,)
    weights : normalized weights, shape (N,)
    cum_weights : cumulative weights, shape (N,)
    mr_losses : market-risk component per scenario, shape (N,)
    cr_losses : credit-risk component per scenario, shape (N,)
    alpha : confidence level

    Returns
    -------
    dict with keys: etl_mr, etl_cr, etl_total, decomp_residual, M_gt, M_eq, eta, q
    """
    base = compute_etl(sorted_losses, weights, cum_weights, alpha)
    q = base["q"]
    eta = base["eta"]
    M_eq = base["M_eq"]
    inv_alpha = 1.0 / (1.0 - alpha)

    mask_gt = sorted_losses > q
    mask_eq = sorted_losses == q

    wmr_gt = float(np.sum(weights[mask_gt] * mr_losses[mask_gt]))
    wcr_gt = float(np.sum(weights[mask_gt] * cr_losses[mask_gt]))

    if M_eq > 0.0:
        # Conditional mean on the atom
        mr_atom = float(np.sum(weights[mask_eq] * mr_losses[mask_eq])) / M_eq
        cr_atom = float(np.sum(weights[mask_eq] * cr_losses[mask_eq])) / M_eq
    else:
        mr_atom = cr_atom = 0.0

    etl_mr = inv_alpha * (wmr_gt + eta * mr_atom)
    etl_cr = inv_alpha * (wcr_gt + eta * cr_atom)

    decomp_residual = abs(etl_mr + etl_cr - base["etl"])

    return {
        "etl_mr": etl_mr,
        "etl_cr": etl_cr,
        "etl_total": base["etl"],
        "decomp_residual": decomp_residual,
        "M_gt": base["M_gt"],
        "M_eq": M_eq,
        "eta": eta,
        "q": q,
    }


def compute_all_risk_measures(
    sorted_losses: np.ndarray,
    weights: np.ndarray,
    mr_losses: np.ndarray,
    cr_losses: np.ndarray,
    alphas: Optional[List[float]] = None,
) -> pd.DataFrame:
    """Compute quantile, ETL, and ETL contributions for multiple alpha levels.

    Parameters
    ----------
    sorted_losses : total sorted losses
    weights : normalized weights (must sum to 1)
    mr_losses : market-risk components (same order)
    cr_losses : credit-risk components
    alphas : list of confidence levels; defaults to [0.95, 0.975, 0.99, 0.995]

    Returns
    -------
    DataFrame with columns: alpha, q_alpha, etl, etl_mr, etl_cr,
                             mr_share, cr_share, decomp_residual
    """
    if alphas is None:
        alphas = [0.95, 0.975, 0.99, 0.995]

    cum_weights = np.cumsum(weights)
    rows = []
    for a in alphas:
        contrib = compute_etl_contributions(
            sorted_losses, weights, cum_weights, mr_losses, cr_losses, alpha=a
        )
        etl_total = contrib["etl_total"]
        mr_share = contrib["etl_mr"] / etl_total if etl_total > 0 else np.nan
        cr_share = contrib["etl_cr"] / etl_total if etl_total > 0 else np.nan
        rows.append({
            "alpha": a,
            "q_alpha": contrib["q"],
            "etl": etl_total,
            "etl_mr": contrib["etl_mr"],
            "etl_cr": contrib["etl_cr"],
            "mr_share": mr_share,
            "cr_share": cr_share,
            "decomp_residual": contrib["decomp_residual"],
        })
    return pd.DataFrame(rows)
