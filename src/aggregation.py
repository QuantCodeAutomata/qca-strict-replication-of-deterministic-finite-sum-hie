"""
Two-level hierarchical loss aggregation with deterministic finite-sum formulas.

Implements chunked n^4 computation to handle 1.6e9 scenarios for n=200
without storing all scenarios simultaneously.

Key design: process portfolio-1 scenarios in chunks; for each chunk compute
the full interaction with all portfolio-2 scenarios using numpy broadcasting.
Memory usage: O(chunk_size * n^2) instead of O(n^4).

# Custom — Context7 found no library equivalent for paper's hierarchical
# finite-sum aggregation formulas (paper Equations 10-17)
"""
from __future__ import annotations

import numpy as np
from scipy.stats import norm
from scipy.special import logsumexp

# Clipping bound for Phi^{-1} to avoid ±inf
_Z_CLIP = 37.0
_EPS = 1e-12


# ---------------------------------------------------------------------------
# Z-transform: map within-portfolio losses to Gaussian quantile domain
# ---------------------------------------------------------------------------

def compute_z_transforms(
    sorted_losses: np.ndarray,
    weights: np.ndarray,
) -> np.ndarray:
    """Compute Z_i(s) = Phi^{-1}(F_i_mid(l_s)) for each within-portfolio scenario.

    Uses the midpoint CDF convention (analogous to the midpoint plotting positions
    u_m=(m-0.5)/n used for the leaf marginals):

        F_i_mid(l_s) = cumsum(weights)[s] - weights[s]/2

    This prevents F_i(l_max)=1.0 and F_i(l_min)=0.0 at the support endpoints,
    which would make Z_i = ±∞ and cause the top-level Gaussian kernel to
    collapse the distribution to a single point.  The midpoint CDF is the natural
    analogue of the leaf plotting-position convention for the weighted within-
    portfolio distribution.  Implementation note: paper is silent on this
    convention but the right-continuous choice (cumsum without correction) leads
    to numerically degenerate results.

    Parameters
    ----------
    sorted_losses : sorted ascending scenario losses, shape (n^2,)
    weights : normalized probability weights, shape (n^2,)

    Returns
    -------
    z_transforms : Phi^{-1}(F_i_mid(l_s)) clipped to [-37, 37], shape (n^2,)
    """
    # Midpoint CDF: cumulative weight up to (but not including) l_s + half the
    # weight at l_s.  Avoids F=1 at l_max and F=0 at l_min.
    cum_weights = np.cumsum(weights) - 0.5 * weights

    # Clip to avoid Phi^{-1}(0) = -inf and Phi^{-1}(1) = +inf
    cum_clipped = np.clip(cum_weights, _EPS, 1.0 - _EPS)

    # Phi^{-1} via scipy.stats.norm.ppf — Context7 confirmed
    z_vals = norm.ppf(cum_clipped)

    # Hard clip to [-37, 37]
    z_vals = np.clip(z_vals, -_Z_CLIP, _Z_CLIP)
    return z_vals


def compute_within_portfolio_cdf_midpoint(weights: np.ndarray) -> np.ndarray:
    """Midpoint CDF for mirrored Clayton top-level copula mapping.

    Returns F_i_mid(l_s) = cumsum[s] - weights[s]/2, clipped to (eps, 1-eps).
    Prevents (1-F)=0 which would cause log-singularities in the Clayton kernel.
    """
    f_mid = np.cumsum(weights) - 0.5 * weights
    return np.clip(f_mid, _EPS, 1.0 - _EPS)


# ---------------------------------------------------------------------------
# Top-level Gaussian hierarchy: chunked CDF, quantile, and ETL
# ---------------------------------------------------------------------------

def _gaussian_top_log_weights_chunk(
    log_unnorm1_chunk: np.ndarray,  # (chunk,)
    z1_chunk: np.ndarray,           # (chunk,)
    log_unnorm2: np.ndarray,        # (n2,)
    z2: np.ndarray,                 # (n2,)
    rho_top: float,
) -> np.ndarray:
    """Compute top-level log-weights for a chunk of portfolio-1 scenarios.

    log W_{st} = log_unnorm1[s] + log_unnorm2[t]
               - rho_top^2/(2*(1-rho_top^2)) * (Z1[s]^2 + Z2[t]^2)
               + rho_top/(1-rho_top^2) * Z1[s] * Z2[t]

    Parameters
    ----------
    log_unnorm1_chunk : log unnormalized weights for portfolio-1 chunk, shape (chunk,)
    z1_chunk : Z-transforms for portfolio-1 chunk, shape (chunk,)
    log_unnorm2 : log unnormalized weights for portfolio-2, shape (n2,)
    z2 : Z-transforms for portfolio-2, shape (n2,)
    rho_top : top-level Gaussian correlation

    Returns
    -------
    log_w_chunk : shape (chunk, n2)
    """
    rho2 = rho_top ** 2
    denom = 1.0 - rho2
    diag_factor = rho2 / (2.0 * denom)
    cross_factor = rho_top / denom

    # Portfolio-1 component (chunk,)
    log_w1_adj = log_unnorm1_chunk - diag_factor * z1_chunk ** 2

    # Portfolio-2 component (n2,)
    log_w2_adj = log_unnorm2 - diag_factor * z2 ** 2

    # Cross term: cross_factor * Z1[s] * Z2[t]
    cross = cross_factor * np.outer(z1_chunk, z2)  # (chunk, n2)

    # Total log-weight
    log_w = log_w1_adj[:, None] + log_w2_adj[None, :] + cross  # (chunk, n2)
    return log_w


def compute_cdf_gaussian_top(
    l_val: float,
    losses1: np.ndarray,
    log_unnorm1: np.ndarray,
    z1: np.ndarray,
    losses2: np.ndarray,
    log_unnorm2: np.ndarray,
    z2: np.ndarray,
    rho_top: float,
    chunk_size: int = 200,
) -> tuple[float, float]:
    """Compute F_L(l_val) for the Gaussian top-level hierarchy.

    Uses chunked n^2 x n^2 computation to avoid n^4 memory allocation.

    Returns
    -------
    (F_L_val, norm_const) where F_L_val is normalized CDF value
    and norm_const is the sum of all unnormalized weights (for diagnostics).
    """
    # Custom — Context7 found no library equivalent
    n1 = len(losses1)
    total_weight_below = 0.0
    total_weight = 0.0

    for i_start in range(0, n1, chunk_size):
        i_end = min(i_start + chunk_size, n1)
        l1_chunk = losses1[i_start:i_end]   # (chunk,)
        lw1_chunk = log_unnorm1[i_start:i_end]
        z1_chunk = z1[i_start:i_end]

        log_w = _gaussian_top_log_weights_chunk(
            lw1_chunk, z1_chunk, log_unnorm2, z2, rho_top
        )  # (chunk, n2)
        w = np.exp(log_w)  # (chunk, n2)

        # Total loss matrix and indicator
        l_total = l1_chunk[:, None] + losses2[None, :]  # (chunk, n2)
        indicator = l_total <= l_val  # Theta(0)=1: <=

        total_weight_below += np.sum(w * indicator)
        total_weight += np.sum(w)

    if total_weight == 0.0:
        return 0.0, 0.0

    return float(total_weight_below / total_weight), float(total_weight)


def _enumerate_gaussian_all_weights(
    losses1: np.ndarray,
    log_unnorm1: np.ndarray,
    z1: np.ndarray,
    losses2: np.ndarray,
    log_unnorm2: np.ndarray,
    z2: np.ndarray,
    rho_top: float,
    chunk_size: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Enumerate every top-level Gaussian scenario (total loss, normalised weight).

    Used when total scenarios (n1² × n2²) fit comfortably in memory.

    Returns
    -------
    sorted_losses : shape (n1²*n2²,), sorted ascending
    sorted_weights : shape (n1²*n2²,), normalised (sum == 1)
    """
    n1, n2 = len(losses1), len(losses2)
    total = n1 * n2
    all_losses = np.empty(total, dtype=np.float64)
    all_log_w  = np.empty(total, dtype=np.float64)

    for i_start in range(0, n1, chunk_size):
        i_end = min(i_start + chunk_size, n1)
        log_w_chunk = _gaussian_top_log_weights_chunk(
            log_unnorm1[i_start:i_end], z1[i_start:i_end],
            log_unnorm2, z2, rho_top,
        )  # (chunk, n2)
        l_chunk = losses1[i_start:i_end, None] + losses2[None, :]  # (chunk, n2)

        chunk_len = (i_end - i_start) * n2
        base = i_start * n2
        all_losses[base:base + chunk_len] = l_chunk.ravel()
        all_log_w [base:base + chunk_len] = log_w_chunk.ravel()

    log_sum = logsumexp(all_log_w)
    weights = np.exp(all_log_w - log_sum)
    weights /= weights.sum()  # enforce exact sum = 1

    sort_idx = np.argsort(all_losses, kind="stable")
    return all_losses[sort_idx], weights[sort_idx]


def compute_quantile_gaussian_top(
    losses1: np.ndarray,
    log_unnorm1: np.ndarray,
    z1: np.ndarray,
    losses2: np.ndarray,
    log_unnorm2: np.ndarray,
    z2: np.ndarray,
    rho_top: float,
    alpha: float = 0.99,
    chunk_size: int = 200,
    n_candidates: int = 500,
    exact_threshold: int = 500_000,
) -> float:
    """Find the alpha-quantile of the top-level Gaussian aggregate loss.

    When the total number of top-level scenarios (n1² × n2²) does not exceed
    *exact_threshold*, all scenarios are enumerated and the quantile is extracted
    from the exact discrete weighted support (guaranteeing the result is an
    actual support point). Otherwise a binary search over a fine grid is used.

    Parameters
    ----------
    n_candidates : grid points for the fallback grid search
    exact_threshold : scenario-count cutoff for switching to exact enumeration
    """
    n_total = len(losses1) * len(losses2)

    if n_total <= exact_threshold:
        # Exact path: enumerate every scenario, sort, find discrete quantile
        sorted_losses, sorted_weights = _enumerate_gaussian_all_weights(
            losses1, log_unnorm1, z1, losses2, log_unnorm2, z2, rho_top, chunk_size
        )
        cum_w = np.cumsum(sorted_weights)
        m_alpha = int(np.searchsorted(cum_w, alpha, side="left"))
        m_alpha = min(m_alpha, len(sorted_losses) - 1)
        return float(sorted_losses[m_alpha])

    # Fallback: grid-based binary search (large n)
    l_min = losses1[0] + losses2[0]
    l_max = losses1[-1] + losses2[-1]
    candidates = np.linspace(l_min, l_max, n_candidates)

    lo, hi = 0, len(candidates) - 1
    q_alpha = candidates[hi]

    while lo <= hi:
        mid = (lo + hi) // 2
        cdf_val, _ = compute_cdf_gaussian_top(
            candidates[mid], losses1, log_unnorm1, z1,
            losses2, log_unnorm2, z2, rho_top, chunk_size
        )
        if cdf_val >= alpha:
            q_alpha = candidates[mid]
            hi = mid - 1
        else:
            lo = mid + 1

    return float(q_alpha)


def compute_etl_gaussian_top(
    q_alpha: float,
    losses1: np.ndarray,
    log_unnorm1: np.ndarray,
    z1: np.ndarray,
    mr1: np.ndarray,
    cr1: np.ndarray,
    losses2: np.ndarray,
    log_unnorm2: np.ndarray,
    z2: np.ndarray,
    mr2: np.ndarray,
    cr2: np.ndarray,
    rho_top: float,
    alpha: float = 0.99,
    chunk_size: int = 200,
) -> dict:
    """Compute atom-corrected ETL and market/credit contributions (Gaussian hierarchy).

    Atom-corrected ETL (paper Eq. 16):
      q = q_alpha
      M_>(q) = sum_r p_r * 1{l_r > q}
      M_=(q) = sum_r p_r * 1{l_r == q}
      eta(q) = (1-alpha) - M_>(q)
      ETL = (1/(1-alpha)) * [sum_{l_r>q} p_r*l_r + eta*q]

    ETL contributions (paper Eq. 19-23):
      ETLCont_MR = (1/(1-alpha)) * [sum_{l_r>q} p_r*l_r^MR + eta * E[L^MR | L=q]]
      where E[L^MR | L=q] = sum_{r: l_r=q} (p_r/M_=(q)) * l_r^MR

    Parameters
    ----------
    q_alpha : quantile value
    losses1, losses2 : within-portfolio loss arrays (sorted)
    log_unnorm1, log_unnorm2 : unnormalized log-weights
    z1, z2 : Z-transforms
    mr1, cr1, mr2, cr2 : market and credit components of losses per scenario
    rho_top : top-level Gaussian correlation
    alpha : confidence level

    Returns
    -------
    dict with keys: etl, mr_contrib, cr_contrib, M_gt, M_eq, eta, norm_const
    """
    n1 = len(losses1)
    inv_alpha = 1.0 / (1.0 - alpha)

    total_weight = 0.0
    M_gt = 0.0   # sum p_r * 1{l>q}
    M_eq = 0.0   # sum p_r * 1{l=q}
    wl_gt = 0.0  # sum p_r*l_r for l>q (unnormalized)
    wmr_gt = 0.0
    wcr_gt = 0.0
    wmr_eq = 0.0  # sum p_r*l_r^MR for l=q
    wcr_eq = 0.0

    for i_start in range(0, n1, chunk_size):
        i_end = min(i_start + chunk_size, n1)
        l1_c = losses1[i_start:i_end]
        lw1_c = log_unnorm1[i_start:i_end]
        z1_c = z1[i_start:i_end]
        mr1_c = mr1[i_start:i_end]
        cr1_c = cr1[i_start:i_end]

        log_w = _gaussian_top_log_weights_chunk(
            lw1_c, z1_c, log_unnorm2, z2, rho_top
        )
        w = np.exp(log_w)  # (chunk, n2)

        l_total = l1_c[:, None] + losses2[None, :]  # (chunk, n2)
        mr_total = mr1_c[:, None] + mr2[None, :]
        cr_total = cr1_c[:, None] + cr2[None, :]

        mask_gt = l_total > q_alpha
        mask_eq = np.abs(l_total - q_alpha) < 1e-12

        total_weight += np.sum(w)
        M_gt += np.sum(w[mask_gt])
        M_eq += np.sum(w[mask_eq])
        wl_gt += np.sum(w[mask_gt] * l_total[mask_gt])
        wmr_gt += np.sum(w[mask_gt] * mr_total[mask_gt])
        wcr_gt += np.sum(w[mask_gt] * cr_total[mask_gt])
        wmr_eq += np.sum(w[mask_eq] * mr_total[mask_eq])
        wcr_eq += np.sum(w[mask_eq] * cr_total[mask_eq])

    if total_weight == 0.0:
        return {
            "etl": np.nan, "mr_contrib": np.nan, "cr_contrib": np.nan,
            "M_gt": 0.0, "M_eq": 0.0, "eta": 0.0, "norm_const": 0.0,
        }

    # Normalize all accumulated quantities
    M_gt_n = M_gt / total_weight
    M_eq_n = M_eq / total_weight
    wl_gt_n = wl_gt / total_weight
    wmr_gt_n = wmr_gt / total_weight
    wcr_gt_n = wcr_gt / total_weight
    wmr_eq_n = wmr_eq / total_weight
    wcr_eq_n = wcr_eq / total_weight

    eta = (1.0 - alpha) - M_gt_n
    eta = max(eta, 0.0)  # numerical safety

    etl = inv_alpha * (wl_gt_n + eta * q_alpha)

    if M_eq_n > 0.0:
        # Conditional market/credit mean at the atom
        mr_atom = wmr_eq_n / M_eq_n
        cr_atom = wcr_eq_n / M_eq_n
    else:
        mr_atom = 0.0
        cr_atom = 0.0

    etl_mr = inv_alpha * (wmr_gt_n + eta * mr_atom)
    etl_cr = inv_alpha * (wcr_gt_n + eta * cr_atom)

    return {
        "etl": float(etl),
        "mr_contrib": float(etl_mr),
        "cr_contrib": float(etl_cr),
        "M_gt": float(M_gt_n),
        "M_eq": float(M_eq_n),
        "eta": float(eta),
        "norm_const": float(total_weight),
    }


# ---------------------------------------------------------------------------
# Top-level mirrored Clayton hierarchy: chunked CDF, quantile, and ETL
# ---------------------------------------------------------------------------

def _clayton_top_log_weight_chunk(
    log_unnorm1_chunk: np.ndarray,   # (chunk,)
    f1_chunk: np.ndarray,             # (chunk,) – CDF values F_1(l_s)
    log_unnorm2: np.ndarray,          # (n2,)
    f2: np.ndarray,                   # (n2,) – CDF values F_2(l_t)
    theta: float,
) -> np.ndarray:
    """Compute top-level log-weights for mirrored Clayton hierarchy.

    log W_{st}^C = log_unnorm1[s] + log_unnorm2[t] + log_B(F1[s], F2[t], theta)

    log_B(f1, f2, theta) =
      (-1/theta - 2) * log[(1-f1)^(-theta) + (1-f2)^(-theta) - 1]
      + (-theta - 1) * [log(1-f1) + log(1-f2)]

    Parameters
    ----------
    f1_chunk : F_1(l_s) values for chunk, clipped to (eps, 1-eps)
    f2 : F_2(l_t) values for all portfolio-2 scenarios

    Returns
    -------
    log_w_chunk : shape (chunk, n2)
    """
    # Custom — Context7 found no library equivalent (paper Eq. 13)
    v1 = np.clip(1.0 - f1_chunk, _EPS, 1.0 - _EPS)  # (chunk,)
    v2 = np.clip(1.0 - f2, _EPS, 1.0 - _EPS)         # (n2,)

    log_v1 = np.log(v1)  # (chunk,)
    log_v2 = np.log(v2)  # (n2,)

    # (1-f1)^(-theta) for chunk
    term1 = np.exp(-theta * log_v1)  # (chunk,)
    # (1-f2)^(-theta) for all portfolio-2
    term2 = np.exp(-theta * log_v2)  # (n2,)

    # Bracket = term1 + term2 - 1, shape (chunk, n2)
    bracket = term1[:, None] + term2[None, :] - 1.0
    bracket = np.maximum(bracket, 1e-300)
    log_bracket = np.log(bracket)  # (chunk, n2)

    # log_B = (-1/theta - 2)*log_bracket + (-theta-1)*(log_v1 + log_v2)
    log_B = (-1.0 / theta - 2.0) * log_bracket + (-theta - 1.0) * (
        log_v1[:, None] + log_v2[None, :]
    )

    log_w = log_unnorm1_chunk[:, None] + log_unnorm2[None, :] + log_B
    return log_w


def compute_cdf_clayton_top(
    l_val: float,
    losses1: np.ndarray,
    log_unnorm1: np.ndarray,
    f1_cdf: np.ndarray,
    losses2: np.ndarray,
    log_unnorm2: np.ndarray,
    f2_cdf: np.ndarray,
    theta: float,
    chunk_size: int = 200,
) -> tuple[float, float]:
    """Compute F_L(l_val) for the mirrored Clayton top-level hierarchy."""
    # Custom — Context7 found no library equivalent
    n1 = len(losses1)
    total_weight_below = 0.0
    total_weight = 0.0

    for i_start in range(0, n1, chunk_size):
        i_end = min(i_start + chunk_size, n1)
        l1_c = losses1[i_start:i_end]
        lw1_c = log_unnorm1[i_start:i_end]
        f1_c = f1_cdf[i_start:i_end]

        log_w = _clayton_top_log_weight_chunk(lw1_c, f1_c, log_unnorm2, f2_cdf, theta)
        w = np.exp(log_w)

        l_total = l1_c[:, None] + losses2[None, :]
        indicator = l_total <= l_val

        total_weight_below += np.sum(w * indicator)
        total_weight += np.sum(w)

    if total_weight == 0.0:
        return 0.0, 0.0
    return float(total_weight_below / total_weight), float(total_weight)


def compute_quantile_clayton_top(
    losses1: np.ndarray,
    log_unnorm1: np.ndarray,
    f1_cdf: np.ndarray,
    losses2: np.ndarray,
    log_unnorm2: np.ndarray,
    f2_cdf: np.ndarray,
    theta: float,
    alpha: float = 0.99,
    chunk_size: int = 200,
    n_candidates: int = 500,
) -> float:
    """Find the alpha-quantile of the top-level mirrored Clayton aggregate loss."""
    l_min = losses1[0] + losses2[0]
    l_max = losses1[-1] + losses2[-1]
    candidates = np.linspace(l_min, l_max, n_candidates)

    lo, hi = 0, len(candidates) - 1
    q_alpha = candidates[hi]

    while lo <= hi:
        mid = (lo + hi) // 2
        cdf_val, _ = compute_cdf_clayton_top(
            candidates[mid], losses1, log_unnorm1, f1_cdf,
            losses2, log_unnorm2, f2_cdf, theta, chunk_size
        )
        if cdf_val >= alpha:
            q_alpha = candidates[mid]
            hi = mid - 1
        else:
            lo = mid + 1

    return float(q_alpha)


def compute_etl_clayton_top(
    q_alpha: float,
    losses1: np.ndarray,
    log_unnorm1: np.ndarray,
    f1_cdf: np.ndarray,
    mr1: np.ndarray,
    cr1: np.ndarray,
    losses2: np.ndarray,
    log_unnorm2: np.ndarray,
    f2_cdf: np.ndarray,
    mr2: np.ndarray,
    cr2: np.ndarray,
    theta: float,
    alpha: float = 0.99,
    chunk_size: int = 200,
) -> dict:
    """Compute atom-corrected ETL and market/credit contributions (mirrored Clayton)."""
    n1 = len(losses1)
    inv_alpha = 1.0 / (1.0 - alpha)

    total_weight = 0.0
    M_gt = M_eq = 0.0
    wl_gt = wmr_gt = wcr_gt = 0.0
    wmr_eq = wcr_eq = 0.0

    for i_start in range(0, n1, chunk_size):
        i_end = min(i_start + chunk_size, n1)
        l1_c = losses1[i_start:i_end]
        lw1_c = log_unnorm1[i_start:i_end]
        f1_c = f1_cdf[i_start:i_end]
        mr1_c = mr1[i_start:i_end]
        cr1_c = cr1[i_start:i_end]

        log_w = _clayton_top_log_weight_chunk(lw1_c, f1_c, log_unnorm2, f2_cdf, theta)
        w = np.exp(log_w)

        l_total = l1_c[:, None] + losses2[None, :]
        mr_total = mr1_c[:, None] + mr2[None, :]
        cr_total = cr1_c[:, None] + cr2[None, :]

        mask_gt = l_total > q_alpha
        mask_eq = np.abs(l_total - q_alpha) < 1e-12

        total_weight += np.sum(w)
        M_gt += np.sum(w[mask_gt])
        M_eq += np.sum(w[mask_eq])
        wl_gt += np.sum(w[mask_gt] * l_total[mask_gt])
        wmr_gt += np.sum(w[mask_gt] * mr_total[mask_gt])
        wcr_gt += np.sum(w[mask_gt] * cr_total[mask_gt])
        wmr_eq += np.sum(w[mask_eq] * mr_total[mask_eq])
        wcr_eq += np.sum(w[mask_eq] * cr_total[mask_eq])

    if total_weight == 0.0:
        return {
            "etl": np.nan, "mr_contrib": np.nan, "cr_contrib": np.nan,
            "M_gt": 0.0, "M_eq": 0.0, "eta": 0.0, "norm_const": 0.0,
        }

    M_gt_n = M_gt / total_weight
    M_eq_n = M_eq / total_weight
    wl_gt_n = wl_gt / total_weight
    wmr_gt_n = wmr_gt / total_weight
    wcr_gt_n = wcr_gt / total_weight
    wmr_eq_n = wmr_eq / total_weight
    wcr_eq_n = wcr_eq / total_weight

    eta = max((1.0 - alpha) - M_gt_n, 0.0)
    etl = inv_alpha * (wl_gt_n + eta * q_alpha)

    if M_eq_n > 0.0:
        mr_atom = wmr_eq_n / M_eq_n
        cr_atom = wcr_eq_n / M_eq_n
    else:
        mr_atom = cr_atom = 0.0

    etl_mr = inv_alpha * (wmr_gt_n + eta * mr_atom)
    etl_cr = inv_alpha * (wcr_gt_n + eta * cr_atom)

    return {
        "etl": float(etl),
        "mr_contrib": float(etl_mr),
        "cr_contrib": float(etl_cr),
        "M_gt": float(M_gt_n),
        "M_eq": float(M_eq_n),
        "eta": float(eta),
        "norm_const": float(total_weight),
    }


# ---------------------------------------------------------------------------
# Single-pass histogram builders — replace multi-call CDF/quantile/ETL loops
# ---------------------------------------------------------------------------

def _build_histogram_gaussian_top(
    losses1: np.ndarray,
    log_unnorm1: np.ndarray,
    z1: np.ndarray,
    mr1: np.ndarray,
    cr1: np.ndarray,
    losses2: np.ndarray,
    log_unnorm2: np.ndarray,
    z2: np.ndarray,
    mr2: np.ndarray,
    cr2: np.ndarray,
    rho_top: float,
    n_bins: int = 2000,
    chunk_size: int = 200,
) -> dict:
    """Build weighted loss histogram for the Gaussian hierarchy in ONE pass.

    Returns a dict with:
      bin_edges (n_bins+1,), hist_w (n_bins,), hist_mr (n_bins,),
      hist_cr (n_bins,), total_weight, bin_mids (n_bins,)

    Using a histogram avoids repeated n^4 passes for quantile/CDF/ETL queries.
    CDF resolution: (l_max - l_min) / n_bins.

    # Custom — Context7 found no library equivalent (paper Equations 10-11)
    """
    l_min = float(losses1[0] + losses2[0])
    l_max = float(losses1[-1] + losses2[-1])
    bin_edges = np.linspace(l_min, l_max, n_bins + 1)
    bin_mids = 0.5 * (bin_edges[:-1] + bin_edges[1:])

    hist_w = np.zeros(n_bins, dtype=np.float64)
    hist_mr = np.zeros(n_bins, dtype=np.float64)
    hist_cr = np.zeros(n_bins, dtype=np.float64)
    total_weight = 0.0
    n1 = len(losses1)

    for i_start in range(0, n1, chunk_size):
        i_end = min(i_start + chunk_size, n1)
        l1_c = losses1[i_start:i_end]
        lw1_c = log_unnorm1[i_start:i_end]
        z1_c = z1[i_start:i_end]
        mr1_c = mr1[i_start:i_end]
        cr1_c = cr1[i_start:i_end]

        log_w = _gaussian_top_log_weights_chunk(lw1_c, z1_c, log_unnorm2, z2, rho_top)
        w = np.exp(log_w)  # (chunk, n2)

        l_total = l1_c[:, None] + losses2[None, :]  # (chunk, n2)
        mr_total = mr1_c[:, None] + mr2[None, :]
        cr_total = cr1_c[:, None] + cr2[None, :]

        # Digitize into bins (right-continuity: include left endpoint)
        flat_l = l_total.ravel()
        flat_w = w.ravel()
        flat_mr = mr_total.ravel()
        flat_cr = cr_total.ravel()

        bin_idx = np.searchsorted(bin_edges, flat_l, side="right") - 1
        np.clip(bin_idx, 0, n_bins - 1, out=bin_idx)

        hist_w += np.bincount(bin_idx, weights=flat_w, minlength=n_bins)
        hist_mr += np.bincount(bin_idx, weights=flat_w * flat_mr, minlength=n_bins)
        hist_cr += np.bincount(bin_idx, weights=flat_w * flat_cr, minlength=n_bins)
        total_weight += flat_w.sum()

    return {
        "bin_edges": bin_edges,
        "bin_mids": bin_mids,
        "hist_w": hist_w,
        "hist_mr": hist_mr,
        "hist_cr": hist_cr,
        "total_weight": total_weight,
    }


def _build_histogram_clayton_top(
    losses1: np.ndarray,
    log_unnorm1: np.ndarray,
    f1_cdf: np.ndarray,
    mr1: np.ndarray,
    cr1: np.ndarray,
    losses2: np.ndarray,
    log_unnorm2: np.ndarray,
    f2_cdf: np.ndarray,
    mr2: np.ndarray,
    cr2: np.ndarray,
    theta: float,
    n_bins: int = 2000,
    chunk_size: int = 200,
) -> dict:
    """Build weighted loss histogram for the mirrored Clayton hierarchy in ONE pass.

    # Custom — Context7 found no library equivalent (paper Equations 12-13)
    """
    l_min = float(losses1[0] + losses2[0])
    l_max = float(losses1[-1] + losses2[-1])
    bin_edges = np.linspace(l_min, l_max, n_bins + 1)
    bin_mids = 0.5 * (bin_edges[:-1] + bin_edges[1:])

    hist_w = np.zeros(n_bins, dtype=np.float64)
    hist_mr = np.zeros(n_bins, dtype=np.float64)
    hist_cr = np.zeros(n_bins, dtype=np.float64)
    total_weight = 0.0
    n1 = len(losses1)

    for i_start in range(0, n1, chunk_size):
        i_end = min(i_start + chunk_size, n1)
        l1_c = losses1[i_start:i_end]
        lw1_c = log_unnorm1[i_start:i_end]
        f1_c = f1_cdf[i_start:i_end]
        mr1_c = mr1[i_start:i_end]
        cr1_c = cr1[i_start:i_end]

        log_w = _clayton_top_log_weight_chunk(lw1_c, f1_c, log_unnorm2, f2_cdf, theta)
        w = np.exp(log_w)  # (chunk, n2)

        l_total = l1_c[:, None] + losses2[None, :]
        mr_total = mr1_c[:, None] + mr2[None, :]
        cr_total = cr1_c[:, None] + cr2[None, :]

        flat_l = l_total.ravel()
        flat_w = w.ravel()
        flat_mr = mr_total.ravel()
        flat_cr = cr_total.ravel()

        bin_idx = np.searchsorted(bin_edges, flat_l, side="right") - 1
        np.clip(bin_idx, 0, n_bins - 1, out=bin_idx)

        hist_w += np.bincount(bin_idx, weights=flat_w, minlength=n_bins)
        hist_mr += np.bincount(bin_idx, weights=flat_w * flat_mr, minlength=n_bins)
        hist_cr += np.bincount(bin_idx, weights=flat_w * flat_cr, minlength=n_bins)
        total_weight += flat_w.sum()

    return {
        "bin_edges": bin_edges,
        "bin_mids": bin_mids,
        "hist_w": hist_w,
        "hist_mr": hist_mr,
        "hist_cr": hist_cr,
        "total_weight": total_weight,
    }


def cdf_from_histogram(
    l_val: float,
    hdata: dict,
) -> float:
    """Evaluate F_L(l_val) from a prebuilt histogram (right-continuous).

    Convention Theta(0)=1: bin with left_edge ≤ l_val is included.
    """
    bin_edges = hdata["bin_edges"]
    hist_w = hdata["hist_w"]
    tw = hdata["total_weight"]
    if tw == 0.0:
        return 0.0

    # Number of bins with left_edge ≤ l_val, i.e. right_edge > l_val
    # searchsorted(edges, l_val, side='right') gives first edge > l_val
    # so all bins 0..(idx-1) have left_edge ≤ l_val
    idx = np.searchsorted(bin_edges, l_val, side="right")  # 1..n_bins+1
    idx = int(np.clip(idx, 0, len(hist_w)))
    return float(np.sum(hist_w[:idx]) / tw)


def quantile_from_histogram(
    hdata: dict,
    alpha: float,
) -> float:
    """Extract alpha-quantile from histogram: smallest l with F_L(l) >= alpha.

    Uses bin midpoints as representative loss values.
    """
    hist_w = hdata["hist_w"]
    bin_mids = hdata["bin_mids"]
    tw = hdata["total_weight"]
    if tw == 0.0:
        return float(bin_mids[-1])

    cum = np.cumsum(hist_w) / tw
    idx = int(np.searchsorted(cum, alpha, side="left"))
    idx = min(idx, len(bin_mids) - 1)
    return float(bin_mids[idx])


def etl_from_histogram(
    q_alpha: float,
    hdata: dict,
    alpha: float,
) -> dict:
    """Compute atom-corrected ETL and market/credit contributions from histogram.

    Applies paper's atom-corrected formula (Eq. 16, 19-23) to histogram.
    The 'atom' at q_alpha spans one histogram bin (approximate atom correction).

    Parameters
    ----------
    q_alpha : alpha-quantile (from quantile_from_histogram)
    hdata : dict from _build_histogram_*_top
    alpha : confidence level

    Returns
    -------
    dict with etl, mr_contrib, cr_contrib, M_gt, M_eq, eta
    """
    bin_edges = hdata["bin_edges"]
    bin_mids = hdata["bin_mids"]
    hist_w = hdata["hist_w"]
    hist_mr = hdata["hist_mr"]
    hist_cr = hdata["hist_cr"]
    tw = hdata["total_weight"]

    if tw == 0.0:
        return {"etl": np.nan, "mr_contrib": np.nan, "cr_contrib": np.nan,
                "M_gt": 0.0, "M_eq": 0.0, "eta": 0.0}

    # Identify bin containing q_alpha
    q_bin = int(np.searchsorted(bin_edges, q_alpha, side="right")) - 1
    q_bin = int(np.clip(q_bin, 0, len(hist_w) - 1))

    pw = hist_w / tw
    pw_mr = hist_mr / tw
    pw_cr = hist_cr / tw

    # M_gt = mass strictly above q_alpha (bins > q_bin)
    M_gt = float(pw[q_bin + 1:].sum())
    # M_eq = mass at the q_alpha bin (approximate atom)
    M_eq = float(pw[q_bin])

    eta = max((1.0 - alpha) - M_gt, 0.0)

    inv_1a = 1.0 / (1.0 - alpha)

    # ETL: sum of weighted losses strictly above q + atom contribution
    wl_gt = float(np.sum(pw[q_bin + 1:] * bin_mids[q_bin + 1:]))
    wl_eq_mean = float(bin_mids[q_bin])  # representative loss in atom bin
    etl = inv_1a * (wl_gt + eta * wl_eq_mean)

    # MR contribution
    wmr_gt = float(np.sum(pw_mr[q_bin + 1:]))
    mr_atom_mean = float(pw_mr[q_bin] / pw[q_bin]) if pw[q_bin] > 0 else 0.0
    etl_mr = inv_1a * (wmr_gt + eta * mr_atom_mean)

    # CR contribution
    wcr_gt = float(np.sum(pw_cr[q_bin + 1:]))
    cr_atom_mean = float(pw_cr[q_bin] / pw[q_bin]) if pw[q_bin] > 0 else 0.0
    etl_cr = inv_1a * (wcr_gt + eta * cr_atom_mean)

    return {
        "etl": etl,
        "mr_contrib": etl_mr,
        "cr_contrib": etl_cr,
        "M_gt": M_gt,
        "M_eq": M_eq,
        "eta": eta,
    }


def build_hierarchy_gaussian(
    mr1_samples: np.ndarray,
    cr1_samples: np.ndarray,
    mr1_z: np.ndarray,
    cr1_z: np.ndarray,
    mr2_samples: np.ndarray,
    cr2_samples: np.ndarray,
    mr2_z: np.ndarray,
    cr2_z: np.ndarray,
    gamma_1: float,
    gamma_2: float,
    rho_top: float,
    n_bins: int = 2000,
    chunk_size: int = 200,
) -> dict:
    """Build full Gaussian hierarchy: within-portfolio CDFs + top-level histogram.

    One-stop function used by all experiments.

    Returns
    -------
    dict with:
      losses1, w1, lw1, z1_top : portfolio-1 within-distribution and Z-transforms
      losses2, w2, lw2, z2_top
      mr1, cr1, mr2, cr2 : component arrays aligned with within-portfolio scenarios
      hdata_gauss : histogram dict from _build_histogram_gaussian_top
      wsum_dev1, wsum_dev2 : pre-normalization deviation diagnostics
    """
    from src.copulas import compute_within_portfolio_gaussian

    losses1, w1, lw1 = compute_within_portfolio_gaussian(
        mr1_samples, cr1_samples, mr1_z, cr1_z, gamma=gamma_1)
    losses2, w2, lw2 = compute_within_portfolio_gaussian(
        mr2_samples, cr2_samples, mr2_z, cr2_z, gamma=gamma_2)

    z1_top = compute_z_transforms(losses1, w1)
    z2_top = compute_z_transforms(losses2, w2)

    _, mr1, cr1 = build_scenario_components(mr1_samples, cr1_samples)
    _, mr2, cr2 = build_scenario_components(mr2_samples, cr2_samples)

    # Build histogram in one pass
    hdata = _build_histogram_gaussian_top(
        losses1, lw1, z1_top, mr1, cr1,
        losses2, lw2, z2_top, mr2, cr2,
        rho_top, n_bins=n_bins, chunk_size=chunk_size)

    # Pre-normalization diagnostic
    n = len(mr1_samples)
    tw1 = np.sum(np.exp(lw1))
    theo1 = n ** 2 * np.sqrt(1.0 - gamma_1 ** 2)
    tw2 = np.sum(np.exp(lw2))
    theo2 = n ** 2 * np.sqrt(1.0 - gamma_2 ** 2)
    wsum_dev1 = abs(tw1 - theo1) / theo1
    wsum_dev2 = abs(tw2 - theo2) / theo2

    return {
        "losses1": losses1, "w1": w1, "lw1": lw1, "z1_top": z1_top,
        "losses2": losses2, "w2": w2, "lw2": lw2, "z2_top": z2_top,
        "mr1": mr1, "cr1": cr1, "mr2": mr2, "cr2": cr2,
        "hdata": hdata,
        "wsum_dev1": float(wsum_dev1),
        "wsum_dev2": float(wsum_dev2),
    }


def build_hierarchy_clayton(
    mr1_samples: np.ndarray,
    cr1_samples: np.ndarray,
    mr1_u: np.ndarray,
    cr1_u: np.ndarray,
    mr2_samples: np.ndarray,
    cr2_samples: np.ndarray,
    mr2_u: np.ndarray,
    cr2_u: np.ndarray,
    zeta_1: float,
    zeta_2: float,
    theta: float,
    n_bins: int = 2000,
    chunk_size: int = 200,
) -> dict:
    """Build full mirrored Clayton hierarchy: within-portfolio CDFs + top-level histogram."""
    from src.copulas import compute_within_portfolio_mirrored_clayton

    losses1, w1, lw1 = compute_within_portfolio_mirrored_clayton(
        mr1_samples, cr1_samples, mr1_u, cr1_u, zeta=zeta_1)
    losses2, w2, lw2 = compute_within_portfolio_mirrored_clayton(
        mr2_samples, cr2_samples, mr2_u, cr2_u, zeta=zeta_2)

    # Midpoint CDF prevents (1-F)=0 singularity in Clayton kernel at l_max
    f1_cdf = compute_within_portfolio_cdf_midpoint(w1)
    f2_cdf = compute_within_portfolio_cdf_midpoint(w2)

    _, mr1, cr1 = build_scenario_components(mr1_samples, cr1_samples)
    _, mr2, cr2 = build_scenario_components(mr2_samples, cr2_samples)

    hdata = _build_histogram_clayton_top(
        losses1, lw1, f1_cdf, mr1, cr1,
        losses2, lw2, f2_cdf, mr2, cr2,
        theta, n_bins=n_bins, chunk_size=chunk_size)

    return {
        "losses1": losses1, "w1": w1, "lw1": lw1, "f1_cdf": f1_cdf,
        "losses2": losses2, "w2": w2, "lw2": lw2, "f2_cdf": f2_cdf,
        "mr1": mr1, "cr1": cr1, "mr2": mr2, "cr2": cr2,
        "hdata": hdata,
    }


# ---------------------------------------------------------------------------
# Helper: build within-portfolio scenario component arrays
# ---------------------------------------------------------------------------

def build_scenario_components(
    mr_samples: np.ndarray,
    cr_samples: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Build total loss, MR component, and CR component for all n^2 within-portfolio pairs.

    Returns sorted (by total loss) arrays aligned with within-portfolio distribution arrays.

    Parameters
    ----------
    mr_samples : market-risk samples, sorted ascending, shape (n,)
    cr_samples : credit-risk samples, sorted ascending, shape (n,)

    Returns
    -------
    loss_total : l_{ab} = mr[a] + cr[b], sorted ascending, shape (n^2,)
    loss_mr : mr[a] for each pair, shape (n^2,)
    loss_cr : cr[b] for each pair, shape (n^2,)
    """
    loss_matrix = mr_samples[:, None] + cr_samples[None, :]  # (n, n)
    mr_matrix = mr_samples[:, None] * np.ones((1, len(cr_samples)))  # (n, n)
    cr_matrix = np.ones((len(mr_samples), 1)) * cr_samples[None, :]  # (n, n)

    losses = loss_matrix.ravel()
    mrs = mr_matrix.ravel()
    crs = cr_matrix.ravel()

    sort_idx = np.argsort(losses, kind="stable")
    return losses[sort_idx], mrs[sort_idx], crs[sort_idx]
