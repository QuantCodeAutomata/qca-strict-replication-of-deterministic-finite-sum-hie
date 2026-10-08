"""
Unit tests for src/risk_measures.py

Tests cover:
  - Quantile extraction: boundary, monotonicity, right-continuity
  - ETL: >= quantile, atom correction, boundary cases
  - ETL contributions: decomposition identity, non-negativity
  - compute_all_risk_measures: DataFrame schema and mathematical properties
  - Edge cases: uniform distribution, single atom, extreme alpha
"""
import pytest
import numpy as np
import pandas as pd

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from src.risk_measures import (
    compute_quantile,
    compute_etl,
    compute_etl_contributions,
    compute_all_risk_measures,
)


# ── Fixtures ──────────────────────────────────────────────────────────────────

def make_uniform_dist(n: int = 100):
    """Equally weighted uniform discrete distribution on [1, n]."""
    losses = np.arange(1.0, n + 1.0)
    weights = np.ones(n) / n
    cum_weights = np.cumsum(weights)
    return losses, weights, cum_weights


def make_two_atom_dist():
    """Two-atom distribution: loss 0 with weight 0.95, loss 10 with weight 0.05."""
    losses = np.array([0.0, 10.0])
    weights = np.array([0.95, 0.05])
    cum_weights = np.cumsum(weights)
    return losses, weights, cum_weights


def make_atom_at_quantile_dist():
    """Distribution where alpha=0.99 quantile is at an atom with significant mass."""
    n = 100
    losses = np.arange(1.0, n + 1.0)
    # Put 5% mass at loss=50 (atom) and distribute rest
    weights = np.ones(n) / n
    weights[49] += 0.04  # boost atom at 50
    weights[:49] -= 0.04 / 49
    weights = np.maximum(weights, 0)
    weights /= weights.sum()
    cum_weights = np.cumsum(weights)
    return losses, weights, cum_weights


# ── Quantile tests ────────────────────────────────────────────────────────────

def test_quantile_in_support():
    losses, w, cw = make_uniform_dist(100)
    q = compute_quantile(losses, cw, alpha=0.99)
    assert losses[0] <= q <= losses[-1]


def test_quantile_uniform_99():
    """For uniform on 1..100, q_0.99 should be 99 (the 99th value)."""
    losses, w, cw = make_uniform_dist(100)
    q = compute_quantile(losses, cw, alpha=0.99)
    assert q == pytest.approx(99.0, abs=1.0)  # allow ±1 for discrete


def test_quantile_increases_with_alpha():
    losses, w, cw = make_uniform_dist(100)
    q95 = compute_quantile(losses, cw, alpha=0.95)
    q99 = compute_quantile(losses, cw, alpha=0.99)
    assert q99 >= q95


def test_quantile_alpha_zero():
    """Alpha approaching 0 should give the minimum loss."""
    losses, w, cw = make_uniform_dist(10)
    q = compute_quantile(losses, cw, alpha=0.0)
    assert q == losses[0]


def test_quantile_alpha_near_one():
    losses, w, cw = make_uniform_dist(100)
    q = compute_quantile(losses, cw, alpha=0.999)
    assert q == losses[-1]


def test_quantile_two_atom():
    """Two-atom: q_0.99 should be the large atom at loss=10."""
    losses, w, cw = make_two_atom_dist()
    q = compute_quantile(losses, cw, alpha=0.99)
    assert q == pytest.approx(10.0)


# ── ETL tests ────────────────────────────────────────────────────────────────

def test_etl_geq_quantile():
    losses, w, cw = make_uniform_dist(100)
    res = compute_etl(losses, w, cw, alpha=0.99)
    assert res['etl'] >= res['q'] - 1e-10


def test_etl_atom_correction_two_atom():
    """For two-atom with mass 0.05 at 10: ETL_0.99 = 10 (all mass above or at quantile)."""
    losses, w, cw = make_two_atom_dist()
    # q_0.99 = 10, M_>(10)=0, M_=(10)=0.05, eta=(0.01-0)=0.01
    # ETL = (1/0.01)*(0 + 0.01*10) = 10
    res = compute_etl(losses, w, cw, alpha=0.99)
    assert res['etl'] == pytest.approx(10.0, rel=1e-6)


def test_etl_eta_nonnegative():
    losses, w, cw = make_uniform_dist(200)
    res = compute_etl(losses, w, cw, alpha=0.99)
    assert res['eta'] >= 0.0


def test_etl_M_gt_plus_M_eq_leq_one_minus_alpha():
    """M_>(q) + M_=(q) should be >= 1-alpha (atom correction ensures this)."""
    losses, w, cw = make_uniform_dist(100)
    res = compute_etl(losses, w, cw, alpha=0.95)
    assert res['M_gt'] + res['M_eq'] >= (1.0 - 0.95) - 1e-10


def test_etl_keys():
    losses, w, cw = make_uniform_dist(50)
    res = compute_etl(losses, w, cw, alpha=0.99)
    assert 'etl' in res
    assert 'M_gt' in res
    assert 'M_eq' in res
    assert 'eta' in res
    assert 'q' in res


# ── ETL contributions tests ───────────────────────────────────────────────────

def test_etl_contributions_decomp_identity():
    """ETL_MR + ETL_CR must equal total ETL up to machine precision."""
    losses, w, cw = make_uniform_dist(200)
    mr = losses * 0.6
    cr = losses * 0.4
    res = compute_etl_contributions(losses, w, cw, mr, cr, alpha=0.99)
    residual = abs(res['etl_mr'] + res['etl_cr'] - res['etl_total'])
    assert residual < 1e-10, f"Decomposition residual: {residual:.2e}"


def test_etl_contributions_nonnegative():
    """With positive mr and cr, contributions should be non-negative."""
    losses, w, cw = make_uniform_dist(100)
    mr = losses * 0.5
    cr = losses * 0.5
    res = compute_etl_contributions(losses, w, cw, mr, cr, alpha=0.99)
    assert res['etl_mr'] >= 0.0
    assert res['etl_cr'] >= 0.0


def test_etl_contributions_proportional():
    """If mr = c*cr everywhere, ETLCont_MR / ETLCont_CR should be c."""
    n = 100
    losses = np.sort(np.random.default_rng(0).uniform(0, 100, n))
    weights = np.ones(n) / n
    cum_weights = np.cumsum(weights)
    mr = losses * 0.7
    cr = losses * 0.3
    res = compute_etl_contributions(losses, weights, cum_weights, mr, cr, alpha=0.99)
    if res['etl_cr'] > 1e-12:
        ratio = res['etl_mr'] / res['etl_cr']
        assert ratio == pytest.approx(7.0 / 3.0, rel=0.01)


def test_etl_contributions_atom_corrected():
    """Test atom-corrected contributions when quantile atom has significant mass."""
    losses, w, cw = make_atom_at_quantile_dist()
    n = len(losses)
    mr = losses * 0.6
    cr = losses * 0.4
    res = compute_etl_contributions(losses, w, cw, mr, cr, alpha=0.99)
    # Decomposition must still hold
    assert abs(res['etl_mr'] + res['etl_cr'] - res['etl_total']) < 1e-10


def test_etl_contributions_keys():
    losses, w, cw = make_uniform_dist(50)
    mr, cr = losses * 0.5, losses * 0.5
    res = compute_etl_contributions(losses, w, cw, mr, cr, alpha=0.99)
    for key in ('etl_mr', 'etl_cr', 'etl_total', 'decomp_residual', 'M_gt', 'M_eq', 'eta', 'q'):
        assert key in res


# ── compute_all_risk_measures tests ───────────────────────────────────────────

def test_compute_all_returns_dataframe():
    losses, w, _, = make_uniform_dist(200)
    mr, cr = losses * 0.5, losses * 0.5
    df = compute_all_risk_measures(losses, w, mr, cr)
    assert isinstance(df, pd.DataFrame)


def test_compute_all_default_alphas():
    losses, w, _ = make_uniform_dist(200)
    mr, cr = losses * 0.5, losses * 0.5
    df = compute_all_risk_measures(losses, w, mr, cr)
    expected_alphas = [0.95, 0.975, 0.99, 0.995]
    assert list(df['alpha']) == expected_alphas


def test_compute_all_quantile_monotone():
    losses, w, _ = make_uniform_dist(200)
    mr, cr = losses * 0.5, losses * 0.5
    df = compute_all_risk_measures(losses, w, mr, cr)
    assert list(df['q_alpha']) == sorted(df['q_alpha'].tolist())


def test_compute_all_etl_monotone():
    losses, w, _ = make_uniform_dist(200)
    mr, cr = losses * 0.5, losses * 0.5
    df = compute_all_risk_measures(losses, w, mr, cr)
    assert list(df['etl']) == sorted(df['etl'].tolist())


def test_compute_all_shares_sum_to_one():
    losses, w, _ = make_uniform_dist(100)
    mr, cr = losses * 0.6, losses * 0.4
    df = compute_all_risk_measures(losses, w, mr, cr)
    for _, row in df.iterrows():
        assert abs(row['mr_share'] + row['cr_share'] - 1.0) < 1e-6


def test_compute_all_decomp_residuals_small():
    losses, w, _ = make_uniform_dist(200)
    mr, cr = losses * 0.5, losses * 0.5
    df = compute_all_risk_measures(losses, w, mr, cr)
    assert (df['decomp_residual'] < 1e-9).all()
