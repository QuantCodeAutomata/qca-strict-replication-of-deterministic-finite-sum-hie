"""
Unit tests for src/aggregation.py

Tests cover:
  - Z-transform computation correctness
  - Gaussian top-level CDF boundary and monotonicity (small n)
  - Clayton top-level CDF boundary and monotonicity (small n)
  - Normalization consistency across CDF evaluations
  - ETL > quantile property
  - Scenario component decomposition identity
"""
import pytest
import numpy as np
from scipy.stats import norm

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from src.aggregation import (
    compute_z_transforms,
    compute_cdf_gaussian_top,
    compute_cdf_clayton_top,
    compute_quantile_gaussian_top,
    compute_quantile_clayton_top,
    compute_etl_gaussian_top,
    compute_etl_clayton_top,
    build_scenario_components,
)
from src.copulas import (
    compute_within_portfolio_gaussian,
    compute_within_portfolio_mirrored_clayton,
)
from src.marginals import generate_all_leaves

# Use small n for speed
N_TEST = 15
GAMMA_1, GAMMA_2, RHO_TOP = 0.75, 0.5, 0.5
ZETA_1, ZETA_2, THETA = 2.3468, 1.0, 1.0
ALPHA = 0.99
CHUNK = 50  # small chunk for tests


@pytest.fixture(scope='module')
def leaves():
    return generate_all_leaves(n=N_TEST, seed=2024)


@pytest.fixture(scope='module')
def gaussian_hier(leaves):
    losses1, w1, lw1 = compute_within_portfolio_gaussian(
        leaves['mr1_samples'], leaves['cr1_samples'],
        leaves['mr1_z'], leaves['cr1_z'], gamma=GAMMA_1)
    losses2, w2, lw2 = compute_within_portfolio_gaussian(
        leaves['mr2_samples'], leaves['cr2_samples'],
        leaves['mr2_z'], leaves['cr2_z'], gamma=GAMMA_2)
    z1 = compute_z_transforms(losses1, w1)
    z2 = compute_z_transforms(losses2, w2)
    _, mr1, cr1 = build_scenario_components(leaves['mr1_samples'], leaves['cr1_samples'])
    _, mr2, cr2 = build_scenario_components(leaves['mr2_samples'], leaves['cr2_samples'])
    return dict(losses1=losses1, lw1=lw1, z1=z1, mr1=mr1, cr1=cr1,
                losses2=losses2, lw2=lw2, z2=z2, mr2=mr2, cr2=cr2)


@pytest.fixture(scope='module')
def clayton_hier(leaves):
    losses1, w1, lw1 = compute_within_portfolio_mirrored_clayton(
        leaves['mr1_samples'], leaves['cr1_samples'],
        leaves['mr1_u'], leaves['cr1_u'], zeta=ZETA_1)
    losses2, w2, lw2 = compute_within_portfolio_mirrored_clayton(
        leaves['mr2_samples'], leaves['cr2_samples'],
        leaves['mr2_u'], leaves['cr2_u'], zeta=ZETA_2)
    f1 = np.cumsum(w1)
    f2 = np.cumsum(w2)
    _, mr1, cr1 = build_scenario_components(leaves['mr1_samples'], leaves['cr1_samples'])
    _, mr2, cr2 = build_scenario_components(leaves['mr2_samples'], leaves['cr2_samples'])
    return dict(losses1=losses1, lw1=lw1, f1=f1, mr1=mr1, cr1=cr1,
                losses2=losses2, lw2=lw2, f2=f2, mr2=mr2, cr2=cr2)


# ── Z-transform tests ─────────────────────────────────────────────────────────

def test_z_transforms_shape(gaussian_hier):
    assert gaussian_hier['z1'].shape == (N_TEST ** 2,)


def test_z_transforms_finite(gaussian_hier):
    assert np.all(np.isfinite(gaussian_hier['z1']))


def test_z_transforms_nondecreasing(gaussian_hier):
    """Cumulative weights increase => Z-transforms should be non-decreasing."""
    assert np.all(np.diff(gaussian_hier['z1']) >= -1e-10)


def test_z_transforms_clipping():
    """Extreme weights at boundaries should be clipped to [-37, 37]."""
    # Artificially create almost-all-mass-at-one-point weights
    n = 5
    losses = np.linspace(1, 5, n)
    weights = np.zeros(n)
    weights[-1] = 1.0  # all mass at last point
    z = compute_z_transforms(losses, weights)
    assert np.all(np.isfinite(z))
    assert np.all(z <= 37.0)
    assert np.all(z >= -37.0)


# ── Gaussian top-level CDF tests ──────────────────────────────────────────────

def test_gaussian_cdf_below_min_is_zero(gaussian_hier):
    g = gaussian_hier
    l_min = g['losses1'][0] + g['losses2'][0]
    f, nc = compute_cdf_gaussian_top(
        l_min - 10.0, g['losses1'], g['lw1'], g['z1'],
        g['losses2'], g['lw2'], g['z2'], RHO_TOP, CHUNK)
    assert f == pytest.approx(0.0, abs=1e-8)


def test_gaussian_cdf_above_max_is_one(gaussian_hier):
    g = gaussian_hier
    l_max = g['losses1'][-1] + g['losses2'][-1]
    f, nc = compute_cdf_gaussian_top(
        l_max + 100.0, g['losses1'], g['lw1'], g['z1'],
        g['losses2'], g['lw2'], g['z2'], RHO_TOP, CHUNK)
    assert f == pytest.approx(1.0, abs=1e-8)


def test_gaussian_cdf_monotone(gaussian_hier):
    g = gaussian_hier
    l_min = g['losses1'][0] + g['losses2'][0]
    l_max = g['losses1'][-1] + g['losses2'][-1]
    grid = np.linspace(l_min, l_max, 10)
    cdf_vals = []
    for l_val in grid:
        f, _ = compute_cdf_gaussian_top(
            l_val, g['losses1'], g['lw1'], g['z1'],
            g['losses2'], g['lw2'], g['z2'], RHO_TOP, CHUNK)
        cdf_vals.append(f)
    diffs = np.diff(cdf_vals)
    assert np.all(diffs >= -1e-6), f"CDF not monotone: {diffs}"


def test_gaussian_cdf_in_unit_interval(gaussian_hier):
    g = gaussian_hier
    l_mid = (g['losses1'][N_TEST // 2] + g['losses2'][N_TEST // 2])
    f, _ = compute_cdf_gaussian_top(
        l_mid, g['losses1'], g['lw1'], g['z1'],
        g['losses2'], g['lw2'], g['z2'], RHO_TOP, CHUNK)
    assert 0.0 <= f <= 1.0


# ── Clayton top-level CDF tests ───────────────────────────────────────────────

def test_clayton_cdf_below_min_is_zero(clayton_hier):
    c = clayton_hier
    l_min = c['losses1'][0] + c['losses2'][0]
    f, _ = compute_cdf_clayton_top(
        l_min - 10.0, c['losses1'], c['lw1'], c['f1'],
        c['losses2'], c['lw2'], c['f2'], THETA, CHUNK)
    assert f == pytest.approx(0.0, abs=1e-8)


def test_clayton_cdf_above_max_is_one(clayton_hier):
    c = clayton_hier
    l_max = c['losses1'][-1] + c['losses2'][-1]
    f, _ = compute_cdf_clayton_top(
        l_max + 100.0, c['losses1'], c['lw1'], c['f1'],
        c['losses2'], c['lw2'], c['f2'], THETA, CHUNK)
    assert f == pytest.approx(1.0, abs=1e-8)


def test_clayton_cdf_monotone(clayton_hier):
    c = clayton_hier
    l_min = c['losses1'][0] + c['losses2'][0]
    l_max = c['losses1'][-1] + c['losses2'][-1]
    grid = np.linspace(l_min, l_max, 10)
    cdf_vals = [
        compute_cdf_clayton_top(
            l, c['losses1'], c['lw1'], c['f1'],
            c['losses2'], c['lw2'], c['f2'], THETA, CHUNK)[0]
        for l in grid
    ]
    assert np.all(np.diff(cdf_vals) >= -1e-6)


# ── Quantile and ETL tests ────────────────────────────────────────────────────

def test_gaussian_quantile_in_support(gaussian_hier):
    g = gaussian_hier
    q = compute_quantile_gaussian_top(
        g['losses1'], g['lw1'], g['z1'],
        g['losses2'], g['lw2'], g['z2'],
        RHO_TOP, alpha=0.99, chunk_size=CHUNK, n_candidates=50)
    l_min = g['losses1'][0] + g['losses2'][0]
    l_max = g['losses1'][-1] + g['losses2'][-1]
    assert l_min <= q <= l_max


def test_gaussian_etl_geq_quantile(gaussian_hier):
    g = gaussian_hier
    q = compute_quantile_gaussian_top(
        g['losses1'], g['lw1'], g['z1'],
        g['losses2'], g['lw2'], g['z2'],
        RHO_TOP, alpha=0.99, chunk_size=CHUNK, n_candidates=50)
    res = compute_etl_gaussian_top(
        q, g['losses1'], g['lw1'], g['z1'], g['mr1'], g['cr1'],
        g['losses2'], g['lw2'], g['z2'], g['mr2'], g['cr2'],
        RHO_TOP, alpha=0.99, chunk_size=CHUNK)
    assert res['etl'] >= q - 1e-6, f"ETL {res['etl']} < quantile {q}"


def test_gaussian_etl_decomp_identity(gaussian_hier):
    """MR + CR contributions should equal total ETL."""
    g = gaussian_hier
    q = compute_quantile_gaussian_top(
        g['losses1'], g['lw1'], g['z1'],
        g['losses2'], g['lw2'], g['z2'],
        RHO_TOP, alpha=0.99, chunk_size=CHUNK, n_candidates=50)
    res = compute_etl_gaussian_top(
        q, g['losses1'], g['lw1'], g['z1'], g['mr1'], g['cr1'],
        g['losses2'], g['lw2'], g['z2'], g['mr2'], g['cr2'],
        RHO_TOP, alpha=0.99, chunk_size=CHUNK)
    decomp_err = abs(res['mr_contrib'] + res['cr_contrib'] - res['etl'])
    assert decomp_err < 1e-6, f"Decomposition error: {decomp_err:.2e}"


def test_clayton_etl_decomp_identity(clayton_hier):
    """MR + CR contributions should equal total ETL for Clayton."""
    c = clayton_hier
    q = compute_quantile_clayton_top(
        c['losses1'], c['lw1'], c['f1'],
        c['losses2'], c['lw2'], c['f2'],
        THETA, alpha=0.99, chunk_size=CHUNK, n_candidates=50)
    res = compute_etl_clayton_top(
        q, c['losses1'], c['lw1'], c['f1'], c['mr1'], c['cr1'],
        c['losses2'], c['lw2'], c['f2'], c['mr2'], c['cr2'],
        THETA, alpha=0.99, chunk_size=CHUNK)
    decomp_err = abs(res['mr_contrib'] + res['cr_contrib'] - res['etl'])
    assert decomp_err < 1e-6


# ── Scenario component tests ──────────────────────────────────────────────────

def test_build_scenario_components_shape(leaves):
    l, mr, cr = build_scenario_components(leaves['mr1_samples'], leaves['cr1_samples'])
    assert l.shape == (N_TEST ** 2,)
    assert mr.shape == (N_TEST ** 2,)
    assert cr.shape == (N_TEST ** 2,)


def test_build_scenario_components_sum(leaves):
    l, mr, cr = build_scenario_components(leaves['mr1_samples'], leaves['cr1_samples'])
    np.testing.assert_allclose(l, mr + cr, rtol=1e-10)


def test_build_scenario_components_sorted(leaves):
    l, _, _ = build_scenario_components(leaves['mr1_samples'], leaves['cr1_samples'])
    assert np.all(np.diff(l) >= 0)
