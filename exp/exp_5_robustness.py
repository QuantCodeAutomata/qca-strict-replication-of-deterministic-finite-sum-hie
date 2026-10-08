#!/usr/bin/env python3
"""
Exp 5: Robustness studies using the histogram-based API.

Three substudies:
  A. Confidence-level sensitivity: alpha in {0.90, 0.95, 0.975, 0.99, 0.995}
     (N=200, seed=42, both Gaussian and Clayton)
  B. Sample-size sensitivity: n in {50, 100, 200}
     (alpha=0.99, n_bins scaled with n, both families)
  C. Depth sensitivity: depth m=2 (four leaves) vs m=3 (eight leaves, N=30)
     Depth-3 uses approximate two-stage aggregation with build_hierarchy_*
     applied twice – each meta-portfolio is compressed to N_COMPRESSED
     samples drawn from its level-2 histogram.
     APPROXIMATION: documented as 'approximate depth-3 aggregation'.
"""
import sys
import os
import time

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.stats import norm

from src.marginals import generate_all_leaves, generate_market_leaf, generate_credit_leaf
from src.aggregation import (
    build_hierarchy_gaussian, build_hierarchy_clayton,
    quantile_from_histogram, etl_from_histogram,
)

SEED = 42
GAMMA_1, GAMMA_2, RHO_TOP = 0.75, 0.50, 0.50
ZETA_1, ZETA_2, THETA = 2.3468, 1.0, 1.0
ALPHA = 0.99

RESULTS_DIR = os.path.join(os.path.dirname(__file__), '..', 'results')
os.makedirs('results', exist_ok=True)


# ── Helpers ──────────────────────────────────────────────────────────────────

def _build_both(leaves, n_bins=2000, chunk=200):
    """Build Gaussian and Clayton hierarchies; return (hdata_g, hdata_c)."""
    g = build_hierarchy_gaussian(
        leaves['mr1_samples'], leaves['cr1_samples'],
        leaves['mr1_z'], leaves['cr1_z'],
        leaves['mr2_samples'], leaves['cr2_samples'],
        leaves['mr2_z'], leaves['cr2_z'],
        GAMMA_1, GAMMA_2, RHO_TOP,
        n_bins=n_bins, chunk_size=chunk,
    )
    c = build_hierarchy_clayton(
        leaves['mr1_samples'], leaves['cr1_samples'],
        leaves['mr1_u'], leaves['cr1_u'],
        leaves['mr2_samples'], leaves['cr2_samples'],
        leaves['mr2_u'], leaves['cr2_u'],
        ZETA_1, ZETA_2, THETA,
        n_bins=n_bins, chunk_size=chunk,
    )
    return g['hdata'], c['hdata']


def _risk_row(hdata_g, hdata_c, alpha, n, label=''):
    """Extract q, ETL, MR share for both families at given alpha."""
    q_g = quantile_from_histogram(hdata_g, alpha)
    e_g = etl_from_histogram(q_g, hdata_g, alpha)
    q_c = quantile_from_histogram(hdata_c, alpha)
    e_c = etl_from_histogram(q_c, hdata_c, alpha)
    return dict(
        label=label, alpha=alpha, n=n,
        q_gauss=q_g, etl_gauss=e_g['etl'],
        mr_share_g=e_g['mr_contrib'] / e_g['etl'] if e_g['etl'] > 0 else np.nan,
        cr_share_g=e_g['cr_contrib'] / e_g['etl'] if e_g['etl'] > 0 else np.nan,
        q_clay=q_c, etl_clay=e_c['etl'],
        mr_share_c=e_c['mr_contrib'] / e_c['etl'] if e_c['etl'] > 0 else np.nan,
        cr_share_c=e_c['cr_contrib'] / e_c['etl'] if e_c['etl'] > 0 else np.nan,
    )


def _sample_from_histogram(hdata, n_c, rng):
    """Draw n_c sorted samples from a histogram via inverse-CDF sampling."""
    hist_w = hdata['hist_w']
    bin_mids = hdata['bin_mids']
    tw = hdata['total_weight']
    probs = hist_w / tw
    probs = probs / probs.sum()
    idx = rng.choice(len(bin_mids), size=n_c, p=probs, replace=True)
    return np.sort(bin_mids[idx])


# ── Substudy A ────────────────────────────────────────────────────────────────

def substudy_a():
    """Confidence-level sensitivity: alpha in {0.90, 0.95, 0.975, 0.99, 0.995}."""
    print("\n" + "=" * 65)
    print("Substudy A: Confidence Level Sensitivity (N=200, seed=42)")
    ALPHAS = [0.90, 0.95, 0.975, 0.99, 0.995]
    N_A = 200

    t0 = time.time()
    leaves = generate_all_leaves(n=N_A, seed=SEED)
    hdata_g, hdata_c = _build_both(leaves, n_bins=2000, chunk=200)
    print(f"  Hierarchies built [{time.time()-t0:.1f}s]")

    rows = []
    print(f"\n  {'alpha':>7} {'q_G':>8} {'ETL_G':>8} {'MR%_G':>7} {'CR%_G':>7}"
          f" | {'q_C':>8} {'ETL_C':>8} {'MR%_C':>7} {'CR%_C':>7}")
    print("  " + "-" * 73)
    for alpha in ALPHAS:
        r = _risk_row(hdata_g, hdata_c, alpha, N_A, label=f'alpha={alpha}')
        rows.append(r)
        print(f"  {alpha:7.3f} {r['q_gauss']:8.4f} {r['etl_gauss']:8.4f}"
              f" {r['mr_share_g']:7.2%} {r['cr_share_g']:7.2%}"
              f" | {r['q_clay']:8.4f} {r['etl_clay']:8.4f}"
              f" {r['mr_share_c']:7.2%} {r['cr_share_c']:7.2%}")
    return rows, hdata_g, hdata_c


# ── Substudy B ────────────────────────────────────────────────────────────────

def substudy_b():
    """Sample-size sensitivity: n in {50, 100, 200}, alpha=0.99."""
    print("\n" + "=" * 65)
    print("Substudy B: Sample Size Sensitivity (alpha=0.99)")
    NS      = [50, 100, 200]
    N_BINS_MAP = {50: 500, 100: 1000, 200: 2000}

    rows = []
    print(f"\n  {'n':>5} {'n_bins':>7} {'q_G':>8} {'ETL_G':>8} {'q_C':>8} {'ETL_C':>8}"
          f" {'runtime':>9}")
    print("  " + "-" * 58)
    for n_val in NS:
        n_bins = N_BINS_MAP[n_val]
        t0 = time.time()
        leaves = generate_all_leaves(n=n_val, seed=SEED)
        hdata_g, hdata_c = _build_both(leaves, n_bins=n_bins, chunk=min(200, n_val))
        r = _risk_row(hdata_g, hdata_c, ALPHA, n_val, label=f'n={n_val}')
        r['n_bins'] = n_bins
        r['runtime'] = time.time() - t0
        rows.append(r)
        print(f"  {n_val:5d} {n_bins:7d} {r['q_gauss']:8.4f} {r['etl_gauss']:8.4f}"
              f" {r['q_clay']:8.4f} {r['etl_clay']:8.4f} {r['runtime']:9.1f}s")
    return rows


# ── Substudy C ────────────────────────────────────────────────────────────────

def substudy_c(hdata_g_200, hdata_c_200):
    """Depth sensitivity: m=2 (standard) vs m=3 (approximate, N=30).

    Depth m=2: four-leaf two-portfolio hierarchy (N=200, standard result).
    Depth m=3: eight-leaf hierarchy implemented as approximate two-stage
               aggregation using build_hierarchy_* twice:
      Stage 1: Build two meta-portfolio histograms (each from n^4 = 810,000
               scenarios at N=30), for the (P1,P2) and (P3,P4) leaf pairs.
      Stage 2: Draw N_COMPRESSED samples from each meta-histogram to form
               'compressed' leaf distributions, then call build_hierarchy_*
               once more to produce the final depth-3 aggregate distribution.
    NOTE: The compressed samples treat each meta-portfolio as a single-asset
    leaf (CR component set to zero, within-level gamma≈0/zeta≈0 for near-
    independence), making this an APPROXIMATION documented as such.
    """
    print("\n" + "=" * 65)
    print("Substudy C: Hierarchical Depth Sensitivity (b=2, m in {2,3})")

    rows = []

    # Depth 2: read from standard N=200 histograms
    q_g2  = quantile_from_histogram(hdata_g_200, ALPHA)
    etl_g2 = etl_from_histogram(q_g2, hdata_g_200, ALPHA)['etl']
    q_c2  = quantile_from_histogram(hdata_c_200, ALPHA)
    etl_c2 = etl_from_histogram(q_c2, hdata_c_200, ALPHA)['etl']
    print(f"\n  Depth 2 (N=200, four leaves, deterministic histogram):")
    print(f"    Gaussian:  q_0.99={q_g2:.4f}  ETL_0.99={etl_g2:.4f}")
    print(f"    Clayton:   q_0.99={q_c2:.4f}  ETL_0.99={etl_c2:.4f}")
    rows.append(dict(family='Gaussian', depth=2, N=200,
                     q99=q_g2, etl99=etl_g2, approx=False))
    rows.append(dict(family='Clayton',  depth=2, N=200,
                     q99=q_c2, etl99=etl_c2, approx=False))

    # Depth 3: eight leaves, N=30
    N_D3 = 30
    N_COMPRESSED = 50   # compressed samples drawn from each meta-histogram
    GAMMA_SMALL  = 1e-4  # ~independence within-level for final aggregation
    ZETA_SMALL   = 0.01  # ~independence within-level for final aggregation
    N_BINS_D3    = 500

    print(f"\n  Depth 3 (APPROXIMATE, N={N_D3}, eight leaves, two-stage aggregation):")
    print(f"    N_COMPRESSED={N_COMPRESSED}, gamma_final={GAMMA_SMALL}, zeta_final={ZETA_SMALL}")

    rng_d3 = np.random.default_rng(SEED + 100)

    # Generate 8 leaves: 4 MR + 4 CR
    mr1, mr1_u, mr1_z = generate_market_leaf(N_D3, 0.0, 0.25, rng_d3)
    cr1, cr1_u, cr1_z = generate_credit_leaf(N_D3, 1.0, 1.0, 2.5, rng_d3)
    mr2, mr2_u, mr2_z = generate_market_leaf(N_D3, 0.0, 0.25, rng_d3)
    cr2, cr2_u, cr2_z = generate_credit_leaf(N_D3, 1.0, 1.0, 2.5, rng_d3)
    mr3, mr3_u, mr3_z = generate_market_leaf(N_D3, 0.0, 0.25, rng_d3)
    cr3, cr3_u, cr3_z = generate_credit_leaf(N_D3, 1.0, 1.0, 2.5, rng_d3)
    mr4, mr4_u, mr4_z = generate_market_leaf(N_D3, 0.0, 0.25, rng_d3)
    cr4, cr4_u, cr4_z = generate_credit_leaf(N_D3, 1.0, 1.0, 2.5, rng_d3)

    # Stage 1 – Gaussian: build meta-portfolio histograms
    print(f"    [Gaussian] Building meta1 and meta2 histograms (n^4={N_D3**4} scenarios)...")
    t0 = time.time()
    meta1_g = build_hierarchy_gaussian(
        mr1, cr1, mr1_z, cr1_z, mr2, cr2, mr2_z, cr2_z,
        GAMMA_1, GAMMA_2, RHO_TOP, n_bins=N_BINS_D3, chunk_size=N_D3,
    )
    meta2_g = build_hierarchy_gaussian(
        mr3, cr3, mr3_z, cr3_z, mr4, cr4, mr4_z, cr4_z,
        GAMMA_1, GAMMA_2, RHO_TOP, n_bins=N_BINS_D3, chunk_size=N_D3,
    )
    print(f"    Stage 1 done [{time.time()-t0:.1f}s]")

    # Stage 2 – Gaussian: sample compressed distributions and aggregate
    rng_c = np.random.default_rng(SEED + 200)
    meta1_comp = _sample_from_histogram(meta1_g['hdata'], N_COMPRESSED, rng_c)
    meta2_comp = _sample_from_histogram(meta2_g['hdata'], N_COMPRESSED, rng_c)
    zeros_n    = np.zeros(N_COMPRESSED)
    z_comp     = norm.ppf((np.arange(1, N_COMPRESSED + 1) - 0.5) / N_COMPRESSED)

    print(f"    [Gaussian] Final aggregation of compressed meta-portfolios...")
    t0 = time.time()
    # Treat meta1_comp as MR leaf, zeros as CR leaf (gamma≈0 → near-uniform weights)
    final_g = build_hierarchy_gaussian(
        meta1_comp, zeros_n, z_comp, z_comp,
        meta2_comp, zeros_n, z_comp, z_comp,
        GAMMA_SMALL, GAMMA_SMALL, RHO_TOP,
        n_bins=N_BINS_D3, chunk_size=N_COMPRESSED,
    )
    hdata_d3_g = final_g['hdata']
    q_d3_g   = quantile_from_histogram(hdata_d3_g, ALPHA)
    etl_d3_g = etl_from_histogram(q_d3_g, hdata_d3_g, ALPHA)['etl']
    print(f"    Stage 2 done [{time.time()-t0:.1f}s]")
    print(f"    Gaussian depth-3 (approx): q_0.99={q_d3_g:.4f}  ETL_0.99={etl_d3_g:.4f}")
    rows.append(dict(family='Gaussian', depth=3, N=N_D3,
                     q99=q_d3_g, etl99=etl_d3_g, approx=True))

    # Stage 1 – Clayton: build meta-portfolio histograms
    print(f"\n    [Clayton] Building meta1 and meta2 histograms...")
    t0 = time.time()
    meta1_c = build_hierarchy_clayton(
        mr1, cr1, mr1_u, cr1_u, mr2, cr2, mr2_u, cr2_u,
        ZETA_1, ZETA_2, THETA, n_bins=N_BINS_D3, chunk_size=N_D3,
    )
    meta2_c = build_hierarchy_clayton(
        mr3, cr3, mr3_u, cr3_u, mr4, cr4, mr4_u, cr4_u,
        ZETA_1, ZETA_2, THETA, n_bins=N_BINS_D3, chunk_size=N_D3,
    )
    print(f"    Stage 1 done [{time.time()-t0:.1f}s]")

    # Stage 2 – Clayton: sample compressed distributions and aggregate
    rng_c2 = np.random.default_rng(SEED + 300)
    meta1_comp_c = _sample_from_histogram(meta1_c['hdata'], N_COMPRESSED, rng_c2)
    meta2_comp_c = _sample_from_histogram(meta2_c['hdata'], N_COMPRESSED, rng_c2)
    u_comp = (np.arange(1, N_COMPRESSED + 1) - 0.5) / N_COMPRESSED

    print(f"    [Clayton] Final aggregation of compressed meta-portfolios...")
    t0 = time.time()
    # Treat meta1_comp_c as MR leaf, zeros as CR leaf (zeta≈0 → near-uniform weights)
    final_c = build_hierarchy_clayton(
        meta1_comp_c, zeros_n, u_comp, u_comp,
        meta2_comp_c, zeros_n, u_comp, u_comp,
        ZETA_SMALL, ZETA_SMALL, THETA,
        n_bins=N_BINS_D3, chunk_size=N_COMPRESSED,
    )
    hdata_d3_c = final_c['hdata']
    q_d3_c   = quantile_from_histogram(hdata_d3_c, ALPHA)
    etl_d3_c = etl_from_histogram(q_d3_c, hdata_d3_c, ALPHA)['etl']
    print(f"    Stage 2 done [{time.time()-t0:.1f}s]")
    print(f"    Clayton depth-3 (approx): q_0.99={q_d3_c:.4f}  ETL_0.99={etl_d3_c:.4f}")
    rows.append(dict(family='Clayton', depth=3, N=N_D3,
                     q99=q_d3_c, etl99=etl_d3_c, approx=True))

    # Depth-3 scaling analysis
    print("\n  Depth-3 qualitative tail risk scaling:")
    for fam in ['Gaussian', 'Clayton']:
        r2 = next(r for r in rows if r['family'] == fam and r['depth'] == 2)
        r3 = next(r for r in rows if r['family'] == fam and r['depth'] == 3)
        etl_ratio = r3['etl99'] / r2['etl99'] if r2['etl99'] > 0 else np.nan
        q_ratio   = r3['q99']   / r2['q99']   if r2['q99']   > 0 else np.nan
        print(f"    {fam}: ETL ratio (d3/d2)={etl_ratio:.3f}  q ratio={q_ratio:.3f}")

    return rows


# ── Plots ─────────────────────────────────────────────────────────────────────

def make_plots(rows_a, rows_b, rows_c):
    """Save robustness study plots to results/exp5_*.png."""
    # Plot A: ETL and q vs alpha
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    ax = axes[0]
    alphas_a = [r['alpha'] for r in rows_a]
    ax.plot(alphas_a, [r['etl_gauss'] for r in rows_a], 'b-o', ms=7, lw=2, label='Gaussian ETL')
    ax.plot(alphas_a, [r['etl_clay']  for r in rows_a], 'r-s', ms=7, lw=2, label='Clayton ETL')
    ax.set_xlabel('Confidence level alpha', fontsize=12)
    ax.set_ylabel('ETL_alpha', fontsize=12)
    ax.set_title('Substudy A: ETL vs Confidence Level')
    ax.legend(); ax.grid(True, alpha=0.3)

    ax = axes[1]
    ax.plot(alphas_a, [r['q_gauss'] for r in rows_a], 'b-o', ms=7, lw=2, label='Gaussian q')
    ax.plot(alphas_a, [r['q_clay']  for r in rows_a], 'r-s', ms=7, lw=2, label='Clayton q')
    ax.set_xlabel('Confidence level alpha', fontsize=12)
    ax.set_ylabel('q_alpha', fontsize=12)
    ax.set_title('Substudy A: Quantile vs Confidence Level')
    ax.legend(); ax.grid(True, alpha=0.3)

    plt.suptitle('Exp 5 – Substudy A: Confidence Level Sensitivity', fontsize=13)
    plt.tight_layout()
    p = os.path.join(RESULTS_DIR, 'exp5_substudy_a.png')
    plt.savefig(p, dpi=120, bbox_inches='tight'); plt.close()
    print(f"Saved: {p}")

    # Plot B: ETL and q vs n
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    ns_b = [r['n'] for r in rows_b]
    ax = axes[0]
    ax.plot(ns_b, [r['etl_gauss'] for r in rows_b], 'b-o', ms=7, lw=2, label='Gaussian')
    ax.plot(ns_b, [r['etl_clay']  for r in rows_b], 'r-s', ms=7, lw=2, label='Clayton')
    ax.set_xlabel('Leaf sample size n', fontsize=12)
    ax.set_ylabel('ETL_0.99', fontsize=12)
    ax.set_title('Substudy B: ETL_0.99 vs Sample Size')
    ax.legend(); ax.grid(True, alpha=0.3)

    ax = axes[1]
    ax.plot(ns_b, [r['q_gauss'] for r in rows_b], 'b-o', ms=7, lw=2, label='Gaussian')
    ax.plot(ns_b, [r['q_clay']  for r in rows_b], 'r-s', ms=7, lw=2, label='Clayton')
    ax.set_xlabel('Leaf sample size n', fontsize=12)
    ax.set_ylabel('q_0.99', fontsize=12)
    ax.set_title('Substudy B: q_0.99 vs Sample Size')
    ax.legend(); ax.grid(True, alpha=0.3)

    plt.suptitle('Exp 5 – Substudy B: Sample Size Sensitivity', fontsize=13)
    plt.tight_layout()
    p = os.path.join(RESULTS_DIR, 'exp5_substudy_b.png')
    plt.savefig(p, dpi=120, bbox_inches='tight'); plt.close()
    print(f"Saved: {p}")

    # Plot C: depth scaling
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    for fam, color, marker in [('Gaussian', 'blue', 'o'), ('Clayton', 'red', 's')]:
        sub = [r for r in rows_c if r['family'] == fam]
        depths = [r['depth'] for r in sub]
        etls   = [r['etl99'] for r in sub]
        qs     = [r['q99']   for r in sub]
        labels = [f"{'*' if r['approx'] else ''}{r['depth']}" for r in sub]
        axes[0].plot(depths, etls, f'-{marker}', color=color, ms=9, lw=2,
                     label=f'{fam}')
        axes[1].plot(depths, qs,   f'-{marker}', color=color, ms=9, lw=2,
                     label=f'{fam}')
    for ax, ylabel, title in [
        (axes[0], 'ETL_0.99', 'ETL_0.99 vs Hierarchy Depth'),
        (axes[1], 'q_0.99',   'q_0.99 vs Hierarchy Depth'),
    ]:
        ax.set_xlabel('Hierarchy depth m', fontsize=12)
        ax.set_ylabel(ylabel, fontsize=12)
        ax.set_title(title)
        ax.set_xticks([2, 3])
        ax.legend(); ax.grid(True, alpha=0.3)

    plt.suptitle('Exp 5 – Substudy C: Depth Sensitivity\n'
                 '(depth 3 = approximate two-stage aggregation, N=30)', fontsize=12)
    plt.tight_layout()
    p = os.path.join(RESULTS_DIR, 'exp5_substudy_c.png')
    plt.savefig(p, dpi=120, bbox_inches='tight'); plt.close()
    print(f"Saved: {p}")


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    print("=" * 65)
    print("Exp 5: Robustness Studies (histogram-based API)")
    print(f"  Gaussian: gamma_1={GAMMA_1}, gamma_2={GAMMA_2}, rho_top={RHO_TOP}")
    print(f"  Clayton:  zeta_1={ZETA_1}, zeta_2={ZETA_2}, theta={THETA}")
    print("=" * 65)
    t_start = time.time()

    try:
        # Substudy A
        rows_a, hdata_g_200, hdata_c_200 = substudy_a()

        # Substudy B
        rows_b = substudy_b()

        # Substudy C (uses N=200 histograms from substudy A for depth-2 baseline)
        rows_c = substudy_c(hdata_g_200, hdata_c_200)

        # Plots
        print("\nSaving plots...")
        make_plots(rows_a, rows_b, rows_c)

        # Final summary
        print("\n" + "=" * 65)
        print("SUMMARY")
        print("\nSubstudy A – Confidence Level Sensitivity (N=200):")
        print(f"  {'alpha':>7} {'q_G':>8} {'ETL_G':>8} {'MR%_G':>7} | {'q_C':>8} {'ETL_C':>8} {'MR%_C':>7}")
        print("  " + "-" * 60)
        for r in rows_a:
            print(f"  {r['alpha']:7.3f} {r['q_gauss']:8.4f} {r['etl_gauss']:8.4f}"
                  f" {r['mr_share_g']:7.2%} | {r['q_clay']:8.4f} {r['etl_clay']:8.4f}"
                  f" {r['mr_share_c']:7.2%}")

        print("\nSubstudy B – Sample Size Sensitivity (alpha=0.99):")
        print(f"  {'n':>5} {'n_bins':>7} {'q_G':>8} {'ETL_G':>8} {'q_C':>8} {'ETL_C':>8} {'rt(s)':>7}")
        print("  " + "-" * 55)
        for r in rows_b:
            print(f"  {r['n']:5d} {r['n_bins']:7d} {r['q_gauss']:8.4f} {r['etl_gauss']:8.4f}"
                  f" {r['q_clay']:8.4f} {r['etl_clay']:8.4f} {r['runtime']:7.1f}")

        print("\nSubstudy C – Depth Sensitivity (b=2, m in {2,3}):")
        print(f"  {'Family':<10} {'depth':>6} {'N':>5} {'q_0.99':>9} {'ETL_0.99':>10} {'approx':>8}")
        print("  " + "-" * 52)
        for r in rows_c:
            print(f"  {r['family']:<10} {r['depth']:6d} {r['N']:5d}"
                  f" {r['q99']:9.4f} {r['etl99']:10.4f} {str(r['approx']):>8}")
        print("=" * 65)

    except Exception as e:
        import traceback
        print(f"\nERROR: {e}")
        traceback.print_exc()
        return

    print(f"\nTotal runtime: {(time.time()-t_start)/60:.2f} minutes")


if __name__ == '__main__':
    main()
