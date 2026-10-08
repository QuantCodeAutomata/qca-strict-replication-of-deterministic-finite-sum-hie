#!/usr/bin/env python3
"""
Exp 4: Comparative tail-behavior analysis of Gaussian vs mirrored Clayton
hierarchies using the histogram-based API.

Evaluates CDF on a 200-point grid spanning [q_0.10, q_0.999] for each family,
computes q_alpha and ETL_alpha for alpha in [0.90, 0.95, 0.975, 0.99, 0.995],
and reports MR/CR ETL contribution shares.

Saves:
  results/exp4_survival.png   – survival functions on semi-log axes
  results/exp4_etl_alpha.png  – ETL vs alpha for both families
  results/exp4_mr_share.png   – MR share vs alpha for both families
"""
import sys
import os
import time

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from src.marginals import generate_all_leaves
from src.aggregation import (
    build_hierarchy_gaussian, build_hierarchy_clayton,
    cdf_from_histogram, quantile_from_histogram, etl_from_histogram,
)

SEED = 42
N = 200
GAMMA_1, GAMMA_2, RHO_TOP = 0.75, 0.50, 0.50
ZETA_1, ZETA_2, THETA = 2.3468, 1.0, 1.0
ALPHAS = [0.90, 0.95, 0.975, 0.99, 0.995]
N_BINS = 2000
CHUNK = 200
N_GRID = 200   # CDF grid points

RESULTS_DIR = os.path.join(os.path.dirname(__file__), '..', 'results')
os.makedirs('results', exist_ok=True)


def main():
    print("=" * 65)
    print("Exp 4: Gaussian vs Clayton – Comparative Tail Analysis")
    print(f"  N={N}, seed={SEED}  alphas={ALPHAS}")
    print("=" * 65)
    t_start = time.time()

    try:
        # Step 1: Generate leaves (same for both families)
        print("\nStep 1: Generating leaf samples...")
        leaves = generate_all_leaves(n=N, seed=SEED)

        # Step 2: Build Gaussian hierarchy histogram
        print("\nStep 2: Building Gaussian hierarchy histogram...")
        t0 = time.time()
        g_result = build_hierarchy_gaussian(
            leaves['mr1_samples'], leaves['cr1_samples'],
            leaves['mr1_z'], leaves['cr1_z'],
            leaves['mr2_samples'], leaves['cr2_samples'],
            leaves['mr2_z'], leaves['cr2_z'],
            GAMMA_1, GAMMA_2, RHO_TOP,
            n_bins=N_BINS, chunk_size=CHUNK,
        )
        hdata_g = g_result['hdata']
        print(f"  Done [{time.time()-t0:.1f}s]")

        # Step 3: Build Clayton hierarchy histogram
        print("\nStep 3: Building Clayton hierarchy histogram...")
        t0 = time.time()
        c_result = build_hierarchy_clayton(
            leaves['mr1_samples'], leaves['cr1_samples'],
            leaves['mr1_u'], leaves['cr1_u'],
            leaves['mr2_samples'], leaves['cr2_samples'],
            leaves['mr2_u'], leaves['cr2_u'],
            ZETA_1, ZETA_2, THETA,
            n_bins=N_BINS, chunk_size=CHUNK,
        )
        hdata_c = c_result['hdata']
        print(f"  Done [{time.time()-t0:.1f}s]")

        # Step 4a: CDF on grid [q_0.10, q_0.999] for each family (200 points)
        print("\nStep 4: CDF on 200-point grid [q_0.10, q_0.999]...")
        grids, cdfs_g, cdfs_c = {}, {}, {}
        for fam, hdata, cdf_dict in [('Gaussian', hdata_g, cdfs_g), ('Clayton', hdata_c, cdfs_c)]:
            l_lo = quantile_from_histogram(hdata, 0.10)
            l_hi = quantile_from_histogram(hdata, 0.999)
            l_grid = np.linspace(l_lo, l_hi, N_GRID)
            grids[fam] = l_grid
            for l in l_grid:
                cdf_dict[l] = cdf_from_histogram(l, hdata)
            print(f"  {fam}: [{l_lo:.3f}, {l_hi:.3f}]")

        # Step 4b: q_alpha and ETL_alpha for each family and alpha
        print("\nStep 4b: q_alpha and ETL_alpha for each alpha level...")
        rows = []
        print(f"\n  {'alpha':>7} | {'q_G':>8} {'ETL_G':>8} {'MR_G':>7} {'CR_G':>7}"
              f" | {'q_C':>8} {'ETL_C':>8} {'MR_C':>7} {'CR_C':>7}")
        print("  " + "-" * 71)
        for alpha in ALPHAS:
            q_g   = quantile_from_histogram(hdata_g, alpha)
            etl_g = etl_from_histogram(q_g, hdata_g, alpha)
            mr_g  = etl_g['mr_contrib']
            cr_g  = etl_g['cr_contrib']
            msh_g = mr_g / etl_g['etl'] if etl_g['etl'] > 0 else np.nan
            csh_g = cr_g / etl_g['etl'] if etl_g['etl'] > 0 else np.nan

            q_c   = quantile_from_histogram(hdata_c, alpha)
            etl_c = etl_from_histogram(q_c, hdata_c, alpha)
            mr_c  = etl_c['mr_contrib']
            cr_c  = etl_c['cr_contrib']
            msh_c = mr_c / etl_c['etl'] if etl_c['etl'] > 0 else np.nan
            csh_c = cr_c / etl_c['etl'] if etl_c['etl'] > 0 else np.nan

            print(f"  {alpha:7.3f} | {q_g:8.4f} {etl_g['etl']:8.4f} {msh_g:7.2%} {csh_g:7.2%}"
                  f" | {q_c:8.4f} {etl_c['etl']:8.4f} {msh_c:7.2%} {csh_c:7.2%}")
            rows.append(dict(
                alpha=alpha,
                q_G=q_g, etl_G=etl_g['etl'], mr_G=mr_g, cr_G=cr_g,
                mr_share_G=msh_g, cr_share_G=csh_g,
                q_C=q_c, etl_C=etl_c['etl'], mr_C=mr_c, cr_C=cr_c,
                mr_share_C=msh_c, cr_share_C=csh_c,
                q_diff=q_c - q_g, etl_diff=etl_c['etl'] - etl_g['etl'],
            ))

        # Step 5: CDF table at key loss levels (adaptive)
        key_probs = [0.50, 0.70, 0.80, 0.90, 0.95, 0.99, 0.995]
        print("\nCDF TABLE at key probability levels:")
        print(f"  {'prob':>6} {'l_G':>8} {'F_G':>8} {'l_C':>8} {'F_C':>8} {'|diff|':>8}")
        print("  " + "-" * 52)
        for p in key_probs:
            l_g = quantile_from_histogram(hdata_g, p)
            l_c = quantile_from_histogram(hdata_c, p)
            f_g = cdf_from_histogram(l_g, hdata_g)
            f_c = cdf_from_histogram(l_c, hdata_c)
            print(f"  {p:6.3f} {l_g:8.3f} {f_g:8.4f} {l_c:8.3f} {f_c:8.4f}"
                  f" {abs(f_g - f_c):8.4f}")

        # Step 6: Gaussian vs Clayton differences
        r99 = next(r for r in rows if r['alpha'] == 0.99)
        print(f"\nGaussian vs Clayton documented differences (alpha=0.99):")
        print(f"  q_C - q_G    = {r99['q_diff']:+.4f}")
        print(f"  ETL_C - ETL_G = {r99['etl_diff']:+.4f}")
        print(f"  Clayton ETL > Gaussian ETL: {r99['etl_C'] > r99['etl_G']}")
        print(f"  MR share G={r99['mr_share_G']:.2%}  C={r99['mr_share_C']:.2%}  diff={r99['mr_share_C']-r99['mr_share_G']:+.2%}")

        # Step 7: Plots

        # 7a: Survival functions (semi-log)
        fig, ax = plt.subplots(figsize=(9, 6))
        l_g_arr = grids['Gaussian']
        l_c_arr = grids['Clayton']
        surv_g  = np.array([1.0 - cdfs_g[l] for l in l_g_arr])
        surv_c  = np.array([1.0 - cdfs_c[l] for l in l_c_arr])
        ax.semilogy(l_g_arr, np.maximum(surv_g, 1e-6), 'b-', lw=2, label='Gaussian')
        ax.semilogy(l_c_arr, np.maximum(surv_c, 1e-6), 'r-', lw=2, label='Clayton')
        for alpha, sym_g, sym_c in [(0.99, 'bo', 'rs'), (0.995, 'b^', 'r^')]:
            q_g = next(r['q_G'] for r in rows if r['alpha'] == alpha)
            q_c = next(r['q_C'] for r in rows if r['alpha'] == alpha)
            ax.semilogy([q_g], [1 - alpha], sym_g, ms=8, label=f'q_{alpha} Gaussian')
            ax.semilogy([q_c], [1 - alpha], sym_c, ms=8, label=f'q_{alpha} Clayton')
        ax.set_xlabel('Total Loss L', fontsize=12)
        ax.set_ylabel('Survival probability  P(L > l)  [log scale]', fontsize=12)
        ax.set_title('Exp 4: Survival Functions – Gaussian vs Clayton', fontsize=13)
        ax.legend(fontsize=9)
        ax.grid(True, alpha=0.3)
        plt.tight_layout()
        p1 = os.path.join(RESULTS_DIR, 'exp4_survival.png')
        plt.savefig(p1, dpi=120, bbox_inches='tight')
        plt.close()
        print(f"\nSaved: {p1}")

        # 7b: ETL vs alpha
        fig, ax = plt.subplots(figsize=(8, 5))
        alphas_arr = [r['alpha'] for r in rows]
        ax.plot(alphas_arr, [r['etl_G'] for r in rows], 'b-o', lw=2, ms=7, label='Gaussian')
        ax.plot(alphas_arr, [r['etl_C'] for r in rows], 'r-s', lw=2, ms=7, label='Clayton')
        ax.set_xlabel('Confidence level alpha', fontsize=12)
        ax.set_ylabel('ETL_alpha', fontsize=12)
        ax.set_title('Exp 4: ETL vs Confidence Level – Gaussian vs Clayton', fontsize=13)
        ax.legend()
        ax.grid(True, alpha=0.3)
        plt.tight_layout()
        p2 = os.path.join(RESULTS_DIR, 'exp4_etl_alpha.png')
        plt.savefig(p2, dpi=120, bbox_inches='tight')
        plt.close()
        print(f"Saved: {p2}")

        # 7c: MR share vs alpha
        fig, ax = plt.subplots(figsize=(8, 5))
        ax.plot(alphas_arr, [r['mr_share_G'] for r in rows], 'b-o', lw=2, ms=7,
                label='MR share (Gaussian)')
        ax.plot(alphas_arr, [r['cr_share_G'] for r in rows], 'b--o', lw=1.5, ms=7,
                label='CR share (Gaussian)')
        ax.plot(alphas_arr, [r['mr_share_C'] for r in rows], 'r-s', lw=2, ms=7,
                label='MR share (Clayton)')
        ax.plot(alphas_arr, [r['cr_share_C'] for r in rows], 'r--s', lw=1.5, ms=7,
                label='CR share (Clayton)')
        ax.set_xlabel('Confidence level alpha', fontsize=12)
        ax.set_ylabel('ETL contribution share', fontsize=12)
        ax.set_title('Exp 4: MR/CR Share vs Confidence Level', fontsize=13)
        ax.legend(fontsize=9)
        ax.grid(True, alpha=0.3)
        plt.tight_layout()
        p3 = os.path.join(RESULTS_DIR, 'exp4_mr_share.png')
        plt.savefig(p3, dpi=120, bbox_inches='tight')
        plt.close()
        print(f"Saved: {p3}")

        # Side-by-side comparison table
        print("\n" + "=" * 65)
        print("SIDE-BY-SIDE TABLE: alpha | q_G | q_C | ETL_G | ETL_C | MR%_G | MR%_C")
        print(f"  {'alpha':>7} {'q_G':>8} {'q_C':>8} {'ETL_G':>8} {'ETL_C':>8}"
              f" {'MRsh_G':>8} {'MRsh_C':>8}")
        print("  " + "-" * 59)
        for r in rows:
            print(f"  {r['alpha']:7.3f} {r['q_G']:8.4f} {r['q_C']:8.4f}"
                  f" {r['etl_G']:8.4f} {r['etl_C']:8.4f}"
                  f" {r['mr_share_G']:8.2%} {r['mr_share_C']:8.2%}")
        print("=" * 65)

    except Exception as e:
        import traceback
        print(f"\nERROR: {e}")
        traceback.print_exc()
        return

    print(f"\nTotal runtime: {(time.time()-t_start)/60:.2f} minutes")


if __name__ == '__main__':
    main()
