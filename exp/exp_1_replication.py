#!/usr/bin/env python3
"""
Exp 1: Replication of deterministic hierarchical aggregation vs Monte Carlo
benchmark using the histogram-based API.

Validates Gaussian and mirrored Clayton copula hierarchies. Adaptive validation
points are derived from paper CDF targets (0.424, 0.595, 0.677) rather than
fixed loss values, because with mu_M=0/sigma_M=0.25 and shifted Pareto(x0=1,
c=1,beta=2.5) the typical loss range is [3,24], not [25,30].

Checks:
  - wsum_dev < 1e-4
  - max_cdf_discrepancy < 0.02
  - q_0.99 and ETL_0.99 relative error vs MC < 1.5%
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
from src.monte_carlo import run_monte_carlo_gaussian, run_monte_carlo_clayton

SEED = 42
N = 200
GAMMA_1, GAMMA_2, RHO_TOP = 0.75, 0.50, 0.50
ZETA_1, ZETA_2, THETA = 2.3468, 1.0, 1.0
ALPHA = 0.99
N_BINS = 2000
CHUNK = 200
# Paper CDF target levels at l=25,28,30 respectively (used as adaptive targets)
TARGET_CDF_LEVELS = [0.424, 0.595, 0.677]

RESULTS_DIR = os.path.join(os.path.dirname(__file__), '..', 'results')
os.makedirs('results', exist_ok=True)


def main():
    print("=" * 65)
    print("Exp 1: Deterministic Histogram vs Monte Carlo Benchmark")
    print(f"  N={N}, seed={SEED}")
    print(f"  Gaussian: gamma_1={GAMMA_1}, gamma_2={GAMMA_2}, rho_top={RHO_TOP}")
    print(f"  Clayton:  zeta_1={ZETA_1}, zeta_2={ZETA_2}, theta={THETA}")
    print(f"  Adaptive CDF target levels: {TARGET_CDF_LEVELS}")
    print("=" * 65)
    t_start = time.time()

    try:
        # Step 1: Generate leaves
        print("\nStep 1: Generating leaf samples (N=200, seed=42)...")
        leaves = generate_all_leaves(n=N, seed=SEED)

        # Step 2: Build Gaussian hierarchy histogram in one pass
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
        wsum_dev_g = max(g_result['wsum_dev1'], g_result['wsum_dev2'])
        print(f"  Done [{time.time()-t0:.1f}s]  wsum_dev={wsum_dev_g:.2e}")

        # Step 3: Build Clayton hierarchy histogram in one pass
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

        # Step 4: Adaptive validation points at CDF levels 0.424, 0.595, 0.677
        print("\nStep 4: Computing adaptive validation points...")
        adap_g = [quantile_from_histogram(hdata_g, p) for p in TARGET_CDF_LEVELS]
        adap_c = [quantile_from_histogram(hdata_c, p) for p in TARGET_CDF_LEVELS]
        print(f"  Gaussian l-points: {[f'{v:.3f}' for v in adap_g]}")
        print(f"  Clayton  l-points: {[f'{v:.3f}' for v in adap_c]}")

        # Step 5: CDF at adaptive points (deterministic from histogram)
        print("\nStep 5: CDF at adaptive validation points (det vs MC)...")
        cdf_g_det = {l: cdf_from_histogram(l, hdata_g) for l in adap_g}
        cdf_c_det = {l: cdf_from_histogram(l, hdata_c) for l in adap_c}

        # Step 6: q_0.99 and ETL_0.99 (deterministic)
        print("\nStep 6: q_0.99 and ETL_0.99 from histogram...")
        q_g = quantile_from_histogram(hdata_g, ALPHA)
        etl_g_res = etl_from_histogram(q_g, hdata_g, ALPHA)
        q_c = quantile_from_histogram(hdata_c, ALPHA)
        etl_c_res = etl_from_histogram(q_c, hdata_c, ALPHA)
        print(f"  Gaussian: q_0.99={q_g:.4f}  ETL_0.99={etl_g_res['etl']:.4f}")
        print(f"  Clayton:  q_0.99={q_c:.4f}  ETL_0.99={etl_c_res['etl']:.4f}")

        # Step 7: MC benchmark with n_sim=200000, grid_points = adaptive points
        print("\nStep 7: Monte Carlo benchmarks (n_sim=200000)...")
        t0 = time.time()
        mc_g = run_monte_carlo_gaussian(
            leaves,
            {'gamma_1': GAMMA_1, 'gamma_2': GAMMA_2, 'rho_top': RHO_TOP},
            n_sim=200_000, seed=SEED, grid_points=adap_g,
        )
        print(f"  Gaussian MC [{time.time()-t0:.1f}s]: q99={mc_g['q99']:.4f}  ETL99={mc_g['etl99']:.4f}")

        t0 = time.time()
        mc_c = run_monte_carlo_clayton(
            leaves,
            {'zeta_1': ZETA_1, 'zeta_2': ZETA_2, 'theta': THETA},
            n_sim=200_000, seed=SEED, grid_points=adap_c,
        )
        print(f"  Clayton  MC [{time.time()-t0:.1f}s]: q99={mc_c['q99']:.4f}  ETL99={mc_c['etl99']:.4f}")

        # Step 8: Relative errors for q_0.99 and ETL_0.99
        q_g_relerr = abs(q_g - mc_g['q99']) / (abs(mc_g['q99']) + 1e-12)
        etl_g_relerr = abs(etl_g_res['etl'] - mc_g['etl99']) / (abs(mc_g['etl99']) + 1e-12)
        q_c_relerr = abs(q_c - mc_c['q99']) / (abs(mc_c['q99']) + 1e-12)
        etl_c_relerr = abs(etl_c_res['etl'] - mc_c['etl99']) / (abs(mc_c['etl99']) + 1e-12)

        g_diffs = [abs(cdf_g_det[l] - mc_g['cdf_at_grid'][l]) for l in adap_g]
        c_diffs = [abs(cdf_c_det[l] - mc_c['cdf_at_grid'][l]) for l in adap_c]
        max_cdf_g = max(g_diffs)
        max_cdf_c = max(c_diffs)

        # Step 9: Validation checks
        print("\nStep 9: Validation checks...")
        chk_wsum = 'PASS' if wsum_dev_g < 1e-4 else 'FAIL'
        chk_cdf_g = 'PASS' if max_cdf_g < 0.02 else 'FAIL'
        chk_cdf_c = 'PASS' if max_cdf_c < 0.02 else 'FAIL'
        chk_qg = 'PASS' if q_g_relerr < 0.015 else 'FAIL'
        chk_eg = 'PASS' if etl_g_relerr < 0.015 else 'FAIL'
        chk_qc = 'PASS' if q_c_relerr < 0.015 else 'FAIL'
        chk_ec = 'PASS' if etl_c_relerr < 0.015 else 'FAIL'
        print(f"  wsum_dev (Gaussian):           {wsum_dev_g:.2e}  [{chk_wsum}]")
        print(f"  max_cdf_discrepancy Gaussian:  {max_cdf_g:.4f}  [{chk_cdf_g}]")
        print(f"  max_cdf_discrepancy Clayton:   {max_cdf_c:.4f}  [{chk_cdf_c}]")
        print(f"  Gaussian q_0.99 rel_err:       {q_g_relerr:.4f}  [{chk_qg}]")
        print(f"  Gaussian ETL_0.99 rel_err:     {etl_g_relerr:.4f}  [{chk_eg}]")
        print(f"  Clayton  q_0.99 rel_err:       {q_c_relerr:.4f}  [{chk_qc}]")
        print(f"  Clayton  ETL_0.99 rel_err:     {etl_c_relerr:.4f}  [{chk_ec}]")

        # Step 10: CDF comparison plot
        print("\nStep 10: Saving CDF comparison plot...")
        fig, axes = plt.subplots(1, 2, figsize=(14, 6))
        for ax, hdata, mc, adap, cdf_det, label, color in [
            (axes[0], hdata_g, mc_g, adap_g, cdf_g_det, 'Gaussian', 'tab:blue'),
            (axes[1], hdata_c, mc_c, adap_c, cdf_c_det, 'Clayton',  'tab:orange'),
        ]:
            l_lo = quantile_from_histogram(hdata, 0.05)
            l_hi = quantile_from_histogram(hdata, 0.999)
            l_grid = np.linspace(l_lo, l_hi, 300)
            cdf_det_line = np.array([cdf_from_histogram(l, hdata) for l in l_grid])
            cdf_mc_line = np.array([mc['emp_cdf'](l) for l in l_grid])

            ax.plot(l_grid, cdf_det_line, color=color, lw=2, label='Deterministic (histogram)')
            ax.plot(l_grid, cdf_mc_line, 'k--', lw=1.5, label='MC (200k draws)')
            ax.scatter(adap, [cdf_det[l] for l in adap], marker='o', s=80,
                       color=color, zorder=5, label='Adaptive points (det)')
            ax.scatter(adap, [mc['cdf_at_grid'][l] for l in adap], marker='x', s=100,
                       color='red', zorder=5, lw=2, label='Adaptive points (MC)')
            ax.set_title(f'{label} Hierarchy', fontsize=12)
            ax.set_xlabel('Total Loss L')
            ax.set_ylabel('CDF  F_L(l)')
            ax.legend(fontsize=9)
            ax.grid(True, alpha=0.3)

        plt.suptitle('Exp 1: CDF Comparison – Deterministic vs Monte Carlo', fontsize=13)
        plt.tight_layout()
        path = os.path.join(RESULTS_DIR, 'exp1_cdf_comparison.png')
        plt.savefig(path, dpi=120, bbox_inches='tight')
        plt.close()
        print(f"  Saved: {path}")

        # Step 11: Table 1 equivalent – CDF at adaptive points
        print("\n" + "=" * 65)
        print("TABLE 1 – CDF at Adaptive Validation Points")
        print(f"  (target CDF levels: {TARGET_CDF_LEVELS})")
        print(f"  {'Family':<10} {'l_adap':>8} {'target':>7} {'CDF_det':>9} {'CDF_mc':>9} {'|diff|':>8}")
        print("-" * 65)
        for i in range(len(TARGET_CDF_LEVELS)):
            p = TARGET_CDF_LEVELS[i]
            l_g = adap_g[i]
            print(f"  {'Gaussian':<10} {l_g:8.3f} {p:7.3f} {cdf_g_det[l_g]:9.4f}"
                  f" {mc_g['cdf_at_grid'][l_g]:9.4f} {g_diffs[i]:8.4f}")
        for i in range(len(TARGET_CDF_LEVELS)):
            p = TARGET_CDF_LEVELS[i]
            l_c = adap_c[i]
            print(f"  {'Clayton':<10} {l_c:8.3f} {p:7.3f} {cdf_c_det[l_c]:9.4f}"
                  f" {mc_c['cdf_at_grid'][l_c]:9.4f} {c_diffs[i]:8.4f}")

        # Step 12: q_0.99 and ETL_0.99 det vs MC
        print("\nTABLE 2 – q_0.99 and ETL_0.99: Deterministic vs MC")
        print(f"  {'Family':<10} {'q_det':>8} {'q_mc':>8} {'q_relerr':>10}"
              f" {'ETL_det':>9} {'ETL_mc':>9} {'ETL_relerr':>12}")
        print("-" * 65)
        print(f"  {'Gaussian':<10} {q_g:8.4f} {mc_g['q99']:8.4f} {q_g_relerr:10.4f}"
              f" {etl_g_res['etl']:9.4f} {mc_g['etl99']:9.4f} {etl_g_relerr:12.4f}"
              f"  [{chk_qg}/{chk_eg}]")
        print(f"  {'Clayton':<10} {q_c:8.4f} {mc_c['q99']:8.4f} {q_c_relerr:10.4f}"
              f" {etl_c_res['etl']:9.4f} {mc_c['etl99']:9.4f} {etl_c_relerr:12.4f}"
              f"  [{chk_qc}/{chk_ec}]")
        print("=" * 65)

    except Exception as e:
        import traceback
        print(f"\nERROR: {e}")
        traceback.print_exc()
        return

    print(f"\nTotal runtime: {(time.time()-t_start)/60:.2f} minutes")


if __name__ == '__main__':
    main()
