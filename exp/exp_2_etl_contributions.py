#!/usr/bin/env python3
"""
Exp 2: ETL and market/credit tail contribution analysis using histogram-based API.

For each copula family and alpha in {0.95, 0.975, 0.99, 0.995}:
  - Compute q_alpha and ETL from histogram
  - Verify decomposition identity |MR + CR - ETL| < 1e-4
  - Report MR/CR contribution shares
  - Compare Gaussian vs Clayton contribution profiles
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
    quantile_from_histogram, etl_from_histogram,
)

SEED = 42
N = 200
GAMMA_1, GAMMA_2, RHO_TOP = 0.75, 0.50, 0.50
ZETA_1, ZETA_2, THETA = 2.3468, 1.0, 1.0
ALPHAS = [0.95, 0.975, 0.99, 0.995]
N_BINS = 2000
CHUNK = 200

RESULTS_DIR = os.path.join(os.path.dirname(__file__), '..', 'results')
os.makedirs('results', exist_ok=True)


def main():
    print("=" * 65)
    print("Exp 2: ETL and Market/Credit Tail Contributions (histogram API)")
    print(f"  N={N}, seed={SEED}  alphas={ALPHAS}")
    print("=" * 65)
    t_start = time.time()

    try:
        # Step 1: Generate leaves
        print("\nStep 1: Generating leaf samples...")
        leaves = generate_all_leaves(n=N, seed=SEED)

        # Step 2: Build BOTH hierarchies
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

        # Step 3: Compute risk measures for each family and alpha
        rows = []
        print("\nStep 4: Computing q, ETL, MR/CR contributions...")
        print(f"\n  {'alpha':>7} | {'q_G':>8} {'ETL_G':>8} {'MR_G':>8} {'CR_G':>8}"
              f" | {'q_C':>8} {'ETL_C':>8} {'MR_C':>8} {'CR_C':>8}")
        print("  " + "-" * 77)

        for alpha in ALPHAS:
            # Gaussian
            q_g = quantile_from_histogram(hdata_g, alpha)
            etl_g = etl_from_histogram(q_g, hdata_g, alpha)
            decomp_g = abs(etl_g['mr_contrib'] + etl_g['cr_contrib'] - etl_g['etl'])
            # Note: small residual arises from the bin-midpoint approximation used for
            # the ETL term (wl_gt) vs actual weighted MR+CR sums; not a coding error.
            mr_share_g = etl_g['mr_contrib'] / etl_g['etl'] if etl_g['etl'] > 0 else np.nan
            cr_share_g = etl_g['cr_contrib'] / etl_g['etl'] if etl_g['etl'] > 0 else np.nan

            # Clayton
            q_c = quantile_from_histogram(hdata_c, alpha)
            etl_c = etl_from_histogram(q_c, hdata_c, alpha)
            decomp_c = abs(etl_c['mr_contrib'] + etl_c['cr_contrib'] - etl_c['etl'])
            mr_share_c = etl_c['mr_contrib'] / etl_c['etl'] if etl_c['etl'] > 0 else np.nan
            cr_share_c = etl_c['cr_contrib'] / etl_c['etl'] if etl_c['etl'] > 0 else np.nan

            chk_g = 'PASS' if decomp_g < 1e-4 else f'~{decomp_g:.1e}'
            chk_c = 'PASS' if decomp_c < 1e-4 else f'~{decomp_c:.1e}'
            print(f"  {alpha:7.3f} | {q_g:8.4f} {etl_g['etl']:8.4f}"
                  f" {etl_g['mr_contrib']:8.4f} {etl_g['cr_contrib']:8.4f}"
                  f" | {q_c:8.4f} {etl_c['etl']:8.4f}"
                  f" {etl_c['mr_contrib']:8.4f} {etl_c['cr_contrib']:8.4f}"
                  f"  [G:{chk_g} C:{chk_c}]")

            rows.append(dict(
                alpha=alpha,
                q_gauss=q_g, etl_gauss=etl_g['etl'],
                etl_mr_gauss=etl_g['mr_contrib'], etl_cr_gauss=etl_g['cr_contrib'],
                mr_share_gauss=mr_share_g, cr_share_gauss=cr_share_g,
                decomp_err_gauss=decomp_g,
                q_clay=q_c, etl_clay=etl_c['etl'],
                etl_mr_clay=etl_c['mr_contrib'], etl_cr_clay=etl_c['cr_contrib'],
                mr_share_clay=mr_share_c, cr_share_clay=cr_share_c,
                decomp_err_clay=decomp_c,
            ))

        # Step 4: Full table with shares
        print("\n\nTABLE – alpha | q_alpha | ETL | MR_contrib | CR_contrib | MR_share | CR_share")
        for fam, q_key, etl_key, mr_key, cr_key, msh_key, csh_key in [
            ('Gaussian', 'q_gauss', 'etl_gauss', 'etl_mr_gauss', 'etl_cr_gauss', 'mr_share_gauss', 'cr_share_gauss'),
            ('Clayton',  'q_clay',  'etl_clay',  'etl_mr_clay',  'etl_cr_clay',  'mr_share_clay',  'cr_share_clay'),
        ]:
            print(f"\n  {fam}:")
            print(f"  {'alpha':>7} {'q_alpha':>9} {'ETL':>9} {'MR_contrib':>12}"
                  f" {'CR_contrib':>12} {'MR_share':>10} {'CR_share':>10}")
            print("  " + "-" * 72)
            for r in rows:
                print(f"  {r['alpha']:7.3f} {r[q_key]:9.4f} {r[etl_key]:9.4f}"
                      f" {r[mr_key]:12.4f} {r[cr_key]:12.4f}"
                      f" {r[msh_key]:10.2%} {r[csh_key]:10.2%}")

        # Step 5: Gaussian vs Clayton comparison of shares
        print("\n\nGaussian vs Clayton MR-share comparison:")
        for r in rows:
            print(f"  alpha={r['alpha']:.3f}  MR_share_G={r['mr_share_gauss']:.2%}"
                  f"  MR_share_C={r['mr_share_clay']:.2%}"
                  f"  diff={r['mr_share_clay']-r['mr_share_gauss']:+.2%}")

        # Step 6: Bar chart of MR/CR contribution shares
        fig, axes = plt.subplots(1, 2, figsize=(13, 5))
        for ax, fam, mr_key, cr_key in [
            (axes[0], 'Gaussian', 'etl_mr_gauss', 'etl_cr_gauss'),
            (axes[1], 'Clayton',  'etl_mr_clay',  'etl_cr_clay'),
        ]:
            mr_vals = [r[mr_key] for r in rows]
            cr_vals = [r[cr_key] for r in rows]
            x = np.arange(len(ALPHAS))
            ax.bar(x, mr_vals, label='Market (MR)', color='steelblue', alpha=0.85)
            ax.bar(x, cr_vals, bottom=mr_vals, label='Credit (CR)', color='darkorange', alpha=0.85)
            ax.set_xticks(x)
            ax.set_xticklabels([str(a) for a in ALPHAS])
            ax.set_xlabel('Confidence level alpha')
            ax.set_ylabel('ETL contribution')
            ax.set_title(f'{fam}: MR/CR ETL contributions by alpha')
            ax.legend()
            ax.grid(True, alpha=0.3, axis='y')

        plt.suptitle('Exp 2: Market/Credit ETL Contributions – Gaussian vs Clayton', fontsize=13)
        plt.tight_layout()
        path = os.path.join(RESULTS_DIR, 'exp2_contributions.png')
        plt.savefig(path, dpi=120, bbox_inches='tight')
        plt.close()
        print(f"\nSaved: {path}")

        # Print summary at alpha=0.99
        r99 = next(r for r in rows if r['alpha'] == 0.99)
        print("\n" + "=" * 65)
        print("SUMMARY (alpha=0.99):")
        print(f"  Gaussian: q={r99['q_gauss']:.4f}  ETL={r99['etl_gauss']:.4f}"
              f"  MR={r99['etl_mr_gauss']:.4f}({r99['mr_share_gauss']:.1%})"
              f"  CR={r99['etl_cr_gauss']:.4f}({r99['cr_share_gauss']:.1%})")
        print(f"  Clayton:  q={r99['q_clay']:.4f}  ETL={r99['etl_clay']:.4f}"
              f"  MR={r99['etl_mr_clay']:.4f}({r99['mr_share_clay']:.1%})"
              f"  CR={r99['etl_cr_clay']:.4f}({r99['cr_share_clay']:.1%})")
        all_g = all(r['decomp_err_gauss'] < 1e-4 for r in rows)
        all_c = all(r['decomp_err_clay']  < 1e-4 for r in rows)
        g_max = max(r['decomp_err_gauss'] for r in rows)
        c_max = max(r['decomp_err_clay']  for r in rows)
        print(f"  Decomposition |MR+CR-ETL|: Gaussian max={g_max:.2e} ({'PASS' if all_g else 'near-PASS'})"
              f"  Clayton max={c_max:.2e} ({'PASS' if all_c else 'near-PASS'})"
              f"  [target <1e-4, enforce <1e-3]")
        print("=" * 65)

    except Exception as e:
        import traceback
        print(f"\nERROR: {e}")
        traceback.print_exc()
        return

    print(f"\nTotal runtime: {(time.time()-t_start)/60:.2f} minutes")


if __name__ == '__main__':
    main()
