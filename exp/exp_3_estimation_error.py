#!/usr/bin/env python3
"""
Exp 3: Finite-sum estimation error diagnostics for quantile and ETL,
using the histogram-based API for the aggregate loss distribution.

For alpha=0.99 and both Gaussian/Clayton hierarchies, computes:
  - U_1(a) = (1-a)*ETL(a)  [from etl_from_histogram]
  - U_2(a) = sum_{bins > q_a} (w_bin/tw)*bin_mid^2 + eta_a * q_a^2
  - sigma_ETL (Eq. 30) vs beta_aux in {0.90, 0.925, 0.95, 0.975}
  - sigma_q (Eqs. 31-32) via local density f_Delta(q_alpha) for delta
    in {0.05, 0.10, 0.25, 0.50, 1.0}
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
ALPHA = 0.99
N_BINS = 2000
CHUNK = 200

BETA_AUX_LIST = [0.90, 0.925, 0.95, 0.975]
DELTA_LIST = [0.05, 0.10, 0.25, 0.50, 1.0]

RESULTS_DIR = os.path.join(os.path.dirname(__file__), '..', 'results')
os.makedirs('results', exist_ok=True)


def compute_U2_from_histogram(hdata: dict, alpha: float) -> float:
    """Compute U_2(alpha) from histogram bins above q_alpha.

    U_2(alpha) = sum_{bins strictly above q_alpha bin} (w_bin/tw) * bin_mid^2
                 + eta_alpha * q_alpha^2

    where eta_alpha comes from etl_from_histogram (atom-correction factor).
    """
    bin_edges = hdata['bin_edges']
    bin_mids = hdata['bin_mids']
    hist_w = hdata['hist_w']
    tw = hdata['total_weight']

    q_alpha = quantile_from_histogram(hdata, alpha)
    etl_res = etl_from_histogram(q_alpha, hdata, alpha)
    eta = etl_res['eta']

    # Identify bin containing q_alpha (same logic as etl_from_histogram)
    q_bin = int(np.searchsorted(bin_edges, q_alpha, side='right')) - 1
    q_bin = int(np.clip(q_bin, 0, len(hist_w) - 1))

    pw = hist_w / tw
    # Bins strictly above q_bin
    U2 = float(np.sum(pw[q_bin + 1:] * bin_mids[q_bin + 1:] ** 2)) + eta * q_alpha ** 2
    return U2


def compute_sigma_etl(hdata: dict, alpha: float, beta_aux: float, n: int) -> float:
    """Compute sigma_ETL (paper Eq. 30) from histogram.

    Uses:
      U_1(a) = (1-a)*ETL(a)   [exact relation from ETL definition]
      U_2(a) = histogram sum above q_a (see compute_U2_from_histogram)
      I_1 = U_1(alpha) - U_1(beta_aux)
      I_2 = U_2(alpha) - U_2(beta_aux)

    Eq. (30):
      V = [alpha*q_alpha^2 + (1-beta)*q_beta^2] / d^2
          + I2 / d^2
          - [(1-beta)*q_beta + alpha*q_alpha + I1]^2 / d^2
      sigma_ETL = (1/sqrt(n)) * sqrt(max(0, V))
    """
    q_alpha = quantile_from_histogram(hdata, alpha)
    q_beta  = quantile_from_histogram(hdata, beta_aux)

    etl_alpha = etl_from_histogram(q_alpha, hdata, alpha)['etl']
    etl_beta  = etl_from_histogram(q_beta,  hdata, beta_aux)['etl']

    U1_alpha = (1.0 - alpha)    * etl_alpha
    U1_beta  = (1.0 - beta_aux) * etl_beta

    U2_alpha = compute_U2_from_histogram(hdata, alpha)
    U2_beta  = compute_U2_from_histogram(hdata, beta_aux)

    I1 = U1_alpha - U1_beta
    I2 = U2_alpha - U2_beta

    d  = alpha - beta_aux
    d2 = d ** 2

    term1 = (alpha * q_alpha ** 2 + (1.0 - beta_aux) * q_beta ** 2) / d2
    term2 = I2 / d2
    term3 = ((1.0 - beta_aux) * q_beta + alpha * q_alpha + I1) ** 2 / d2
    V = max(term1 + term2 - term3, 0.0)

    return float(np.sqrt(V) / np.sqrt(n))


def compute_sigma_q(hdata: dict, alpha: float, delta: float, n: int) -> float:
    """Compute sigma_q (paper Eqs. 31-32) from histogram using local density.

    f_Delta(q_alpha) = [F(q+Delta) - F(q-Delta)] / (2*Delta)
    sigma_q = sqrt(alpha*(1-alpha)/n) / f_Delta
    """
    q_alpha = quantile_from_histogram(hdata, alpha)
    f_upper = cdf_from_histogram(q_alpha + delta, hdata)
    f_lower = cdf_from_histogram(q_alpha - delta, hdata)
    f_delta = (f_upper - f_lower) / (2.0 * delta)
    f_delta = max(f_delta, 1e-10)
    return float(np.sqrt(alpha * (1.0 - alpha) / n) / f_delta)


def main():
    print("=" * 65)
    print("Exp 3: Estimation Error Diagnostics (histogram-based API)")
    print(f"  N={N}, seed={SEED}, alpha={ALPHA}")
    print(f"  beta_aux list: {BETA_AUX_LIST}")
    print(f"  delta list:    {DELTA_LIST}")
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

        # Step 3: For alpha=0.99, extract q_alpha, beta_aux q_beta, U_1, U_2
        print("\nStep 4: Computing estimation error metrics...")

        rows_etl = []
        rows_q   = []

        for fam, hdata in [('Gaussian', hdata_g), ('Clayton', hdata_c)]:
            print(f"\n  ── {fam} ──")
            q99 = quantile_from_histogram(hdata, ALPHA)
            etl99 = etl_from_histogram(q99, hdata, ALPHA)['etl']
            U1_99 = (1.0 - ALPHA) * etl99
            U2_99 = compute_U2_from_histogram(hdata, ALPHA)
            print(f"  q_0.99={q99:.4f}  ETL_0.99={etl99:.4f}")
            print(f"  U_1(0.99)={(U1_99):.6f}  U_2(0.99)={U2_99:.6f}")

            # sigma_ETL vs beta_aux
            print(f"\n  sigma_ETL sensitivity vs beta_aux:")
            print(f"  {'beta_aux':>9} {'q_beta':>9} {'U1_beta':>10} {'U2_beta':>10}"
                  f" {'I1':>10} {'I2':>10} {'sigma_ETL':>12} {'rel':>8}")
            print("  " + "-" * 82)
            for beta in BETA_AUX_LIST:
                q_b  = quantile_from_histogram(hdata, beta)
                etl_b = etl_from_histogram(q_b, hdata, beta)['etl']
                U1_b = (1.0 - beta) * etl_b
                U2_b = compute_U2_from_histogram(hdata, beta)
                I1 = U1_99 - U1_b
                I2 = U2_99 - U2_b
                sigma = compute_sigma_etl(hdata, ALPHA, beta, N)
                rel   = sigma / etl99 if etl99 > 0 else np.nan
                print(f"  {beta:9.3f} {q_b:9.4f} {U1_b:10.6f} {U2_b:10.4f}"
                      f" {I1:10.6f} {I2:10.4f} {sigma:12.6f} {rel:8.4f}")
                rows_etl.append(dict(
                    family=fam, beta_aux=beta,
                    q_beta=q_b, U1_beta=U1_b, U2_beta=U2_b,
                    I1=I1, I2=I2, sigma_etl=sigma, rel_sigma=rel,
                ))

            # sigma_q vs delta
            print(f"\n  sigma_q sensitivity vs delta:")
            print(f"  {'delta':>8} {'F_lo':>9} {'F_hi':>9} {'f_Delta':>10} {'sigma_q':>10} {'rel':>8}")
            print("  " + "-" * 58)
            for delta in DELTA_LIST:
                f_lo = cdf_from_histogram(q99 - delta, hdata)
                f_hi = cdf_from_histogram(q99 + delta, hdata)
                f_d  = max((f_hi - f_lo) / (2.0 * delta), 1e-10)
                sigma_q = float(np.sqrt(ALPHA * (1.0 - ALPHA) / N) / f_d)
                rel_q   = sigma_q / q99 if q99 > 0 else np.nan
                print(f"  {delta:8.2f} {f_lo:9.4f} {f_hi:9.4f} {f_d:10.5f}"
                      f" {sigma_q:10.6f} {rel_q:8.4f}")
                rows_q.append(dict(
                    family=fam, delta=delta,
                    f_lo=f_lo, f_hi=f_hi, f_delta=f_d,
                    sigma_q=sigma_q, q99=q99, rel_sigma=rel_q,
                ))

        # Step 5: Sensitivity plots
        print("\nStep 5: Saving sensitivity plots...")
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))

        ax = axes[0]
        for fam, marker in [('Gaussian', '-o'), ('Clayton', '-s')]:
            sub = [r for r in rows_etl if r['family'] == fam]
            betas  = [r['beta_aux']  for r in sub]
            sigmas = [r['sigma_etl'] for r in sub]
            ax.plot(betas, sigmas, marker, label=fam)
        ax.set_xlabel('Auxiliary level beta_aux')
        ax.set_ylabel('sigma_ETL  (Eq. 30)')
        ax.set_title('Sensitivity: sigma_ETL vs beta_aux')
        ax.legend()
        ax.grid(True, alpha=0.3)

        ax = axes[1]
        for fam, marker in [('Gaussian', '-o'), ('Clayton', '-s')]:
            sub = [r for r in rows_q if r['family'] == fam]
            deltas = [r['delta']   for r in sub]
            sigmas = [r['sigma_q'] for r in sub]
            ax.plot(deltas, sigmas, marker, label=fam)
        ax.set_xlabel('Bandwidth Delta')
        ax.set_ylabel('sigma_q  (Eqs. 31-32)')
        ax.set_title('Sensitivity: sigma_q vs Delta')
        ax.legend()
        ax.grid(True, alpha=0.3)

        plt.suptitle('Exp 3: Estimation Error Sensitivity Analysis', fontsize=13)
        plt.tight_layout()
        path = os.path.join(RESULTS_DIR, 'exp3_estimation_error.png')
        plt.savefig(path, dpi=120, bbox_inches='tight')
        plt.close()
        print(f"  Saved: {path}")

        # Print sensitivity table summary
        print("\n" + "=" * 65)
        print("SENSITIVITY TABLE – sigma_ETL vs beta_aux (alpha=0.99)")
        print(f"  {'Family':<10} {'beta_aux':>9} {'sigma_ETL':>12} {'rel_sigma':>12}")
        print("  " + "-" * 46)
        for r in rows_etl:
            print(f"  {r['family']:<10} {r['beta_aux']:9.3f} {r['sigma_etl']:12.6f}"
                  f" {r['rel_sigma']:12.4f}")

        print("\nSENSITIVITY TABLE – sigma_q vs delta (alpha=0.99)")
        print(f"  {'Family':<10} {'delta':>8} {'f_Delta':>10} {'sigma_q':>12} {'rel_sigma':>12}")
        print("  " + "-" * 53)
        for r in rows_q:
            print(f"  {r['family']:<10} {r['delta']:8.2f} {r['f_delta']:10.5f}"
                  f" {r['sigma_q']:12.6f} {r['rel_sigma']:12.4f}")
        print("=" * 65)

    except Exception as e:
        import traceback
        print(f"\nERROR: {e}")
        traceback.print_exc()
        return

    print(f"\nTotal runtime: {(time.time()-t_start)/60:.2f} minutes")


if __name__ == '__main__':
    main()
