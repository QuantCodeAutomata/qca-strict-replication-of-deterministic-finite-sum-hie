# Hierarchical Finite-Sum Loss Aggregation — Strict Replication

Replication of the paper's core numerical validation for a two-level, four-leaf
hierarchical portfolio loss aggregation model using **deterministic finite-sum
formulas** for Gaussian and mirrored Clayton copula hierarchies.

## Repository structure

```
├── data/               # placeholder (no external data needed)
├── src/                # core library modules
│   ├── marginals.py    # leaf-sample generation (lognormal MR, shifted-Pareto CR)
│   ├── copulas.py      # Gaussian & mirrored-Clayton density kernels
│   ├── aggregation.py  # hierarchical finite-sum formulas (chunked n⁴ computation)
│   ├── risk_measures.py# quantile, atom-corrected ETL, MR/CR contributions
│   ├── estimation_error.py  # ETL & quantile SE formulas (Eq. 27-32)
│   └── monte_carlo.py  # MC benchmark generator
├── exp/                # experiment scripts (exp_1 … exp_5)
├── tests/              # pytest unit + integration tests (98 tests, 0 failures)
├── results/            # figures (PNG) and RESULTS.md
└── AGENTS.md           # agent/developer knowledge base
```

## Quick start

```bash
# Run all experiments (requires ~10 min, N=200)
python exp/exp_1_replication.py
python exp/exp_2_etl_contributions.py
python exp/exp_3_estimation_error.py
python exp/exp_4_tail_comparison.py
python exp/exp_5_robustness.py

# Run all tests
pytest --tb=short -q
```

## Key parameters

| Parameter | Value |
|---|---|
| Leaf sample size n | 200 |
| Gaussian γ₁, γ₂, ρ_top | 0.75, 0.50, 0.50 |
| Mirrored Clayton ζ₁, ζ₂, θ | 2.3468, 1.0, 1.0 |
| MR marginal | Lognormal(μ=0, σ=0.25) |
| CR marginal | Shifted Pareto(x₀=1, c=1, β=2.5) |
| Target confidence level α | 0.99 |
| Random seed | 42 |

## Results summary

Deterministic formulas reproduce Monte Carlo to within **0.16 %** (Gaussian) and
**0.25 %** (Clayton) relative error for ETL at α = 0.99 — well within the 1.5 %
threshold reported in the paper. Full metrics are in `results/RESULTS.md`.
