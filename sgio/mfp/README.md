# MyFitnessPal Analysis Pipeline

Implements the MFP temporal-split holdout validation described in Section V-B of the paper.

## Files to add here

```
sgio/mfp/
├── README.md                   (this file)
├── config.yaml                 Hyperparameters (K=10, τ=100, caloric range 800-5000 kcal)
├── preprocess.py               Load MFP Kaggle dataset, apply inclusion criteria, aggregate daily logs
├── crosswalk.py                Map MFP food names → USDA FoodData Central items (fuzzy + embedding match)
├── temporal_split.py           80/20 temporal split: first 80% of days as input, last 20% as holdout
├── run_validation.py           Main entry point: run SGIO on all MFP users, produce results
├── benchmark.py                Temporal holdout natural distance benchmark
├── statistical_tests.py        One-sided Wilcoxon signed-rank tests
└── outputs/                    (gitignored) Per-user recommendation outputs
```

## Reproducing the paper results

```bash
python sgio/mfp/run_validation.py \
    --mfp-dir data/mfp/ \
    --similarity-matrix usda_similarity_matrix.npz \
    --config sgio/mfp/config.yaml \
    --output results/mfp_results.pkl
```

## Key parameters (from paper)

| Parameter | Value | Description |
|-----------|-------|-------------|
| `K` | 10 | Top-K similar neighbors per consumed item |
| `τ` | 100 | Non-amenability threshold |
| Min logged days | 10 | Minimum days to include a user |
| Caloric range | 800–5,000 kcal | Mean daily intake filter |
| Temporal split | 80/20 | Training / holdout fraction |

## Key differences from NHANES formulation

- Decision space is in **servings** (one MFP log entry = one serving), not grams
- Augmented item space is larger (~1,700 items vs. ~135 for NHANES)
- Magnesium excluded from DASH constraints (0% MFP coverage)
- Multi-observation formulation is the natural treatment (K ≫ 10 days per user)

## Expected outputs

- n=200 users
- 0 non-amenable (0%)
- Marginal costs ≈ 0 across all iterations (flat tradeoff path)
- 47% median nutrient distance reduction (W=153, p < 0.001)
