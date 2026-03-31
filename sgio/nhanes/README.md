# NHANES Analysis Pipeline

Implements the NHANES 2017–2018 validation described in Section V-A of the paper.

## Files to add here

```
sgio/nhanes/
├── README.md                   (this file)
├── config.yaml                 Hyperparameters (K=10, τ=100, r_max=4, age range 18-75)
├── preprocess.py               Load NHANES XPT files, apply inclusion criteria, build nutrient vectors
├── crosswalk.py                Map DR1IFDCD food codes → USDA FoodData Central items
├── run_validation.py           Main entry point: run SGIO on all participants, produce results
├── benchmark.py                Day-2 natural distance benchmark
├── statistical_tests.py        One-sided Wilcoxon signed-rank tests (nutrient + food space)
└── outputs/                    (gitignored) Per-participant recommendation outputs
```

## Reproducing the paper results

```bash
python sgio/nhanes/run_validation.py \
    --nhanes-dir data/nhanes/ \
    --similarity-matrix usda_similarity_matrix.npz \
    --config sgio/nhanes/config.yaml \
    --output results/nhanes_results.pkl
```

## Key parameters (from paper)

| Parameter | Value | Description |
|-----------|-------|-------------|
| `K` | 10 | Top-K similar neighbors per consumed item |
| `τ` | 100 | Non-amenability threshold (g² in weighted objective) |
| `r_max` | 4 | Maximum constraint activation iterations |
| Age range | 18–75 | Participant inclusion |
| Reliability | DR1DRSTZ=1, DR2DRSTZ=1 | Both recall days reliable |

## Expected outputs

- n=356 participants after inclusion criteria
- 11 non-amenable (3%), 345 amenable
- 330 receive full 4-iteration tradeoff path
- Wilcoxon p < 0.001 at all r ∈ {1,2,3,4} in both nutrient and food-item space
