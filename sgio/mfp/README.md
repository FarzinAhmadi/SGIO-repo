# sgio/mfp — MyFitnessPal Pipeline

Scripts for the MFP temporal-split holdout validation described in Section V-B of the paper.

## Files

| File | Description |
|------|-------------|
| `myfitnesspal.py` | MFP data loader: reads Kaggle TSV, applies inclusion criteria, builds per-user daily aggregates |
| `nhanes.py` | Shared NHANES data utilities reused in MFP pipeline |
| `usda.py` | USDA nutrient data utilities |
| `mfp_preprocessing.py` | MFP-specific preprocessing — serving-space formulation, temporal split, MFP→USDA crosswalk |
| `mfp_single_user.py` | Single MFP user demo |
| `mfp_cohort_validation.py` | Full MFP cohort validation (n=200) |
| `mfp_multi_obs_single_user.py` | Multi-observation SGIO demo on a single MFP user |

## Data layout expected

```
data/
  myfitnesspal/
    rawdata/
      mfp-diaries.tsv       Raw MFP food diary (from Kaggle)
    processed/
      myfitnesspal_foods.csv
      myfitnesspal_users.csv
  similarity/
    myfitnesspal_to_usda.json   MFP food name → USDA item crosswalk
    usda.npz                    Precomputed similarity matrix (release asset)
    usda_index.parquet
  usda/2017-2018/processed/
    nutrient_values_full.csv
```

## Key differences from the NHANES pipeline

- Decision space is in **servings** (one MFP log entry = one serving)
- Augmented item space is larger (~1,700 items vs. ~135 for NHANES)
- Magnesium excluded from DASH constraints (0% MFP nutrient coverage)
- Multi-observation formulation is the natural treatment (K ≫ 10 days per user)

## Expected outputs

- 200 users; 0 non-amenable (0%)
- Marginal costs ≈ 0 across all iterations (flat tradeoff path)
- 47% median nutrient distance reduction (W=153, p < 0.001)
