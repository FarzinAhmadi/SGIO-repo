# sgio/nhanes — NHANES 2017–2018 Pipeline

Scripts for the NHANES validation described in Section V-A of the paper.

## Files

| File | Description |
|------|-------------|
| `nhanes_to_smgil.py` | Main pipeline: loads NHANES XPT files, applies inclusion criteria, builds `x_vector` and `W_S` per participant, runs SGIO |
| `build_Ab.py` | Constructs the DASH constraint matrix `A` and bounds vector `b` from USDA nutrient data |
| `build_nutrient_matrix.py` | Builds the full nutrient matrix from USDA FoodData Central processed CSV |

End-to-end experiment scripts are in `experiments/` at the repo root.

## Data layout expected

```
data/
  nhanes/2017/
    rawdata/
      DRXFCD_J.xpt        FNDDS food code descriptions
    cleaned/
      day1_interview.csv   Day 1 dietary recall
      day2_interview.csv   Day 2 dietary recall (holdout)
  similarity/
    usda.npz              Precomputed similarity matrix (release asset)
    usda_index.parquet    USDA food index
  crosswalks/
    nhanes_categoricals_manual.json
    nhanes_selected_cols.json
  usda/2017-2018/processed/
    nutrient_values.csv
```

## Key parameters (from paper)

| Parameter | Value |
|-----------|-------|
| `K` | 10 neighbors per food item |
| `τ` | 100 g² (non-amenability threshold) |
| `r_max` | 4 constraint activation iterations |
| Age range | 18–75 |
| Reliability filter | `DR1DRSTZ=1`, `DR2DRSTZ=1` |

## Expected outputs

- 356 participants after inclusion; 11 non-amenable (3%), 345 amenable
- 330 participants receive full 4-iteration tradeoff path
- Wilcoxon p < 0.001 at all r ∈ {1,2,3,4} in both nutrient and food-item space
