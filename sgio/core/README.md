# sgio/core — SGIO Library

The core optimization library. Importable as `smgil` after `pip install -e .`.

## Modules

| File | Description |
|------|-------------|
| `solver.py` | S-MGIL quadratic optimizer — wraps Gurobi to solve the similarity-weighted IO problem |
| `constraints.py` | DASH constraint matrix builder — constructs `A`, `b` from USDA nutrient values |
| `tradeoff.py` | Tradeoff path runner — iterates the solver, tightening one DASH constraint per step; also `tune_weights()` |
| `preprocessing.py` | NHANES crosswalk builder and observation vector (`x_vector`, `W_S`) constructor |
| `config.py` | Gurobi environment setup from env vars; nutrient name constants |
| `validation.py` | Day-2 holdout projection, nutrient/food-space distance metrics, Wilcoxon test helpers |
| `plotting.py` | 6-panel validation figure generator |
| `reporting.py` | Diet summary tables, LaTeX table export |
| `mfp_preprocessing.py` | MFP-specific preprocessing (serving-space formulation) |
| `a4f_preprocessing.py` | AI4FoodDB preprocessing (experimental) |
| `__init__.py` | Public API exports |

## Key functions

```python
# Build NHANES crosswalk (FNDDS food codes → USDA FoodData Central)
build_crosswalk(fndds_xpt_path, usda_parquet_path)

# Build observation vector and similarity weight matrix
build_observation_vector(nhanes_csv, similarity_json, respondent_id, crosswalk)

# Build DASH constraint matrix
build_A_b(nutrient_csv, item_index, crosswalk, x_vector)

# Run SGIO tradeoff path (up to max_iterations constraint activations)
run_smgil_tradeoff(A, b, X_obs, W_S, item_index, meta, max_iterations=4)

# Day-2 holdout validation
project_holdout(day2_csv, item_index, crosswalk)
compute_distances(z_recommendations, x_holdout, W_S)
```
