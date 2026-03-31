# Results

Pre-generated outputs from the NHANES and MFP validations reported in the paper.

## Dashboards

| File | Description |
|------|-------------|
| `dashboard.html` | Interactive per-participant SGIO recommendations from the NHANES validation (scaled nutrient display) |
| `dashboard_unscaled.html` | Same dashboard with raw (unscaled) nutrient values |

Both files are self-contained HTML — open directly in any modern browser, no server required.

## Summary CSVs

| File | Description |
|------|-------------|
| `smgil_all_diets.csv` | Per-participant recommended diets at each tradeoff step (NHANES cohort, n=356) |
| `smgil_day2_validation.csv` | Day-2 holdout distances and Wilcoxon statistics (NHANES validation) |
| `smgil_mfp_temporal_validation.csv` | Holdout distances from MFP temporal-split validation (n=200) |

Note: `nhanes_results.pkl` and `mfp_results.pkl` (raw solver output objects) are gitignored due to size; regenerate with `experiments/cohort_validation.py` and `sgio/mfp/mfp_cohort_validation.py`.

## Figures

| File | Description |
|------|-------------|
| `figures/smgil_validation_figures.png` | NHANES 6-panel validation figure (paper Fig. 2, PNG) |
| `figures/smgil_validation_figures_scaled.pdf` | NHANES validation figure (PDF, print quality) |
| `figures/smgil_mfp_validation_final.pdf` | MFP validation figure (paper Fig. 3, PDF) |

## LaTeX Tables

Pre-rendered `.tex` table fragments included in the paper:

| File | Description |
|------|-------------|
| `tables/mfp_cohort_summary_table.tex` | MFP cohort summary (Table in Section V-B) |
| `tables/mfp_final_table.tex` | MFP final cohort validation results |
| `tables/table1_nutrient_intake_93705.tex` | NHANES participant 93705 — nutrient intake (Table 1 example) |
| `tables/table1_nutrients_93704.tex` | NHANES participant 93704 — nutrient intake |
| `tables/table1_nutrients_93705.tex` | NHANES participant 93705 — nutrient intake (alternate format) |
| `tables/table2_tradeoff_93704.tex` | Tradeoff path summary for participant 93704 |
| `tables/table2_tradeoff_93705.tex` | Tradeoff path summary for participant 93705 |
| `tables/table2_tradeoff_summary_93705.tex` | Condensed tradeoff summary for participant 93705 |
