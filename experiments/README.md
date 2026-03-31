# Experiments

Runnable end-to-end scripts for reproducing the NHANES validation results reported in the paper.

| File | Description |
|------|-------------|
| `single_respondent.py` | Single-respondent demo: builds observation vector, constructs DASH constraints, runs SGIO tradeoff path, and prints a diet report for one NHANES participant |
| `cohort_validation.py` | Full NHANES cohort validation (n=356): runs the Day-1 pipeline and Day-2 holdout projection for every participant; writes `results/smgil_all_diets.csv` and `results/smgil_day2_validation.csv` |
| `plot_validation.py` | Reads `smgil_day2_validation.csv`, generates the 6-panel validation figure (Fig. 2), and reports Wilcoxon signed-rank statistics for all four tradeoff steps |

## Usage

```bash
# Activate environment and load Gurobi credentials
source .venv/bin/activate
export $(cat .env | xargs)

# Single respondent demo
python experiments/single_respondent.py

# Full cohort (takes ~30–60 min depending on hardware)
python experiments/cohort_validation.py

# Generate figures and statistics
python experiments/plot_validation.py
```

For the MFP validation, see `sgio/mfp/mfp_cohort_validation.py`.
