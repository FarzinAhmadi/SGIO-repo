# Notebooks

Reproducibility notebooks that walk through the key analyses end-to-end. All Gurobi credentials are loaded from environment variables (see root `.env.example`).

| Notebook | Description |
|----------|-------------|
| `01_nhanes_pipeline_demo.ipynb` | NHANES → observation vector → SGIO tradeoff path: end-to-end demo for a single respondent |
| `02_nhanes_il_models.ipynb` | IL / GIL / MGIL model exploration on NHANES data; compares inverse-learning formulations |
| `03_mfp_paper_analysis.ipynb` | MFP validation analysis and paper figures (Section V-B) |
| `04_mfp_final_outputs.ipynb` | MFP final paper outputs: cohort summary statistics, LaTeX table generation |

## Setup

Before running, activate the virtual environment and export Gurobi credentials:

```bash
source .venv/bin/activate
export $(cat .env | xargs)
jupyter lab
```
