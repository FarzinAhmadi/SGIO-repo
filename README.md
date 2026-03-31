# SGIO: Similarity-Guided Inverse Optimization for Personalized Dietary Recommendations

**Paper:** "Similarity-Guided Inverse Optimization for Personalized Dietary Recommendations via Food Swapping"
**Authors:** Farzin Ahmadi, Felix Parker, Layne C. Price, Raviteja Anantha, Valerie K. Sullivan, Lawrence J. Appel, Kimia Ghobadi
**Venue:** IEEE Journal of Biomedical and Health Informatics (J-BHI), submitted 2026
**Preprint:** To be submitted

---

## Overview

SGIO closes the gap between nutrient-level dietary science and actionable food-level recommendations. It integrates a food similarity matrix into the Modified Goal-Integrated Inverse Learning (MGIL) framework to generate personalized, item-level DASH-compliant food swap sequences. Every recommended diet is DASH-feasible by construction; the tradeoff path controls how aggressively DASH constraints are activated, not whether they hold.

**Interactive dashboard:** open `results/dashboard.html` in any browser to explore per-participant recommendations from the NHANES validation.

---

## Repository Structure

```
sgio-repo/
│
├── sgio/
│   ├── core/               # Core SGIO library (solver, constraints, tradeoff path)
│   ├── nhanes/             # NHANES 2017-2018 pipeline scripts
│   └── mfp/                # MyFitnessPal pipeline scripts
│
├── similarity/             # Food similarity pipeline — contact: Felix Parker
│
├── experiments/            # Runnable end-to-end experiment scripts
│
├── notebooks/              # Jupyter notebooks for interactive exploration
│
├── data/
│   ├── crosswalks/         # NHANES→USDA and MFP→USDA mapping files (committed)
│   ├── USDA_descriptions.csv
│   └── README.md           # Instructions for downloading NHANES, MFP, USDA data
│
├── results/
│   ├── dashboard.html      # Interactive NHANES validation dashboard
│   ├── dashboard_unscaled.html
│   ├── figures/            # Validation figures (PDFs/PNGs from paper)
│   └── tables/             # LaTeX tables from paper
│
└── paper/                  # LaTeX manuscript source
```

---

## Setup

Requires Python ≥ 3.12 and a [Gurobi](https://www.gurobi.com/) license (WLS or academic).

```bash
# Install uv (fast Python package manager)
pip install uv

# Create virtual environment and install
uv venv .venv --python 3.12
uv pip install -e ".[dev]"
source .venv/bin/activate
```

### Gurobi credentials

Copy `.env.example` to `.env` and fill in your WLS credentials:

```bash
cp .env.example .env
# edit .env with your GRB_WLSACCESSID, GRB_WLSSECRET, GRB_LICENSEID
```

Then load them before running:

```bash
export $(cat .env | xargs)
```

### Data

See [`data/README.md`](data/README.md) for instructions on downloading NHANES XPT files, the MFP Kaggle dataset, and USDA FoodData Central.

The precomputed USDA similarity matrix (`usda.npz`, `usda_index.parquet`) is available as a [GitHub release asset](../../releases) — download and place in `data/similarity/`.

---

## Reproducing the Results

### NHANES Validation

```bash
# Single respondent demo
python experiments/single_respondent.py

# Full cohort validation (n=356)
python experiments/cohort_validation.py

# Generate validation figures and Wilcoxon statistics
python experiments/plot_validation.py
```

### MFP Validation

```bash
# Single MFP user demo
python sgio/mfp/mfp_single_user.py

# Full MFP cohort validation (n=200)
python sgio/mfp/mfp_cohort_validation.py
```

### Notebooks

| Notebook | Description |
|----------|-------------|
| `notebooks/01_nhanes_pipeline_demo.ipynb` | NHANES → observation vector → SGIO tradeoff path |
| `notebooks/02_nhanes_il_models.ipynb` | IL/GIL/MGIL model exploration on NHANES |
| `notebooks/03_mfp_paper_analysis.ipynb` | MFP validation analysis and figures |
| `notebooks/04_mfp_final_outputs.ipynb` | MFP final paper outputs |

---

## Library Quick-Start

```python
from smgil import (
    build_crosswalk,
    build_observation_vector,
    build_A_b,
    run_smgil_tradeoff,
    tune_weights,
)

crosswalk = build_crosswalk(fndds_path, usda_parquet_path)
x_vector, W_S, item_index, meta = build_observation_vector(
    nhanes_csv, similarity_json, respondent_id, crosswalk=crosswalk,
)
A, b, names = build_A_b(nutrient_csv, item_index, crosswalk, x_vector)
W_S_tuned = tune_weights(W_S, meta)

results = run_smgil_tradeoff(
    A, b, x_vector.reshape(1, -1), W_S_tuned, item_index, meta,
    max_iterations=4, cost_threshold=50000,
)
```

---

## Citation

```bibtex
@article{ahmadi2026sgio,
  title   = {Similarity-Guided Inverse Optimization for Personalized Dietary
             Recommendations via Food Swapping},
  author  = {Ahmadi, Farzin and Parker, Felix and Price, Layne C. and
             Anantha, Raviteja and Sullivan, Valerie K. and
             Appel, Lawrence J. and Ghobadi, Kimia},
  journal = {IEEE Journal of Biomedical and Health Informatics},
  year    = {2026},
  note    = {Submitted. Preprint: arXiv:2603.17033}
}
```

---

## License

Code: MIT. See `LICENSE`.
Data: Subject to NHANES public-use terms and MyFitnessPal dataset terms — see `data/README.md`.
