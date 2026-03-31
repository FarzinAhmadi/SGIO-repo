# SGIO: Similarity-Guided Inverse Optimization for Personalized Dietary Recommendations

**Paper:** "Similarity-Guided Inverse Optimization for Personalized Dietary Recommendations via Food Swapping"
**Authors:** Farzin Ahmadi, Felix Parker, Layne C. Price, Raviteja Anantha, Valerie K. Sullivan, Lawrence J. Appel, Kimia Ghobadi
**Venue:** IEEE Journal of Biomedical and Health Informatics (J-BHI), submitted 2026
**Preprint:** [arXiv:2603.XXXXX](https://arxiv.org/abs/2603.XXXXX)

---

## Overview

SGIO closes the gap between nutrient-level dietary science and actionable food-level recommendations. It integrates a food similarity matrix into the Modified Goal-Integrated Inverse Learning (MGIL) framework to generate personalized, item-level DASH-compliant food swap sequences. Every recommended diet is DASH-feasible by construction; the tradeoff path controls how aggressively DASH constraints are activated, not whether they hold.

### Repository Structure

```
sgio-repo/
├── paper/          LaTeX source for the manuscript
├── similarity/     Food similarity scoring pipeline (contact: Felix Parker)
├── sgio/
│   ├── core/       SGIO optimization formulation (MGIL + similarity-discounted objective)
│   ├── nhanes/     NHANES 2017-2018 data pipeline and validation
│   └── mfp/        MyFitnessPal data pipeline and validation
├── data/           Data access instructions (NHANES, MFP, USDA FoodData Central)
├── results/        Validation figures and interactive dashboard
└── notebooks/      Reproducibility notebooks
```

---

## Reproducing the Results

### 1. Environment

```bash
pip install -r requirements.txt
```

### 2. Data

See [`data/README.md`](data/README.md) for instructions on downloading NHANES and MFP datasets.

### 3. Similarity Matrix

The precomputed similarity matrix over USDA FoodData Central items is provided as a release asset: `usda_similarity_matrix.npz`. See [`similarity/README.md`](similarity/README.md) for details on how it was constructed.

### 4. Running the NHANES Validation

```bash
python sgio/nhanes/run_validation.py --config sgio/nhanes/config.yaml
```

### 5. Running the MFP Validation

```bash
python sgio/mfp/run_validation.py --config sgio/mfp/config.yaml
```

### 6. Dashboard

Open `results/dashboard.html` in any modern browser.

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
  note    = {Submitted}
}
```

---

## License
Data: Subject to NHANES public use data terms and MyFitnessPal dataset terms (see `data/README.md`).
