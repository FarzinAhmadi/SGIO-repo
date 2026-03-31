# Data

This directory holds data access scripts and instructions. Raw data files are not committed to the repository.

---

## NHANES 2017–2018

**Source:** [CDC NHANES](https://wwwn.cdc.gov/nchs/nhanes/continuousnhanes/default.aspx?BeginYear=2017)

Files needed:
- `DR1IFF_J.XPT` — Day 1 individual food files
- `DR2IFF_J.XPT` — Day 2 individual food files
- `DEMO_J.XPT` — Demographic variables (for age filter 18–75)

**Inclusion criteria applied in the paper:**
- Adults aged 18–75
- Day 1 dietary recall status `DR1DRSTZ = 1` (reliable)
- Day 2 dietary recall status `DR2DRSTZ = 2` (reliable)
- Both recall days available

**Note:** Sample weights (`WTDRD1`, `WTDR2D`) are available in the files but were not applied; this is a methods paper demonstrating recommendation feasibility, not an epidemiological prevalence study.

Download script:
```bash
python data/download_nhanes.py --year 2017 --output data/nhanes/
```

---

## MyFitnessPal (MFP)

**Source:** [Kaggle — MyFitnessPal Food Diary Dataset](https://www.kaggle.com/datasets/zvikinozadze/myfitnesspal-dataset)
(Kiknozadze, 2020)

Download via Kaggle CLI:
```bash
kaggle datasets download zvikinozadze/myfitnesspal-dataset -p data/mfp/ --unzip
```

**Inclusion criteria applied in the paper:**
- Users with ≥ 10 logged days
- Temporal split: first 80% of days for training/input, last 20% for holdout
- Caloric range: 800–5,000 kcal mean daily intake

---

## USDA FoodData Central

**Source:** [USDA FoodData Central](https://fdc.nal.usda.gov/download-foods.html)

The similarity matrix was built over 7,338 items drawn from the Foundation Foods and FNDDS sub-databases. The precomputed matrix is available as a release asset (`usda_similarity_matrix.npz`) — you do not need to rebuild it to reproduce paper results.

If you wish to rebuild it, see [`similarity/README.md`](../similarity/README.md).
