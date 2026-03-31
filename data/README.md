# Data

Raw data files are **not committed** to this repository (they are gitignored). This file documents what to download, where to put it, and which small reference files are committed.

---

## Committed files

The following files are small enough to commit and are included in the repository:

| File | Description |
|------|-------------|
| `USDA_descriptions.csv` | Human-readable food descriptions for all USDA FoodData Central items used in the similarity matrix |
| `crosswalks/nhanes_categoricals_manual.json` | Manual NHANES food category mappings for the FNDDS→USDA crosswalk |
| `crosswalks/nhanes_selected_cols.json` | NHANES dietary recall column selection configuration |
| `crosswalks/myfitnesspal_to_usda.json` | MFP food name → USDA FoodData Central item crosswalk |

The similarity matrix (`data/similarity/usda.npz`, `data/similarity/usda_index.parquet`) is **not committed** — download it as a GitHub release asset and place it in `data/similarity/`.

---

## NHANES 2017–2018

**Source:** [CDC NHANES](https://wwwn.cdc.gov/nchs/nhanes/continuousnhanes/default.aspx?BeginYear=2017)

Download these XPT files and place them under `data/nhanes/2017/rawdata/`:

- `DRXFCD_J.XPT` — FNDDS food code descriptions
- `DR1IFF_J.XPT` — Day 1 individual food files
- `DR2IFF_J.XPT` — Day 2 individual food files
- `DEMO_J.XPT` — Demographic variables (for age filter 18–75)

The pipeline (`sgio/nhanes/nhanes_to_smgil.py`) also expects preprocessed CSV files in `data/nhanes/2017/cleaned/`:
- `day1_interview.csv` — Day 1 dietary recall
- `day2_interview.csv` — Day 2 dietary recall (holdout)

**Inclusion criteria applied in the paper:**
- Adults aged 18–75
- Day 1 dietary recall status `DR1DRSTZ = 1` (reliable)
- Day 2 dietary recall status `DR2DRSTZ = 1` (reliable)
- Both recall days available

**Note:** Sample weights (`WTDRD1`, `WTDR2D`) are available in the files but were not applied; this is a methods paper demonstrating recommendation feasibility, not an epidemiological prevalence study.

---

## MyFitnessPal (MFP)

**Source:** [Kaggle — MyFitnessPal Food Diary Dataset](https://www.kaggle.com/datasets/zvikinozadze/myfitnesspal-dataset)
(Kiknozadze, 2020)

Download via Kaggle CLI and place in `data/myfitnesspal/rawdata/`:
```bash
kaggle datasets download zvikinozadze/myfitnesspal-dataset -p data/myfitnesspal/rawdata/ --unzip
```

The pipeline expects `mfp-diaries.tsv` (the raw food diary TSV) and writes processed CSVs to `data/myfitnesspal/processed/`.

**Inclusion criteria applied in the paper:**
- Users with ≥ 10 logged days
- Temporal split: first 80% of days for training/input, last 20% for holdout
- Caloric range: 800–5,000 kcal mean daily intake

---

## USDA FoodData Central

**Source:** [USDA FoodData Central](https://fdc.nal.usda.gov/download-foods.html)

Processed nutrient tables go in `data/usda/2017-2018/processed/`:
- `nutrient_values_full.csv` — Full nutrient matrix (MFP pipeline)
- `nutrient_values.csv` — Filtered nutrient matrix (NHANES pipeline)

The similarity matrix was built over items from the Foundation Foods and FNDDS sub-databases. The precomputed matrix is available as a [GitHub release asset](../../releases) — download `usda.npz` and `usda_index.parquet` and place them in `data/similarity/`.

If you wish to rebuild the similarity matrix from scratch, see [`similarity/README.md`](../similarity/README.md).
