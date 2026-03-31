# Food Similarity Scoring Pipeline

**Contact / owner: Felix Parker**

This module builds the multi-signal food similarity matrix over USDA FoodData Central items used by SGIO.

---

## Overview

The similarity pipeline produces a matrix $S \in [0,1]^{N \times N}$ over $N = 7{,}338$ USDA items, where $S_{ij}$ reflects how substitutable food $i$ and food $j$ are — both nutritionally and behaviorally.

Four signals are combined:

| Component | Description | Weight |
|-----------|-------------|--------|
| Dense embedding | Qwen3-Embedding-0.6B cosine similarity over food name + category text | 0.56 |
| Macronutrient profile | Logistic-transformed L1 distance over 8 nutrient fields (per 100g) | 0.28 |
| Category Jaccard | Overlap of USDA 2021–2023 food category tags | 0.16 |
| LLM reranker | Qwen3-Reranker-0.6B score; applied as convex combination with matrix score | — |

The final similarity score for each food pair $(i, j)$ in the top-$K$ neighborhood is:

$$S_{ij} = w_\text{mat} \cdot S_{ij}^\text{mat} + w_\text{rr} \cdot S_{ij}^\text{rr}$$

---

## Files to add here

```
similarity/
├── README.md                   (this file)
├── build_embedding_index.py    Build FAISS index from Qwen3 embeddings
├── compute_macro_scores.py     Pairwise macronutrient profile scores
├── compute_category_scores.py  Category Jaccard scores
├── rerank.py                   LLM reranker refinement (Qwen3-Reranker-0.6B)
├── build_similarity_matrix.py  Combine components → final matrix
├── crosswalk_nhanes.py         Map NHANES food codes → USDA FoodData Central items
├── crosswalk_mfp.py            Map MFP food names → USDA FoodData Central items
└── configs/
    └── similarity_config.yaml  Weights and hyperparameters
```

---

## Release Asset

The precomputed matrix (`usda_similarity_matrix.npz`) is available as a GitHub release asset and does not need to be rebuilt to reproduce paper results. Load it with:

```python
import numpy as np
data = np.load("usda_similarity_matrix.npz")
S = data["similarity"]      # shape (7338, 7338), float32
item_ids = data["item_ids"] # USDA FDC item IDs
```

---

## Rebuild from Scratch

```bash
# 1. Download USDA FoodData Central (see data/README.md)
# 2. Build embeddings and FAISS index
python similarity/build_embedding_index.py --config similarity/configs/similarity_config.yaml
# 3. Compute component scores
python similarity/compute_macro_scores.py
python similarity/compute_category_scores.py
# 4. Rerank top-K neighbors
python similarity/rerank.py
# 5. Combine into final matrix
python similarity/build_similarity_matrix.py --output usda_similarity_matrix.npz
```
