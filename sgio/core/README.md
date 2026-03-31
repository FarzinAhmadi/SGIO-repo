# SGIO Core — Optimization Formulation

This module implements the SGIO optimization formulation: the augmented item space construction, similarity-discounted objective, MGIL tradeoff path, item-level swap extraction, and multi-observation extension.

## Files to add here

```
sgio/core/
├── README.md                   (this file)
├── augmented_space.py          Build F_aug = F_obs ∪ top-K neighbors per consumed item
├── similarity_weights.py       Construct W_S weight matrix (1 - S_{f_i(j), j} per neighbor)
├── sgio_formulation.py         Main SGIO optimization (MGIL + similarity-discounted objective)
├── sgio_multi_obs.py           Multi-observation SGIO (shared binary variables across K days)
├── swap_extraction.py          Extract item-level swap sequences from tradeoff path {z_ℓ}
├── dash_constraints.py         DASH feasibility constraint builder (Az ≤ b)
└── utils.py                    Shared helpers (nutrient vectors, cost computation)
```

## Key formulation reference

See Section IV of the paper (`paper/main.tex`) for the full mathematical formulation.

- **Augmented item space:** $\mathcal{F}_\text{aug} = \mathcal{F}_\text{obs} \cup \bigcup_i \mathcal{N}_K(f_i)$
- **Similarity-discounted objective:** weight matrix $W_S$ where neighbor $j$ of item $i$ has weight $1 - S_{f_i, j}$
- **SGIO formulation:** MGIL with $W_S$ and $\mathcal{F}_\text{aug}$ as decision space
- **Multi-observation:** shared binary indicator variables $v$ across $K$ days
