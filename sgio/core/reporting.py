"""Diet reporting: baseline summaries, constraint checks, and result tables."""

import numpy as np
import pandas as pd


SERVING_SIZE_G_DEFAULT = 100.0


def build_name_map(meta: dict) -> dict[str, str]:
    """Build a food_code -> food_name lookup from observation metadata."""
    name_map = {}
    for item in meta.get("observed_items", []) + meta.get("neighbor_items", []):
        name_map[str(item.get("food_code"))] = item.get(
            "food_name", str(item.get("food_code"))
        )
    return name_map


def baseline_table(
    x_vector: np.ndarray,
    item_index: dict,
    meta: dict,
) -> pd.DataFrame:
    """Participant baseline: food items with observed quantities."""
    name_map = build_name_map(meta)
    rows = []
    for food_code, col_idx in item_index.items():
        qty_g = x_vector[col_idx]
        if qty_g > 1e-6:
            rows.append(
                {
                    "Food Code": food_code,
                    "Food Name": name_map.get(str(food_code), str(food_code)),
                    "Observed (g)": round(qty_g, 1),
                    "Servings": round(qty_g / SERVING_SIZE_G_DEFAULT, 2),
                }
            )
    return pd.DataFrame(rows).sort_values("Observed (g)", ascending=False)


def dash_constraint_table(
    A: np.ndarray,
    b: np.ndarray,
    x_vector: np.ndarray,
    constraint_names: list[str],
    directions: list[str],
) -> pd.DataFrame:
    """Compare observed nutrient intake against DASH bounds."""
    raw = A @ x_vector
    rows = []
    for i, cname in enumerate(constraint_names):
        is_lower = directions[i] == "lower"
        if is_lower:
            actual = -raw[i]
            bound = -b[i]
            ok = actual >= bound
        else:
            actual = raw[i]
            bound = b[i]
            ok = actual <= bound
        rows.append(
            {
                "Constraint": cname,
                "Direction": ">=" if is_lower else "<=",
                "Bound": round(bound, 2),
                "Observed": round(actual, 2),
                "Baseline OK?": "Y" if ok else "N",
            }
        )
    return pd.DataFrame(rows)


def food_allocation_table(
    x_vector: np.ndarray,
    item_index: dict,
    meta: dict,
    tradeoff_path: list[dict],
) -> pd.DataFrame:
    """Food-space table: baseline vs recommended grams per iteration."""
    name_map = build_name_map(meta)
    rows = []
    for fc in sorted(item_index.keys()):
        col_idx = item_index[fc]
        baseline_g = float(x_vector[col_idx])
        row = {
            "Food Code": fc,
            "Food Name": name_map.get(str(fc), str(fc))[:40],
            "Baseline (g)": round(baseline_g, 1),
        }
        for r in tradeoff_path:
            row[f"r={r['iteration']} (g)"] = round(float(r["z"][col_idx]), 1)
        rows.append(row)

    df = pd.DataFrame(rows)
    qty_cols = [c for c in df.columns if c.endswith("(g)")]
    return df[df[qty_cols].max(axis=1) > 0].reset_index(drop=True)


def nutrient_intake_table(
    A: np.ndarray,
    b: np.ndarray,
    x_vector: np.ndarray,
    constraint_names: list[str],
    directions: list[str],
    tradeoff_path: list[dict],
) -> pd.DataFrame:
    """Nutrient-space table: baseline vs recommended intake per iteration."""
    raw = A @ x_vector
    rows = []
    for i, cname in enumerate(constraint_names):
        is_lower = directions[i] == "lower"
        sign = -1.0 if is_lower else 1.0
        bound = -b[i] if is_lower else b[i]
        row = {
            "Constraint": cname,
            "Direction": ">=" if is_lower else "<=",
            "Bound": round(bound, 2),
            "Baseline": round(sign * raw[i], 2),
        }
        for r in tradeoff_path:
            row[f"r={r['iteration']}"] = round(sign * float(A[i] @ r["z"]), 2)
        rows.append(row)
    return pd.DataFrame(rows)


def constraint_satisfaction_table(
    A: np.ndarray,
    b: np.ndarray,
    x_vector: np.ndarray,
    constraint_names: list[str],
    directions: list[str],
    tradeoff_path: list[dict],
) -> pd.DataFrame:
    """Binary satisfaction matrix: Y = within DASH bound, N = violated."""
    _TOL = 1e-6
    raw = A @ x_vector
    rows = []
    for i, cname in enumerate(constraint_names):
        is_lower = directions[i] == "lower"
        if is_lower:
            bound = -b[i]
            obs_ok = -raw[i] >= bound - _TOL
        else:
            bound = b[i]
            obs_ok = raw[i] <= bound + _TOL
        row = {"Constraint": cname, "Baseline": "Y" if obs_ok else "N"}
        for r in tradeoff_path:
            val = float(A[i] @ r["z"])
            if is_lower:
                ok = -val >= bound - _TOL
            else:
                ok = val <= bound + _TOL
            row[f"r={r['iteration']}"] = "Y" if ok else "N"
        rows.append(row)
    return pd.DataFrame(rows)


def summary_table(tradeoff_path: list[dict]) -> pd.DataFrame:
    """Flat swap-level summary table across all iterations."""
    rows = []
    for r in tradeoff_path:
        for s in r["swaps"]:
            rows.append(
                {
                    "Iteration": r["iteration"],
                    "Tight Constraints": ", ".join(r["tight_constraints"]),
                    "dD (marginal)": r["marginal_cost"],
                    "Food Code": s["food_code"],
                    "Food Name": s["food_name"],
                    "Observed (g)": s["observed_qty"],
                    "Recommended (g)": s["recommended_qty"],
                    "Delta (g)": s["delta"],
                    "Action": s["action"],
                    "W_S (switch cost)": s["W_S"],
                    "New Item?": not s["is_observed"],
                }
            )
    return pd.DataFrame(rows)


def print_full_report(
    x_vector: np.ndarray,
    A: np.ndarray,
    b: np.ndarray,
    item_index: dict,
    meta: dict,
    constraint_names: list[str],
    directions: list[str],
    tradeoff_path: list[dict],
    respondent_id: int,
):
    """Print the complete diet report to stdout."""
    pd.set_option("display.max_colwidth", 55)
    pd.set_option("display.float_format", "{:.2f}".format)
    pd.set_option("display.max_rows", 200)

    # Baseline
    print("=" * 70)
    print(f"  PARTICIPANT BASELINE  |  Respondent ID: {respondent_id}")
    print("=" * 70)
    df_bl = baseline_table(x_vector, item_index, meta)
    print(df_bl.to_string(index=False))
    print(f"\n  Total items consumed : {len(df_bl)}")
    print(f"  Total intake (g)     : {df_bl['Observed (g)'].sum():.1f} g\n")

    # DASH constraints
    print("=" * 70)
    print("  DASH CONSTRAINTS vs BASELINE")
    print("=" * 70)
    print(dash_constraint_table(A, b, x_vector, constraint_names, directions).to_string(index=False))
    print()

    # Food allocation
    print("=" * 70)
    print("  RECOMMENDED DIETS -- Food Allocation (grams) per Iteration")
    print("=" * 70)
    print(food_allocation_table(x_vector, item_index, meta, tradeoff_path).to_string(index=False))
    print()

    # Nutrient intake
    print("=" * 70)
    print("  RECOMMENDED DIETS -- Nutrient Intake per Iteration")
    print("=" * 70)
    print(nutrient_intake_table(A, b, x_vector, constraint_names, directions, tradeoff_path).to_string(index=False))
    print()

    # Constraint satisfaction
    print("=" * 70)
    print("  CONSTRAINT SATISFACTION  (Y = within DASH bound, N = violated)")
    print("=" * 70)
    df_sat = constraint_satisfaction_table(A, b, x_vector, constraint_names, directions, tradeoff_path)
    print(df_sat.to_string(index=False))
    print()
    print("  Constraints satisfied per stage:")
    for col in df_sat.columns[1:]:
        n_ok = (df_sat[col] == "Y").sum()
        print(f"    {col:>12s}: {n_ok:2d} / {len(df_sat)}")
    print()

    # Objective summary
    print("=" * 70)
    print("  S-MGIL WEIGHTED OBJECTIVE PER ITERATION")
    print("=" * 70)
    print(f"  {'Stage':<12s}  {'Weighted Dist':>15s}  {'Marginal Cost':>15s}")
    print(f"  {'-' * 12}  {'-' * 15}  {'-' * 15}")
    print(f"  {'Baseline':<12s}  {'0.0000':>15s}  {'---':>15s}")
    for r in tradeoff_path:
        print(
            f"  r={r['iteration']:<10d}  {r['weighted_distance']:>15.4f}"
            f"  {r['marginal_cost']:>15.4f}"
        )
