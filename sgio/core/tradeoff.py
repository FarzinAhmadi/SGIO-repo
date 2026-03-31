"""S-MGIL tradeoff path runner and weight tuning."""

import numpy as np

from smgil.solver import smgil, smgil_multi_obs


def tune_weights(
    W_S: np.ndarray,
    meta: dict,
    obs_penalty_scale: float = 2.0,
    swap_bonus: float = 4000.0,
    min_sim_threshold: float = 0.7,
) -> np.ndarray:
    """Tune the similarity weight vector to encourage food swaps.

    Parameters
    ----------
    W_S                : base weight vector from build_observation_vector()
    meta               : metadata dict from build_observation_vector()
    obs_penalty_scale  : multiply observed-item weights (penalises reductions)
    swap_bonus         : subtract from neighbor costs (makes swapping attractive)
    min_sim_threshold  : zero out neighbors below this similarity
    """
    W = W_S.copy()
    for item in meta["observed_items"]:
        W[item["index"]] *= obs_penalty_scale
    for item in meta["neighbor_items"]:
        idx, sim = item["index"], item["similarity_score"]
        if sim < min_sim_threshold:
            W[idx] = 1e6
        else:
            W[idx] = max(1e-6, W[idx] - swap_bonus)
    return W


def _build_name_map(meta: dict) -> dict:
    name_map = {}
    for item in meta["observed_items"]:
        name_map[item["food_code"]] = item.get("food_name", str(item["food_code"]))
    for item in meta["neighbor_items"]:
        name_map[item["food_code"]] = item.get("food_name", str(item["food_code"]))
    return name_map


def _extract_swaps(
    Z: np.ndarray,
    x_obs_1d: np.ndarray,
    W_S: np.ndarray,
    item_index: dict,
    meta: dict,
    name_map: dict,
) -> list[dict]:
    """Identify food changes between observed and recommended allocations."""
    idx_to_code = {v: k for k, v in item_index.items()}
    obs_codes = {item["food_code"] for item in meta["observed_items"]}
    n = len(Z)
    swaps = []

    for i in range(n):
        code = idx_to_code.get(i)
        z_val = Z[i]
        x_val = x_obs_1d[i]
        delta = z_val - x_val

        if abs(delta) < 0.01:
            continue

        is_obs = code in obs_codes
        action = "reduce" if delta < 0 else ("increase" if is_obs else "ADD_NEW")
        swaps.append(
            {
                "food_code": code,
                "food_name": name_map.get(code, str(code)),
                "is_observed": is_obs,
                "observed_qty": round(x_val, 1),
                "recommended_qty": round(z_val, 1),
                "delta": round(delta, 1),
                "action": action,
                "W_S": round(float(W_S[i]), 3),
            }
        )

    return sorted(swaps, key=lambda s: abs(s["delta"]), reverse=True)


def run_smgil_tradeoff(
    A: np.ndarray,
    b: np.ndarray,
    X_obs: np.ndarray,
    W_S: np.ndarray,
    item_index: dict,
    meta: dict,
    constraint_names: list[str],
    max_iterations: int = 5,
    cost_threshold: float | None = None,
    verbose: bool = True,
    method: str = "bigm",
    eps: float = 0.1,
) -> list[dict]:
    """Run the full S-MGIL tradeoff path for a single respondent.

    Parameters
    ----------
    method : "bigm" (default) or "bilinear" — passed to smgil() solver.
    eps    : slack tolerance for big-M method — passed to smgil() solver.

    Returns a list of dicts per iteration with keys:
        iteration, z, tight_constraints, weighted_distance,
        marginal_cost, swaps
    """
    m = A.shape[0]
    acceptable = np.ones(m)
    preferred = np.zeros(m)
    name_map = _build_name_map(meta)
    x_obs_1d = X_obs[0]

    results = []
    tight = np.zeros(m)
    prev_dist = 0.0

    for ell in range(1, max_iterations + 1):
        Z, C, dist = smgil(
            A, b, X_obs, W_S, acceptable, tight, p=ell,
            preferred=preferred, method=method, eps=eps,
        )

        if np.isnan(dist):
            if verbose:
                print(f"  Iteration {ell}: infeasible -- stopping.")
            break

        marginal = dist - prev_dist
        if cost_threshold is not None and marginal > cost_threshold:
            if verbose:
                print(
                    f"  Iteration {ell}: marginal cost {marginal:.3f} > "
                    f"threshold {cost_threshold} -- stopping."
                )
            break

        acceptable_ind = np.where(acceptable == 1)[0]
        tight_constraints = [
            constraint_names[acceptable_ind[i]]
            for i in range(len(C))
            if C[i] > 0.5
        ]

        swaps = _extract_swaps(Z, x_obs_1d, W_S, item_index, meta, name_map)

        results.append(
            {
                "iteration": ell,
                "z": Z,
                "tight_constraints": tight_constraints,
                "weighted_distance": round(dist, 4),
                "marginal_cost": round(marginal, 4),
                "swaps": swaps,
            }
        )

        tight = C.copy()
        prev_dist = dist

        if verbose:
            print(f"\n-- Iteration {ell} {'─' * 40}")
            print(f"  Tight constraints : {tight_constraints}")
            print(f"  Weighted distance : {dist:.4f}  (marginal: +{marginal:.4f})")
            print("  Food swaps:")
            for s in swaps[:6]:
                direction = "▼" if s["delta"] < 0 else "▲"
                tag = "" if s["is_observed"] else "  <- NEW ITEM"
                print(
                    f"    {direction} {s['food_name'][:45]:<45}  "
                    f"{s['observed_qty']:6.1f} -> {s['recommended_qty']:6.1f} g"
                    f"  W={s['W_S']:.3f}{tag}"
                )

    return results


def run_smgil_tradeoff_multi_obs(
    A: np.ndarray,
    b: np.ndarray,
    X_daily: np.ndarray,
    W_S: np.ndarray,
    item_index: dict,
    meta: dict,
    constraint_names: list[str],
    max_iterations: int = 5,
    cost_threshold: float | None = None,
    verbose: bool = True,
    method: str = "bigm",
    eps: float = 0.1,
) -> list[dict]:
    """Run the multi-observation S-MGIL tradeoff path.

    Each observation day gets its own recommendation, but the set of
    tight constraints is shared across all days.

    Parameters
    ----------
    X_daily : (K, n) observation matrix — one row per training day.
    Other parameters are the same as run_smgil_tradeoff().

    Returns a list of dicts per iteration with keys:
        iteration, z_mean, z_all, tight_constraints, weighted_distance,
        marginal_cost, swaps
    """
    m = A.shape[0]
    K = X_daily.shape[0]
    acceptable = np.ones(m)
    preferred = np.zeros(m)
    name_map = _build_name_map(meta)
    x_bar = X_daily.mean(axis=0)

    results = []
    tight = np.zeros(m)
    prev_dist = 0.0

    for ell in range(1, max_iterations + 1):
        Z_all, Z_mean, C, dist = smgil_multi_obs(
            A, b, X_daily, W_S, acceptable, tight, p=ell,
            preferred=preferred, method=method, eps=eps,
        )

        if np.isnan(dist):
            if verbose:
                print(f"  Iteration {ell}: infeasible -- stopping.")
            break

        # Normalize objective by K for comparability with single-obs
        dist_norm = dist / K
        marginal = dist_norm - prev_dist

        if cost_threshold is not None and marginal > cost_threshold:
            if verbose:
                print(
                    f"  Iteration {ell}: marginal cost {marginal:.3f} > "
                    f"threshold {cost_threshold} -- stopping."
                )
            break

        acceptable_ind = np.where(acceptable == 1)[0]
        tight_constraints = [
            constraint_names[acceptable_ind[i]]
            for i in range(len(C))
            if C[i] > 0.5
        ]

        swaps = _extract_swaps(Z_mean, x_bar, W_S, item_index, meta, name_map)

        results.append(
            {
                "iteration": ell,
                "z": Z_mean,         # consensus recommendation
                "z_all": Z_all,      # per-day recommendations (K, n)
                "tight_constraints": tight_constraints,
                "weighted_distance": round(dist_norm, 4),
                "marginal_cost": round(marginal, 4),
                "swaps": swaps,
            }
        )

        tight = C.copy()
        prev_dist = dist_norm

        if verbose:
            print(f"\n-- Iteration {ell} (multi-obs, K={K}) {'─' * 30}")
            print(f"  Tight constraints : {tight_constraints}")
            print(f"  Weighted distance : {dist_norm:.4f}  (marginal: +{marginal:.4f})")
            # Per-day distance spread
            per_day_dists = np.array([
                np.sqrt(np.dot(W_S * (X_daily[k] - Z_all[k]),
                               X_daily[k] - Z_all[k]))
                for k in range(K)
            ])
            print(f"  Per-day dist range: [{per_day_dists.min():.3f}, "
                  f"{per_day_dists.max():.3f}]  mean={per_day_dists.mean():.3f}")
            print("  Food swaps (mean recommendation):")
            for s in swaps[:6]:
                direction = "▼" if s["delta"] < 0 else "▲"
                tag = "" if s["is_observed"] else "  <- NEW ITEM"
                print(
                    f"    {direction} {s['food_name'][:45]:<45}  "
                    f"{s['observed_qty']:6.1f} -> {s['recommended_qty']:6.1f}"
                    f"  W={s['W_S']:.3f}{tag}"
                )

    return results
