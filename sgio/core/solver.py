"""S-MGIL optimisation solver."""

import numpy as np
import gurobipy as gp
from gurobipy import GRB

from smgil.config import make_gurobi_env


def _compute_bigM(A: np.ndarray, b: np.ndarray, z_upper: float = 5000.0) -> np.ndarray:
    """Compute per-constraint big-M values.

    M_j must be large enough that  A_j z >= b_j - M_j  is non-binding
    for all feasible z >= 0 with per-item intake at most z_upper grams.
    """
    return np.sum(np.abs(A), axis=1) * z_upper + np.abs(b) + 1.0


def recover_theta(
    A: np.ndarray,
    active: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Recover imputed preference vector theta and dual multipliers lambda.

    Given a known active constraint set, sets unit dual multipliers on
    active constraints (any positive lambda satisfies complementary
    slackness) and computes theta = A^T lambda.

    Parameters
    ----------
    A      : (m, n) constraint matrix
    active : (m,) binary — which constraints are active (tight)

    Returns
    -------
    theta  : (n,) imputed objective vector
    lam    : (m,) dual multipliers (lambda_i = 0 for inactive constraints)
    """
    m, _n = A.shape
    lam = np.zeros(m)
    active_ind = np.where(np.array(active) > 0.5)[0]
    if len(active_ind) > 0:
        lam[active_ind] = 1.0
    theta = A.T @ lam
    return theta, lam


def smgil(
    A: np.ndarray,
    b: np.ndarray,
    X: np.ndarray,
    W_S: np.ndarray,
    acceptable: np.ndarray,
    tight: np.ndarray,
    p: int,
    preferred: np.ndarray | None = None,
    method: str = "bigm",
    eps: float = 0.1,
) -> tuple[np.ndarray, np.ndarray, float]:
    """Similarity-Guided MGIL (S-MGIL).

    Uses a W_S-weighted L2 objective:
        min  sum_k sum_i  W_S[i] * (X[k,i] - z[i])^2

    Parameters
    ----------
    A           : (m, n) constraint matrix   Az <= b
    b           : (m,)   RHS
    X           : (K, n) observation matrix  (K=1 for single respondent)
    W_S         : (n,)   diagonal weight vector from build_observation_vector()
    acceptable  : (m,)   binary — which constraints are relevant
    tight       : (m,)   binary — which constraints must stay tight
                          (inherited from previous iteration)
    p           : int    total number of constraints to make tight this round
    preferred   : (m,)   optional binary — preferred constraints to tighten first
    method      : "bigm" (default) uses big-M linearization (convex MIQP);
                  "bilinear" uses the legacy bilinear formulation (requires
                  NonConvex=2)
    eps         : slack tolerance for non-selected constraints (big-M method
                  only); when v_i=0, constraint i must have at least eps slack

    Returns
    -------
    Z   : (n,) optimal item allocation
    C   : (|acceptable|,) binary vector of which constraints are tight
    obj : float weighted objective value
    """
    m, n = A.shape

    if preferred is None:
        preferred = np.zeros(m)

    x_bar = X.mean(axis=0)

    acceptable_ind = np.where(np.array(acceptable) == 1)[0]
    preferred_ind = np.where(np.array(preferred) == 1)[0]
    tight_ind = np.where(np.array(tight) == 1)[0]

    env = make_gurobi_env()
    model = gp.Model("S-MGIL", env=env)
    model.Params.OutputFlag = 0
    if method == "bilinear":
        model.Params.NonConvex = 2

    z = model.addVars(n, lb=0.0, name="z")
    v = model.addVars(len(acceptable_ind), vtype=GRB.BINARY, name="v")

    # Weighted quadratic objective: sum_i W_S[i] * (x_bar[i] - z[i])^2
    obj_expr = gp.quicksum(
        W_S[i] * (x_bar[i] - z[i]) * (x_bar[i] - z[i]) for i in range(n)
    )
    pref_bonus = 1000 * gp.quicksum(v[j] for j in preferred_ind)
    model.setObjective(obj_expr - pref_bonus, GRB.MINIMIZE)

    # Feasibility: Az <= b
    for j in range(m):
        model.addConstr(gp.quicksum(A[j, i] * z[i] for i in range(n)) <= b[j])

    # Tighten acceptable constraints via binary v
    if method == "bigm":
        bigM = _compute_bigM(A, b)
        for idx, j in enumerate(acceptable_ind):
            Aj_z = gp.quicksum(A[j, i] * z[i] for i in range(n))
            # v=1 → Aj_z >= b[j] (combined with feasibility Aj_z <= b[j] → equality)
            # v=0 → non-binding (Aj_z >= b[j] - M)
            model.addConstr(Aj_z >= b[j] - bigM[j] * (1 - v[idx]))
            # v=0 → Aj_z <= b[j] - eps (strict slack)
            # v=1 → Aj_z <= b[j] (just feasibility, already covered)
            model.addConstr(Aj_z <= b[j] - eps * (1 - v[idx]))
    else:
        for idx, j in enumerate(acceptable_ind):
            model.addConstr(
                v[idx] * gp.quicksum(A[j, i] * z[i] for i in range(n))
                == v[idx] * b[j]
            )

    # Total tight constraints this iteration = p
    model.addConstr(gp.quicksum(v[i] for i in range(len(acceptable_ind))) == p)

    # Inherit previously tight constraints
    for idx in tight_ind:
        model.addConstr(v[idx] == 1)

    model.optimize()

    n_acceptable = len(acceptable_ind)
    Z_out = np.zeros(n)
    C_out = np.zeros(n_acceptable)

    if model.status == GRB.OPTIMAL:
        for i in range(n):
            Z_out[i] = z[i].X
        for i in range(n_acceptable):
            C_out[i] = v[i].X
        obj_val = model.ObjVal + 1000 * sum(v[j].X for j in preferred_ind)
        model.dispose()
        return Z_out, C_out, obj_val

    print(f"  S-MGIL status: {model.status} (infeasible or unbounded)")
    model.dispose()
    return np.full(n, np.nan), np.full(n_acceptable, np.nan), np.nan


def smgil_multi_obs(
    A: np.ndarray,
    b: np.ndarray,
    X: np.ndarray,
    W_S: np.ndarray,
    acceptable: np.ndarray,
    tight: np.ndarray,
    p: int,
    preferred: np.ndarray | None = None,
    method: str = "bigm",
    eps: float = 0.1,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, float]:
    """Multi-observation S-MGIL with shared constraint selection.

    Finds K separate recommendations z_1,...,z_K (one per observation day),
    all sharing the same set of p tight constraints.  This is the IO
    interpretation: the shared active set identifies constraints that are
    consistently binding across a person's eating pattern.

    Parameters
    ----------
    A           : (m, n) constraint matrix   Az <= b
    b           : (m,)   RHS
    X           : (K, n) observation matrix — one row per day
    W_S         : (n,)   diagonal weight vector
    acceptable  : (m,)   binary — which constraints are relevant
    tight       : (m,)   binary — inherited tight constraints
    p           : int    total number of constraints to make tight
    preferred   : (m,)   optional binary — preferred constraints to tighten first
    method      : "bigm" (default) or "bilinear"
    eps         : slack tolerance for non-selected constraints (big-M only)

    Returns
    -------
    Z_all  : (K, n) per-day optimal allocations
    Z_mean : (n,)   mean recommendation across days
    C      : (|acceptable|,) binary tight constraint vector
    obj    : float  total weighted objective (sum over all K days)
    """
    m, n = A.shape
    K = X.shape[0]

    if preferred is None:
        preferred = np.zeros(m)

    acceptable_ind = np.where(np.array(acceptable) == 1)[0]
    preferred_ind = np.where(np.array(preferred) == 1)[0]
    tight_ind = np.where(np.array(tight) == 1)[0]

    env = make_gurobi_env()
    model = gp.Model("S-MGIL-MultiObs", env=env)
    model.Params.OutputFlag = 0
    if method == "bilinear":
        model.Params.NonConvex = 2

    # K x n continuous variables: z[k, i] = recommended intake for day k, item i
    z = model.addVars(K, n, lb=0.0, name="z")
    v = model.addVars(len(acceptable_ind), vtype=GRB.BINARY, name="v")

    # Objective: sum_k sum_i W_S[i] * (X[k,i] - z[k,i])^2
    obj_expr = gp.quicksum(
        W_S[i] * (X[k, i] - z[k, i]) * (X[k, i] - z[k, i])
        for k in range(K)
        for i in range(n)
    )
    pref_bonus = 1000 * gp.quicksum(v[j] for j in preferred_ind)
    model.setObjective(obj_expr - pref_bonus, GRB.MINIMIZE)

    # Compute big-M once (shared across all K)
    if method == "bigm":
        bigM = _compute_bigM(A, b)

    # Per-observation constraints
    for k in range(K):
        # Feasibility: A z_k <= b
        for j in range(m):
            model.addConstr(
                gp.quicksum(A[j, i] * z[k, i] for i in range(n)) <= b[j]
            )

        # Tighten acceptable constraints via shared binary v
        if method == "bigm":
            for idx, j in enumerate(acceptable_ind):
                Aj_z = gp.quicksum(A[j, i] * z[k, i] for i in range(n))
                model.addConstr(Aj_z >= b[j] - bigM[j] * (1 - v[idx]))
                model.addConstr(Aj_z <= b[j] - eps * (1 - v[idx]))
        else:
            for idx, j in enumerate(acceptable_ind):
                model.addConstr(
                    v[idx] * gp.quicksum(A[j, i] * z[k, i] for i in range(n))
                    == v[idx] * b[j]
                )

    # Total tight constraints = p (shared across all K observations)
    model.addConstr(gp.quicksum(v[i] for i in range(len(acceptable_ind))) == p)

    # Inherit previously tight constraints
    for idx in tight_ind:
        model.addConstr(v[idx] == 1)

    model.optimize()

    n_acceptable = len(acceptable_ind)
    Z_all = np.zeros((K, n))
    C_out = np.zeros(n_acceptable)

    if model.status == GRB.OPTIMAL:
        for k in range(K):
            for i in range(n):
                Z_all[k, i] = z[k, i].X
        for i in range(n_acceptable):
            C_out[i] = v[i].X
        obj_val = model.ObjVal + 1000 * sum(v[j].X for j in preferred_ind)
        Z_mean = Z_all.mean(axis=0)
        model.dispose()
        return Z_all, Z_mean, C_out, obj_val

    print(f"  S-MGIL-MultiObs status: {model.status} (infeasible or unbounded)")
    model.dispose()
    return (
        np.full((K, n), np.nan),
        np.full(n, np.nan),
        np.full(n_acceptable, np.nan),
        np.nan,
    )
