"""Gurobi environment setup and shared constants."""

import os

import gurobipy as gp


def make_gurobi_env() -> gp.Env:
    """Create a Gurobi environment from GRB_* environment variables.

    Expected env vars: GRB_WLSACCESSID, GRB_WLSSECRET, GRB_LICENSEID.
    Falls back to a default (local license) env if none are set.
    """
    env = gp.Env(empty=True)
    access_id = os.environ.get("GRB_WLSACCESSID")
    if access_id:
        env.setParam("WLSACCESSID", access_id)
        env.setParam("WLSSECRET", os.environ["GRB_WLSSECRET"])
        env.setParam("LICENSEID", int(os.environ["GRB_LICENSEID"]))
    env.start()
    return env
