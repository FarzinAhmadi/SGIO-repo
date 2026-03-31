"""S-MGIL: Similarity-Guided Modified Generalized Inverse Learning."""

from smgil.config import make_gurobi_env
from smgil.constraints import DASH_CONSTRAINTS, build_A_b, check_observed_intake
from smgil.constraints import MFP_DASH_CONSTRAINTS, build_A_b_mfp
from smgil.constraints import A4F_DASH_CONSTRAINTS, build_A_b_a4f
from smgil.preprocessing import build_crosswalk, build_observation_vector
from smgil.mfp_preprocessing import build_mfp_crosswalk, build_mfp_observation_vector, build_mfp_daily_matrix
from smgil.a4f_preprocessing import build_a4f_crosswalk, build_a4f_observation_vector, build_a4f_daily_matrix
from smgil.solver import smgil, smgil_multi_obs, recover_theta
from smgil.tradeoff import run_smgil_tradeoff, run_smgil_tradeoff_multi_obs, tune_weights
