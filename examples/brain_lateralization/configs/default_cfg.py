"""
Default configurations
"""

from yacs.config import CfgNode as CN

# -----------------------------------------------------------------------------
# Config definition
# -----------------------------------------------------------------------------

_C = CN()

# -----------------------------------------------------------------------------
# Dataset
# -----------------------------------------------------------------------------
_C.DATASET = CN()
_C.DATASET.DATASET = "HCP"
_C.DATASET.DOWNLOAD = True
_C.DATASET.ROOT = "data"
_C.DATASET.ATLAS = "BNA"
_C.DATASET.FEATURE = "correlation"
_C.DATASET.TEST_RATIO = 0.0
_C.DATASET.TYPE = "functional"
_C.DATASET.SESSIONS = [None]
_C.DATASET.RUN = "Fisherz"
_C.DATASET.CONNECTION = "intra"
_C.DATASET.NUM_REPEAT = 1
_C.DATASET.MIX_GROUP = False
_C.DATASET.REST1_ONLY = False  # only use rest1 data for training, HCP dataset only
# ---------------------------------------------------------------------------- #
# Solver
# ---------------------------------------------------------------------------- #
_C.SOLVER = CN()
_C.SOLVER.SEED = 2023
_C.SOLVER.L2_HPARAM = 0.1  # hyperparameter for the l2 regularization
_C.SOLVER.LR = 0.04  # Initial learning rate
_C.SOLVER.OPTIMIZER = "lbfgs"
_C.SOLVER.MAX_ITER = 100
_C.SOLVER.LAMBDA_ = [0.0, 1.0, 2.0, 5.0, 8.0, 10.0]

# ---------------------------------------------------------------------------- #
# Misc options
# ---------------------------------------------------------------------------- #
_C.OUTPUT = CN()
_C.OUTPUT.ROOT = "output"


def get_cfg_defaults():
    return _C.clone()
