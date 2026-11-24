"""Project-wide paths and constants.

Everything that a later analysis step depends on (windows, thresholds, seeds)
lives here so the pre-registration can point to one place.
"""
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"
IBL_DATA = DATA / "ibl"
IBL_AGG = IBL_DATA / "aggregates"
IBL_SUBJECT_TRIALS = IBL_DATA / "subject_trials"
IBL_TRIALS = IBL_DATA / "trials"
FEATURE_CACHE = DATA / "features"          # per-session cached features (spike counts, covariates)
RAW_SCRATCH = DATA / "raw_scratch"          # raw downloads; deleted after feature extraction
STEINMETZ_DATA = DATA / "steinmetz"
RESULTS = ROOT / "results"
FIGURES = RESULTS / "figures"

# Open data access (no account needed): IBL public AWS bucket
IBL_BUCKET = "ibl-brain-wide-map-public"
IBL_BUCKET_URL = f"https://{IBL_BUCKET}.s3.amazonaws.com"
BWM_AGG_PREFIX = "aggregates/2026_Q2_IBL_et_al_BWM"   # trials.pqt identical (md5) to 2024_Q2 copy

# ---- GLM-HMM (behaviour only) -------------------------------------------
GLMHMM_K_RANGE = (1, 2, 3, 4, 5)       # K=1 is a plain Bernoulli GLM
GLMHMM_N_INITS_GLOBAL = 20
GLMHMM_N_FOLDS = 5                     # session-level CV folds
GLMHMM_PRIOR_SIGMA = 2.0               # Gaussian prior s.d. on GLM weights (Ashwood 2022)
GLMHMM_DIRICHLET_ALPHA = 2.0           # Dirichlet concentration on transition rows (Ashwood 2022)
GLMHMM_TRANSITION_STICKY_INIT = 0.95
STATE_POSTERIOR_THRESHOLD = 0.8        # label a trial only if p(state) > 0.8
UNBIASED_TRIALS = 90                   # first 90 trials of biased/ephys sessions are 50/50
BLOCK_HAZARD = 1.0 / 60.0              # constant-hazard approximation for the Bayes observer

# ---- Session inclusion (behaviour only; fixed before neural data) -------
MIN_TRIALS_SESSION = 400
MIN_LABELLED_PER_CLASS = 30
MIN_GOOD_UNITS_REGION = 10

# ---- Neural windows (relative to stimOn_times, seconds) -----------------
PRESTIM_WINDOW = (-0.4, -0.1)          # primary
PRESTIM_WINDOW_ALT = (-0.6, -0.1)      # secondary (Findling et al. 2025 window)

# ---- Decoding -----------------------------------------------------------
N_OUTER_FOLDS = 5
N_INNER_FOLDS = 3
C_GRID = (1e-4, 1e-3, 1e-2, 1e-1, 1.0, 10.0)
N_NULL = 200                           # null sessions per real session
FDR_Q = 0.05

SEED = 20261124
