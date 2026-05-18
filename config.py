# config.py — All paths, constants, and hyperparameters in one place

from pathlib import Path

# ── Paths ─────────────────────────────────────────────────────────────────────
DATA_DIR        = Path("data")
TRAIN_PATH      = DATA_DIR / "train.csv"
TEST_PATH       = DATA_DIR / "test.csv"
SAMPLE_SUB_PATH = "sample_submission.csv"

MODEL_DIR       = Path("models")
SUBMISSION_DIR  = Path("submissions")

MODEL_DIR.mkdir(exist_ok=True)
SUBMISSION_DIR.mkdir(exist_ok=True)

# ── Column definitions ────────────────────────────────────────────────────────
METEO_COLS = [
    "prec", "surf_pre", "humidity", "tmp", "dp_tmp", "wb_tmp",
    "tmp_max", "tmp_min", "tmp_range", "surf_tmp",
    "wind", "wind_max", "wind_min", "wind_range",
]

# Columns to drop — year is fictional, day_of_week has no signal (section 17 EDA)
# Note: year, month, day, jdn, day_of_week are added by extract_date_parts() at runtime
# They are excluded in get_feature_cols() in feature_engineering.py
DROP_COLS = ["date", "year", "day_of_week", "jdn"]

TARGET_COL  = "score"
REGION_COL  = "region_id"

# ── Feature engineering settings ──────────────────────────────────────────────
# Number of lag score weeks to include
LAG_WEEKS = [1, 2, 3, 4]

# Rolling window sizes (in weeks)
ROLLING_WINDOWS = [4, 13]

# Features to compute rolling means on
ROLLING_FEATURES = ["prec_sum", "tmp_range_mean", "surf_pre_mean", "humidity_mean"]

# Weekly aggregation functions per raw feature
# Format: output_col_name -> (source_col, agg_function)
WEEKLY_AGG = {
    "prec_sum"       : ("prec",      "sum"),
    "prec_mean"      : ("prec",      "mean"),
    "tmp_mean"       : ("tmp",       "mean"),
    "tmp_range_mean" : ("tmp_range", "mean"),
    "tmp_max_max"    : ("tmp_max",   "max"),
    "tmp_min_min"    : ("tmp_min",   "min"),
    "humidity_mean"  : ("humidity",  "mean"),
    "dp_tmp_mean"    : ("dp_tmp",    "mean"),
    "surf_pre_mean"  : ("surf_pre",  "mean"),
    "surf_tmp_mean"  : ("surf_tmp",  "mean"),
    "wind_mean"      : ("wind",      "mean"),
    "wind_max_max"   : ("wind_max",  "max"),
    "wind_range_mean": ("wind_range","mean"),
}

# Prediction targets — 5 weeks ahead
PRED_WEEKS = 5
PRED_COLS  = [f"pred_week{i}" for i in range(1, PRED_WEEKS + 1)]

# ── LightGBM hyperparameters ──────────────────────────────────────────────────
LGBM_PARAMS = {
    "objective"       : "regression_l1",  # MAE loss — matches evaluation metric
    "metric"          : "mae",
    "learning_rate"   : 0.05,
    "num_leaves"      : 127,
    "max_depth"       : -1,
    "min_child_samples": 50,
    "feature_fraction": 0.8,
    "bagging_fraction": 0.8,
    "bagging_freq"    : 1,
    "lambda_l1"       : 0.1,
    "lambda_l2"       : 0.1,
    "n_estimators"    : 2000,
    "random_state"    : 42,
    "n_jobs"          : -1,
    "verbose"         : -1,
}

LGBM_FIT_PARAMS = {
    "eval_metric"          : "mae",
    "callbacks"            : None,   # set in train.py (early stopping + logging)
}

EARLY_STOPPING_ROUNDS = 100
LOG_EVERY_N_ROUNDS    = 100

# ── Validation settings ───────────────────────────────────────────────────────
# Number of regions held out for validation (region-level split)
VAL_REGION_COUNT = 200

# Random seed
SEED = 42