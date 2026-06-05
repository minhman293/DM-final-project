"""
config_tft.py — Hyperparameters and paths for the TFT drought-severity pipeline.

All paths assume the working directory has:
    data/train.csv
    data/test.csv
    sample_submission.csv

Edit DATA_DIR if your layout differs.
"""

from pathlib import Path

# ── Paths ─────────────────────────────────────────────────────────────────────
DATA_DIR = Path("data")  # set to "data" for real Kaggle environment, "data_local" for local validation
TRAIN_PATH = DATA_DIR / "train.csv"
TEST_PATH = DATA_DIR / "test.csv"
SAMPLE_SUB_PATH = Path("sample_submission.csv")

ARTIFACT_DIR = Path("artifacts_tft")
ARTIFACT_DIR.mkdir(exist_ok=True)
SUBMISSION_DIR = Path("submissions")
SUBMISSION_DIR.mkdir(exist_ok=True)

CHECKPOINT_PATH = ARTIFACT_DIR / "tft_best.ckpt"
# Cache extension is decided at runtime by data_prep (parquet if pyarrow is
# available, else pickle). We store just the stem; the extension is added there.
COMBINED_FRAME_CACHE_STEM = ARTIFACT_DIR / "combined_frame"

# ── Column definitions ───────────────────────────────────────────────────────
REGION_COL = "region_id"
TARGET_COL = "score"

# 14 meteorological features (raw daily columns in train/test)
METEO_COLS_DAILY = [
    "prec", "surf_pre", "humidity",
    "tmp", "dp_tmp", "wb_tmp",
    "tmp_max", "tmp_min", "tmp_range", "surf_tmp",
    "wind", "wind_max", "wind_min", "wind_range",
]

# Weekly-aggregated features fed into TFT as time-varying observed inputs.
# Removed absolute temperatures (tmp_mean, tmp_min) to prevent the Exploding Scaler trap.
WEEKLY_FEATURES = [
    # Stable EDA Features
    "surf_pre_mean", 
    "tmp_range_mean", 
    "wind_mean", 
    "wind_max_max",
    
    # Mildly Shifted
    "prec_sum", 
    "humidity_mean",
    
    # Shift-Immune Dynamics
    "delta_prec",
    "delta_tmp_range"
]

# Aggregation spec: out_col -> (src_col, agg_func)
# Aggregation spec: out_col -> (src_col, agg_func)
WEEKLY_AGG_SPEC = {
    "prec_sum":        ("prec",       "sum"),
    "surf_pre_mean":   ("surf_pre",   "mean"),
    "humidity_mean":   ("humidity",   "mean"),
    "tmp_range_mean":  ("tmp_range",  "mean"),
    "wind_mean":       ("wind",       "mean"),
    "wind_max_max":    ("wind_max",   "max"),
}

# ── Sequence configuration ───────────────────────────────────────────────────
# Encoder sees ENC_LEN weeks of history; decoder predicts DEC_LEN weeks ahead.
# 52 weeks of encoder = full annual cycle. The 91-day test window
# only provides 13 weeks of NEW context, so for test inference we
# splice 39 weeks of recent train history + 13 weeks of test data.
ENC_LEN = 26 # 6 months of history; tuned down from 52 to reduce training time and overfitting.
DEC_LEN = 5

# ── TFT hyperparameters ──────────────────────────────────────────────────────
HIDDEN_SIZE = 64           # main hidden dimension (sometimes called d_model), original: 64
LSTM_LAYERS = 2
ATTENTION_HEAD_SIZE = 4
DROPOUT = 0.15 # original: 0.15
HIDDEN_CONTINUOUS_SIZE = 32

# Quantiles: predict 7 quantiles around the median. q=0.5 minimizes MAE.
QUANTILES = [0.02, 0.1, 0.25, 0.5, 0.75, 0.9, 0.98]
MEDIAN_QUANTILE_IDX = 3   # index of 0.5 inside QUANTILES

# ── Training ─────────────────────────────────────────────────────────────────
BATCH_SIZE = 256 
NUM_WORKERS = 4            # set to 0 if you see DataLoader spawn issues on Windows
MAX_EPOCHS = 15 # # 30 is too long. With 50% truncation, it will converge faster.
LEARNING_RATE = 1e-3
GRADIENT_CLIP_VAL = 0.1
EARLY_STOPPING_PATIENCE = 8 # original: 5

# Subsampling for training-set construction. There are ~2,248 regions × ~782
# scored weeks = 1.76M weeks. Each region can spawn many overlapping
# (encoder, decoder) windows; we cap the stride to control epoch length.
TRAIN_WINDOW_STRIDE = 26    # take a new (encoder, decoder) window every 26 weeks (6 months) per region

# Validation: hold out the last VAL_WEEKS_PER_REGION scored weeks per region.
VAL_WEEKS_PER_REGION = 60  # ~14 months — enough to cover all 5-week horizons

# ── Hardware ─────────────────────────────────────────────────────────────────
ACCELERATOR = "auto"       # "gpu", "cpu", or "auto"
DEVICES = 1

# IMPORTANT: pytorch-forecasting's attention layer uses `-1e9` as the mask
# fill value, which overflows fp16's ~65,504 max. So with `PRECISION="16-mixed"`,
# training crashes with:
#   "RuntimeError: value cannot be converted to type at::Half without overflow"
#
# Two options:
#   1. PRECISION = "32-true"  → safe, ~1.5–2× slower.
#   2. PRECISION = "16-mixed" + PATCH_ATTENTION_FOR_FP16 = True
#      → patches pytorch-forecasting at runtime to use -1e4 instead of -1e9,
#      which is still effectively "negative infinity" after softmax but fits
#      in fp16. Train ~1.8× faster.
#
# Default: option 1 (safer). Switch to option 2 if your epoch time is too long.
PRECISION = "16-mixed"  # "32-true", "16-mixed", or "bf16-mixed" (if supported)
PATCH_ATTENTION_FOR_FP16 = True    # set True if you switch PRECISION to "16-mixed"

SEED = 42