# config_lstm.py — Configuration for LSTM Version 2 (The 0.9029 Anchor)

from pathlib import Path

# ── Paths ─────────────────────────────────────────────────────────────────────
DATA_DIR        = Path("data")
TRAIN_PATH      = DATA_DIR / "train.csv"
TEST_PATH       = DATA_DIR / "test.csv"
SAMPLE_SUB_PATH = "sample_submission.csv"

MODEL_DIR      = Path("models_lstm")
MODEL_DIR.mkdir(exist_ok=True)
SUBMISSION_DIR = Path("submissions")
SUBMISSION_DIR.mkdir(exist_ok=True)

# ── Sequence configuration ────────────────────────────────────────────────────
SEQ_LEN  = 13   # 13 weeks of input weather context
PRED_LEN = 5    # 5 weeks of output score prediction

# ── LSTM Hyperparameters ──────────────────────────────────────────────────────
BATCH_SIZE    = 512
EPOCHS        = 50           
LEARNING_RATE = 1e-3
HIDDEN_SIZE   = 128
NUM_LAYERS    = 2
EMBED_DIM     = 16           # region embedding dimension

# ── Validation settings ───────────────────────────────────────────────────────
VAL_REGION_COUNT       = 200   
EARLY_STOPPING_PATIENCE = 5    
SEED                   = 42

# ── Meteorological features (V2 STRICT) ───────────────────────────────────────
FEATURES = [
    # The ultra-stable EDA features
    "surf_pre_mean", 
    "tmp_range_mean", 
    "wind_mean", 
    "wind_max_max",
    "wind_range_mean",
    
    # Mildly shifted
    "prec_sum", 
    "humidity_mean",
    
    # Simple Shift-Immune Dynamics (NO EMA, NO 2nd-Order)
    "delta_prec",
    "delta_tmp_range",
    "month_sin", 
    "month_cos"
]

# ── Loss weighting ────────────────────────────────────────────────────────────
SUMMER_WEIGHT_MULTIPLIER = 3.0   # 3x loss weight for summer prediction targets

# ── Prediction post-processing ────────────────────────────────────────────────
THRESHOLD_SQUEEZER = 0.4   # predictions below this are forced to 0.0