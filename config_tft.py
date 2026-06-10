from pathlib import Path

DATA_DIR = Path("data")
TRAIN_PATH = DATA_DIR / "train.csv"
TEST_PATH = DATA_DIR / "test.csv"
SAMPLE_SUB_PATH = Path("sample_submission.csv")

ARTIFACT_DIR = Path("artifacts_tft")
ARTIFACT_DIR.mkdir(exist_ok=True)
SUBMISSION_DIR = Path("submissions")
SUBMISSION_DIR.mkdir(exist_ok=True)
CHECKPOINT_PATH = ARTIFACT_DIR / "tft_best.ckpt"
COMBINED_FRAME_CACHE_STEM = ARTIFACT_DIR / "combined_frame"

REGION_COL = "region_id"
TARGET_COL = "score"

METEO_COLS_DAILY = [
    "prec", "surf_pre", "humidity", "tmp", "dp_tmp", "wb_tmp",
    "tmp_max", "tmp_min", "tmp_range", "surf_tmp", "wind", "wind_max", "wind_min", "wind_range"
]

# ADDED: Raw temps and the 4 new Anomaly Z-Scores
WEEKLY_FEATURES = [
    "surf_pre_mean", "tmp_range_mean", "wind_mean", "wind_max_max",
    "prec_sum", "humidity_mean",
    "delta_prec", "delta_tmp_range",
    "tmp_mean", "tmp_max_max", "tmp_min_min",
    "prec_sum_anomaly", "tmp_mean_anomaly", "tmp_max_max_anomaly", "tmp_min_min_anomaly"
]

# ADDED: Mean, Max, and Min aggregations for temperature
WEEKLY_AGG_SPEC = {
    "prec_sum":        ("prec",       "sum"),
    "surf_pre_mean":   ("surf_pre",   "mean"),
    "humidity_mean":   ("humidity",   "mean"),
    "tmp_range_mean":  ("tmp_range",  "mean"),
    "wind_mean":       ("wind",       "mean"),
    "wind_max_max":    ("wind_max",   "max"),
    "tmp_mean":        ("tmp",        "mean"),
    "tmp_max_max":     ("tmp_max",    "max"),
    "tmp_min_min":     ("tmp_min",    "min"),
}

ENC_LEN = 26 
DEC_LEN = 5

HIDDEN_SIZE = 64 
LSTM_LAYERS = 2
ATTENTION_HEAD_SIZE = 4
DROPOUT = 0.15 
HIDDEN_CONTINUOUS_SIZE = 32

QUANTILES = [0.02, 0.1, 0.25, 0.5, 0.75, 0.9, 0.98]
MEDIAN_QUANTILE_IDX = 3  # Median Quantile

BATCH_SIZE = 256 
NUM_WORKERS = 4 
MAX_EPOCHS = 15 
LEARNING_RATE = 1e-3
GRADIENT_CLIP_VAL = 0.1
EARLY_STOPPING_PATIENCE = 8

TRAIN_WINDOW_STRIDE = 26 
VAL_WEEKS_PER_REGION = 60

ACCELERATOR = "auto"
DEVICES = 1
PRECISION = "16-mixed" 
PATCH_ATTENTION_FOR_FP16 = True 

# SEED TO CHANGE (42 -> 123 -> 999)
SEED = 42