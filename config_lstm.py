from pathlib import Path

DATA_DIR = Path("data")
TRAIN_PATH = DATA_DIR / "train.csv"
TEST_PATH = DATA_DIR / "test.csv"
SAMPLE_SUB_PATH = "sample_submission.csv"

MODEL_DIR = Path("models_lstm")
MODEL_DIR.mkdir(exist_ok=True)
SUBMISSION_DIR  = Path("submissions")

# Sequence configuration
SEQ_LEN = 13       # 13 weeks of input weather context
PRED_LEN = 5       # 5 weeks of output score prediction

# LSTM Hyperparameters
BATCH_SIZE = 512
EPOCHS = 30
LEARNING_RATE = 1e-3
HIDDEN_SIZE = 128
NUM_LAYERS = 2
EMBED_DIM = 16     # Dimension for region embeddings

# Meteorological features to keep
FEATURES = [
    "prec_sum", "surf_pre_mean", "humidity_mean", "tmp_mean", 
    "tmp_range_mean", "tmp_max_max", "surf_tmp_mean", 
    "wind_mean", "wind_max_max"
]