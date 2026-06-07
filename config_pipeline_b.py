# config_pipeline_b.py
from pathlib import Path

# Paths
ARTIFACT_DIR = Path("artifacts_tft") # We will steal the TFT's perfectly prepped data!
MODEL_DIR = Path("models_pipeline_b")
MODEL_DIR.mkdir(exist_ok=True)
SUB_DIR = Path("submissions")
SUB_DIR.mkdir(exist_ok=True)

# Data Dimensions
HISTORY = 26       
HORIZONS = 5       
N_VARS = 10        # 8 Pristine Features + month_sin + month_cos
N_REGIONS = 2248   
N_WEEKS = HISTORY  

# Training Hyperparameters
BATCH_SIZE = 256
EPOCHS = 15
LEARNING_RATE = 1e-3
SEEDS = [42] # Just 1 seed for these models to save you time

# --- Model Specific Hyperparameters ---
AUTOGRU_HIDDEN = 128
AUTOGRU_REGION_DIM = 16
AUTOGRU_DROPOUT = 0.2

PATCH_D_MODEL = 64
PATCH_REGION_DIM = 16
PATCH_N_HEADS = 4
PATCH_N_LAYERS = 2
PATCH_DROPOUT = 0.2

TCN_CHANNELS = [N_VARS, 32, 64, 128]
TCN_REGION_DIM = 16
TCN_DROPOUT = 0.2
TCN_KERNEL = 3

DLINEAR_MA_KERNEL = 5
DLINEAR_REGION_DIM = 16

FUSED_CHANNELS = [N_VARS, 32, 64, 128]
FUSED_REGION_DIM = 16
FUSED_FEAT_DIM = 32
FUSED_DROPOUT = 0.2