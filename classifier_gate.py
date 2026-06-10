import numpy as np
import pandas as pd
import lightgbm as lgb
import joblib
from pathlib import Path

# Paths
MODEL_DIR = Path("models_pipeline_b")
SUB_DIR = Path("submissions")
HISTORY = 26
HORIZONS = 5

def main():
    print("1. Loading Pre-processed Arrays...")
    train_features = np.load(MODEL_DIR / "train_features.npy") # (2248, 391, 10)
    train_targets = np.load(MODEL_DIR / "train_targets.npy")   # (2248, 391)
    test_features = np.load(MODEL_DIR / "test_features.npy")   # (2248, 26, 10)
    region_to_idx = joblib.load(MODEL_DIR / "region_to_idx.pkl")
    idx_to_region = {v: k for k, v in region_to_idx.items()}
    
    num_regions, num_weeks, num_feats = train_features.shape

    print("2. Building Classifier Dataset with Context Features...")
    X_train, Y_train = [], []
    
    # We step through the sequence, ensuring no future data leakage
    for r in range(num_regions):
        for t in range(HISTORY - 1, num_weeks - HORIZONS):
            # Flatten weather history
            x = train_features[r, t-HISTORY+1 : t+1, :].flatten()
            
            # The "Golden Features": Past behavior
            last_score = train_targets[r, t]
            reg_mean = np.mean(train_targets[r, :t+1])
            reg_zero_frac = np.mean(train_targets[r, :t+1] == 0)
            
            # Append context + region ID
            x = np.concatenate([x, [last_score, reg_mean, reg_zero_frac, r]])
            
            # Target for the next 5 weeks (1 = non-zero/drought, 0 = zero)
            y = (train_targets[r, t+1 : t+1+HORIZONS] > 0).astype(int)
            
            X_train.append(x)
            Y_train.append(y)

    X_train = np.array(X_train, dtype=np.float32)
    Y_train = np.array(Y_train, dtype=np.int8)
    
    # Build Test Set
    X_test, test_regions = [], []
    for r in range(num_regions):
        x = test_features[r, :, :].flatten()
        last_score = train_targets[r, -1] # Absolute last known score before test window
        reg_mean = np.mean(train_targets[r, :])
        reg_zero_frac = np.mean(train_targets[r, :] == 0)
        
        x = np.concatenate([x, [last_score, reg_mean, reg_zero_frac, r]])
        X_test.append(x)
        test_regions.append(idx_to_region[r])
        
    X_test = np.array(X_test, dtype=np.float32)
    
    print(f"Dataset ready! Training rows: {X_train.shape[0]}, Features: {X_train.shape[1]}")
    
    # LightGBM Params specifically for imbalanced Binary Classification
    lgb_params = {
        'objective': 'binary',
        'metric': 'binary_logloss',
        'boosting_type': 'gbdt',
        'learning_rate': 0.05,
        'num_leaves': 31,
        'is_unbalance': True, # Automatically scales weights for the 57% zeros
        'verbose': -1,
        'n_jobs': -1,
        'random_state': 42
    }

    print("3. Training Binary Classifiers...")
    pred_probs = np.zeros((len(X_test), HORIZONS))
    
    for i in range(HORIZONS):
        print(f"   -> Training Classifier for Week {i+1}...")
        
        split = int(len(X_train) * 0.9)
        train_data = lgb.Dataset(X_train[:split], label=Y_train[:split, i], categorical_feature=[X_train.shape[1]-1])
        val_data   = lgb.Dataset(X_train[split:], label=Y_train[split:, i], categorical_feature=[X_train.shape[1]-1])
        
        model = lgb.train(
            lgb_params, train_data, valid_sets=[val_data], 
            num_boost_round=600, 
            callbacks=[lgb.early_stopping(stopping_rounds=50, verbose=False)]
        )
        
        # We want the probability that the row is NON-ZERO
        pred_probs[:, i] = model.predict(X_test)
        
    print("4. Applying the Classifier Gate to the Smoothed Predictions...")
    # Load your current champion 0.8189 file
    smoothed_sub = pd.read_csv(SUB_DIR / "SMOOTHED_64TFT_36LSTM.csv")
    pred_cols = ["pred_week1", "pred_week2", "pred_week3", "pred_week4", "pred_week5"]
    
    gated_sub = smoothed_sub.copy()
    
    total_cells = len(gated_sub) * HORIZONS
    cells_zeroed = 0
    
    for i, col in enumerate(pred_cols):
        # If the probability of being NON-zero is less than 30% 
        # (Meaning it is > 70% confident it is a ZERO) -> Force to 0.0
        mask_force_zero = pred_probs[:, i] < 0.30
        gated_sub.loc[mask_force_zero, col] = 0.0
        
        # Optional: Gentle pull for uncertain rows (30% to 50% non-zero chance)
        mask_uncertain = (pred_probs[:, i] >= 0.30) & (pred_probs[:, i] < 0.50)
        gated_sub.loc[mask_uncertain, col] *= 0.80 
        
        cells_zeroed += mask_force_zero.sum()

    print(f"\n--- Gating Summary ---")
    print(f"Total Prediction Cells: {total_cells:,}")
    print(f"Cells forcefully zeroed: {cells_zeroed:,} ({(cells_zeroed/total_cells)*100:.1f}%)")

    out_path = SUB_DIR / "GATED_SMOOTHED_FINAL.csv"
    gated_sub.to_csv(out_path, index=False)
    print(f"Success! Final weapon saved to -> {out_path}")

if __name__ == "__main__":
    main()