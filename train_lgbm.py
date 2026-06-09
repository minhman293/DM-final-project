# train_lgbm.py
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
    print("1. Loading Numpy Arrays...")
    train_features = np.load(MODEL_DIR / "train_features.npy") # (2248, 391, 10)
    train_targets = np.load(MODEL_DIR / "train_targets.npy")   # (2248, 391)
    test_features = np.load(MODEL_DIR / "test_features.npy")   # (2248, 26, 10)
    region_to_idx = joblib.load(MODEL_DIR / "region_to_idx.pkl")
    idx_to_region = {v: k for k, v in region_to_idx.items()}
    
    num_regions, num_weeks, num_feats = train_features.shape

    print("2. Flattening Time-Series into Tabular Rows for LightGBM...")
    X_train, Y_train = [], []
    
    # Slide the window to create tabular training rows
    for r in range(num_regions):
        for t in range(HISTORY - 1, num_weeks - HORIZONS):
            # Flatten 26 weeks * 10 features = 260 columns
            x = train_features[r, t-HISTORY+1 : t+1, :].flatten()
            # Append Region ID as the final categorical feature
            x = np.append(x, r) 
            y = train_targets[r, t+1 : t+1+HORIZONS]
            
            X_train.append(x)
            Y_train.append(y)

    X_train = np.array(X_train, dtype=np.float32)
    Y_train = np.array(Y_train, dtype=np.float32)
    
    # Build Test Set
    X_test, test_regions = [], []
    for r in range(num_regions):
        x = test_features[r, :, :].flatten()
        x = np.append(x, r)
        X_test.append(x)
        test_regions.append(idx_to_region[r])
        
    X_test = np.array(X_test, dtype=np.float32)
    
    print(f"Tabular Matrix built! Training Shape: {X_train.shape}")
    
    # LightGBM Hyperparameters configured specifically for MAE
    lgb_params = {
        'objective': 'regression_l1', # Mathematically optimizes for MAE
        'metric': 'mae',
        'boosting_type': 'gbdt',
        'learning_rate': 0.05,
        'num_leaves': 63,
        'feature_fraction': 0.8,
        'verbose': -1,
        'n_jobs': -1,
        'random_state': 42
    }

    print("3. Training 5 separate LightGBM models (one for each future week)...")
    preds = np.zeros((len(X_test), HORIZONS))
    
    for i in range(HORIZONS):
        print(f"   -> Training Model for Week {i+1}...")
        
        # We hold out the last 10% of rows for simple early stopping to prevent overfitting
        split = int(len(X_train) * 0.9)
        train_data = lgb.Dataset(X_train[:split], label=Y_train[:split, i], categorical_feature=[X_train.shape[1]-1])
        val_data   = lgb.Dataset(X_train[split:], label=Y_train[split:, i], categorical_feature=[X_train.shape[1]-1])
        
        model = lgb.train(
            lgb_params, 
            train_data, 
            valid_sets=[val_data], 
            num_boost_round=1000, 
            callbacks=[lgb.early_stopping(stopping_rounds=50, verbose=False)]
        )
        
        preds[:, i] = model.predict(X_test)
        
    print("4. Saving LightGBM Predictions...")
    # Clamp extreme values just in case
    preds = np.clip(preds, 0.0, 5.0)
    
    df = pd.DataFrame(preds, columns=[f"pred_week{i+1}" for i in range(5)])
    df.insert(0, "region_id", test_regions)
    
    out_path = SUB_DIR / "submission_lgbm_trees.csv"
    df.to_csv(out_path, index=False)
    print(f"Saved -> {out_path}")

if __name__ == "__main__":
    main()