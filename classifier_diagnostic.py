import numpy as np
import lightgbm as lgb
import joblib
from pathlib import Path
from sklearn.metrics import precision_score

# Paths
MODEL_DIR = Path("models_pipeline_b")
HISTORY = 26
HORIZONS = 5

def main():
    print("1. Loading Pre-processed Arrays...")
    train_features = np.load(MODEL_DIR / "train_features.npy") 
    train_targets = np.load(MODEL_DIR / "train_targets.npy")   
    
    num_regions, num_weeks, num_feats = train_features.shape

    print("2. Performing Strict Region-Based Split (No Leakage)...")
    # Hold out 20% of REGIONS entirely
    np.random.seed(42)
    all_regions = np.arange(num_regions)
    np.random.shuffle(all_regions)
    
    val_region_count = int(num_regions * 0.2)
    val_regions = set(all_regions[:val_region_count])
    
    X_tr, Y_tr, X_va, Y_va = [], [], [], []
    
    for r in range(num_regions):
        for t in range(HISTORY - 1, num_weeks - HORIZONS):
            x = train_features[r, t-HISTORY+1 : t+1, :].flatten()
            
            last_score = train_targets[r, t]
            reg_mean = np.mean(train_targets[r, :t+1])
            reg_zero_frac = np.mean(train_targets[r, :t+1] == 0)
            
            # Dropped the Region ID categorical feature to prevent memorization
            x = np.concatenate([x, [last_score, reg_mean, reg_zero_frac]])
            y = (train_targets[r, t+1 : t+1+HORIZONS] > 0).astype(int)
            
            if r in val_regions:
                X_va.append(x)
                Y_va.append(y)
            else:
                X_tr.append(x)
                Y_tr.append(y)

    X_tr = np.array(X_tr, dtype=np.float32)
    Y_tr = np.array(Y_tr, dtype=np.int8)
    X_va = np.array(X_va, dtype=np.float32)
    Y_va = np.array(Y_va, dtype=np.int8)

    lgb_params = {
        'objective': 'binary',
        'metric': 'binary_logloss',
        'boosting_type': 'gbdt',
        'learning_rate': 0.05,
        'num_leaves': 31,
        'is_unbalance': True,
        'verbose': -1,
        'n_jobs': -1,
        'random_state': 42
    }

    print("3. Training Classifier (Week 1 only for diagnostic)...")
    # We only test Week 1 to gauge baseline classifier calibration
    train_data = lgb.Dataset(X_tr, label=Y_tr[:, 0])
    val_data   = lgb.Dataset(X_va, label=Y_va[:, 0])
    
    model = lgb.train(
        lgb_params, train_data, valid_sets=[val_data], 
        num_boost_round=600, 
        callbacks=[lgb.early_stopping(stopping_rounds=50, verbose=False)]
    )
    
    print("\n4. Threshold Precision Calibration...")
    preds_va = model.predict(X_va)
    
    thresholds = [0.05, 0.10, 0.15, 0.20, 0.25, 0.30]
    
    for t in thresholds:
        forced_zero_mask = preds_va < t
        coverage = forced_zero_mask.mean()
        
        if coverage > 0:
            # How often was the true score actually 0 when we forced it?
            actually_zero = (Y_va[:, 0][forced_zero_mask] == 0).mean()
            print(f"Threshold {t:.2f} | Precision: {actually_zero:.3f} | Coverage: {coverage:.3f}")
        else:
            print(f"Threshold {t:.2f} | Precision: N/A | Coverage: 0.000")

if __name__ == "__main__":
    main()