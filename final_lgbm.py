import pandas as pd
import numpy as np
import lightgbm as lgb
from pathlib import Path

def main():
    print("1. Loading Anomaly-Aware Data...")
    cache_path = Path("artifacts_tft/combined_frame.parquet")
    if cache_path.exists():
        df = pd.read_parquet(cache_path)
    else:
        df = pd.read_pickle(Path("artifacts_tft/combined_frame.pkl"))

    # Select all engineered features (includes anomalies, month encodings, and weather)
    # We drop the structural columns and the target
    ignore_cols = ['region_id', 'time_idx', 'score', 'score_known', 'is_test', 'month']
    features = [c for c in df.columns if c not in ignore_cols]
    
    # We must explicitly add the CURRENT week's score as a feature, 
    # because Lag-1 Autocorrelation (0.936) is the strongest signal in the dataset.
    df['current_score'] = df['score']
    features.append('current_score')

    print(f"Using {len(features)} features...")

    # Isolate training data
    train_df = df[df['is_test'] == 0].copy()

    # Isolate the jumping-off point for the test set.
    # is_test == 1 is the 13-week context window provided by Kaggle. 
    # To predict the future, we only need the absolute LAST week of that context window.
    test_history = df[df['is_test'] == 1].copy()
    last_test_weeks = test_history.groupby('region_id').tail(1).copy()
    
    # Sort test regions to perfectly match the sample_submission order
    last_test_weeks['region_id'] = last_test_weeks['region_id'].astype(str)
    last_test_weeks = last_test_weeks.sort_values('region_id').reset_index(drop=True)

    predictions = {}
    
    print("2. Executing Direct Multi-Step Forecasting (5 Independent Models)...")
    for h in range(1, 6):
        # Shift the target backward by 'h' weeks. 
        # This aligns THIS week's weather with the drought score 'h' weeks in the future.
        target_col = f'target_week_{h}'
        train_df[target_col] = train_df.groupby('region_id')['score'].shift(-h)
        
        # Drop rows where the future is unknown (end of the sequences)
        valid_train = train_df.dropna(subset=[target_col])
        
        X_train = valid_train[features]
        y_train = valid_train[target_col]
        
        print(f"   -> Training LightGBM for Week {h} Horizon...")
        # Hyperparameters roughly tuned for MAE robustness
        model = lgb.LGBMRegressor(
            n_estimators=250,
            learning_rate=0.05,
            num_leaves=31,
            objective='mae', # Force the tree to optimize for Kaggle's exact metric
            random_state=42,
            n_jobs=-1,
            verbose=-1
        )
        
        model.fit(X_train, y_train)
        
        X_test = last_test_weeks[features]
        preds_h = model.predict(X_test)
        
        # Clamp bounds
        preds_h = np.clip(preds_h, 0.0, 5.0)
        # Squeeze the microscopic noise down to zero
        preds_h = np.where(preds_h < 0.05, 0.0, preds_h)
        
        predictions[f"pred_week{h}"] = preds_h

    print("3. Formatting Final Submission...")
    sub = pd.DataFrame({'region_id': last_test_weeks['region_id']})
    for h in range(1, 6):
        sub[f"pred_week{h}"] = np.round(predictions[f"pred_week{h}"], 4)

    out_path = Path("submissions/LGBM_ANOMALY.csv")
    sub.to_csv(out_path, index=False)
    print(f"Success! Saved to {out_path}")

if __name__ == "__main__":
    main()