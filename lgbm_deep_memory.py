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

    # Ensure absolute strict temporal sorting before rolling calculations
    df = df.sort_values(['region_id', 'time_idx']).reset_index(drop=True)

    print("2. Engineering Deep Memory Features (The 0.79 Recipe)...")
    
    # A. Deep Lags (Where was the drought 1, 2, 4, 13, and 26 weeks ago?)
    # df['score'] is the currently known anchor week.
    df['score_lag_1'] = df['score'] 
    df['score_lag_2'] = df.groupby('region_id')['score'].shift(1)
    df['score_lag_4'] = df.groupby('region_id')['score'].shift(3)
    df['score_lag_13'] = df.groupby('region_id')['score'].shift(12) # ~1 Quarter ago
    df['score_lag_26'] = df.groupby('region_id')['score'].shift(25) # ~Half a year ago

    # B. Cumulative Precipitation (Are we in a long-term deficit?)
    # min_periods=1 prevents dropping the first few rows of the dataset
    df['prec_4w_sum'] = df.groupby('region_id')['prec_sum'].transform(lambda x: x.rolling(4, min_periods=1).sum())
    df['prec_12w_sum'] = df.groupby('region_id')['prec_sum'].transform(lambda x: x.rolling(12, min_periods=1).sum())
    df['prec_26w_sum'] = df.groupby('region_id')['prec_sum'].transform(lambda x: x.rolling(26, min_periods=1).sum())

    # C. Sustained Heatwaves (Is the anomaly persistent?)
    df['tmp_anom_4w_mean'] = df.groupby('region_id')['tmp_mean_anomaly'].transform(lambda x: x.rolling(4, min_periods=1).mean())
    df['tmp_anom_12w_mean'] = df.groupby('region_id')['tmp_mean_anomaly'].transform(lambda x: x.rolling(12, min_periods=1).mean())

    # Define the final feature list
    ignore_cols = ['region_id', 'time_idx', 'score', 'score_known', 'is_test', 'month']
    features = [c for c in df.columns if c not in ignore_cols]
    
    print(f"   -> Upgraded to {len(features)} total features.")

    # Isolate training data
    train_df = df[df['is_test'] == 0].copy()

    # Isolate the exact jumping-off point for the test set
    test_history = df[df['is_test'] == 1].copy()
    last_test_weeks = test_history.groupby('region_id').tail(1).copy()
    last_test_weeks['region_id'] = last_test_weeks['region_id'].astype(str)
    last_test_weeks = last_test_weeks.sort_values('region_id').reset_index(drop=True)

    predictions = {}
    
    print("3. Executing Direct Multi-Step Forecasting...")
    for h in range(1, 6):
        target_col = f'target_week_{h}'
        train_df[target_col] = train_df.groupby('region_id')['score'].shift(-h)
        
        valid_train = train_df.dropna(subset=[target_col])
        
        X_train = valid_train[features]
        y_train = valid_train[target_col]
        
        print(f"   -> Training LightGBM for Week {h} Horizon...")
        
        # Added colsample_bytree to handle the newly correlated rolling features
        model = lgb.LGBMRegressor(
            n_estimators=350,
            learning_rate=0.04,
            num_leaves=31,
            colsample_bytree=0.8,
            objective='mae', 
            random_state=42,
            n_jobs=-1,
            verbose=-1
        )
        
        model.fit(X_train, y_train)
        
        X_test = last_test_weeks[features]
        preds_h = model.predict(X_test)
        
        # Clamp bounds and squeeze microscopic noise
        preds_h = np.clip(preds_h, 0.0, 5.0)
        preds_h = np.where(preds_h < 0.05, 0.0, preds_h)
        
        predictions[f"pred_week{h}"] = preds_h

    print("4. Formatting Final Submission...")
    sub = pd.DataFrame({'region_id': last_test_weeks['region_id']})
    for h in range(1, 6):
        sub[f"pred_week{h}"] = np.round(predictions[f"pred_week{h}"], 4)

    out_path = Path("submissions/LGBM_DEEP_MEMORY.csv")
    sub.to_csv(out_path, index=False)
    print(f"Success! Final dart thrown. Saved to {out_path}")

if __name__ == "__main__":
    main()