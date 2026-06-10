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

    df = df.sort_values(['region_id', 'time_idx']).reset_index(drop=True)

    print("2. Engineering Deep Memory Features...")
    # Lag scores
    df['score_lag_1'] = df['score']
    df['score_lag_2'] = df.groupby('region_id')['score'].shift(1)
    df['score_lag_4'] = df.groupby('region_id')['score'].shift(3)
    df['score_lag_13'] = df.groupby('region_id')['score'].shift(12)
    df['score_lag_26'] = df.groupby('region_id')['score'].shift(25)

    # Score trends (the SLOPE of the score is informative)
    df['score_change_1w'] = df['score_lag_1'] - df['score_lag_2']
    df['score_change_4w'] = df['score_lag_1'] - df['score_lag_4']
    df['score_change_13w'] = df['score_lag_1'] - df['score_lag_13']

    # Cumulative precipitation
    df['prec_4w_sum'] = df.groupby('region_id')['prec_sum'].transform(
        lambda x: x.rolling(4, min_periods=1).sum())
    df['prec_12w_sum'] = df.groupby('region_id')['prec_sum'].transform(
        lambda x: x.rolling(12, min_periods=1).sum())
    df['prec_26w_sum'] = df.groupby('region_id')['prec_sum'].transform(
        lambda x: x.rolling(26, min_periods=1).sum())

    # Temperature anomaly persistence
    df['tmp_anom_4w_mean'] = df.groupby('region_id')['tmp_mean_anomaly'].transform(
        lambda x: x.rolling(4, min_periods=1).mean())
    df['tmp_anom_12w_mean'] = df.groupby('region_id')['tmp_mean_anomaly'].transform(
        lambda x: x.rolling(12, min_periods=1).mean())

    # Region's typical score for this calendar month (target encoding)
    train_scored = df[df['score_known'] == 1].copy()
    train_scored['_m'] = train_scored['month'].astype(int)
    df['_m'] = df['month'].astype(int)
    region_month_mean = (train_scored.groupby(['region_id', '_m'])['score']
                         .mean().rename('region_month_score_mean').reset_index())
    df = df.merge(region_month_mean, on=['region_id', '_m'], how='left')
    df['region_month_score_mean'] = df['region_month_score_mean'].fillna(
        train_scored['score'].mean())
    df = df.drop(columns=['_m'])

    # FIX ISSUE #2: mark rows whose shifted target falls on fake-fill data
    df['_is_train'] = (df['is_test'] == 0).astype(int)

    # Define features
    ignore_cols = ['region_id', 'time_idx', 'score', 'score_known', 'is_test',
                   'month', '_is_train']
    features = [c for c in df.columns if c not in ignore_cols]
    print(f"   Using {len(features)} features.")

    # Build training data (only original train rows, not test/future)
    train_df = df[df['is_test'] == 0].copy()

    # Build the "predict from" row per region — last is_test==1 week
    last_test_weeks = (df[df['is_test'] == 1]
                       .groupby('region_id').tail(1).copy())
    last_test_weeks['region_id'] = last_test_weeks['region_id'].astype(str)
    last_test_weeks = last_test_weeks.sort_values('region_id').reset_index(drop=True)

    predictions = {}

    print("3. Direct multi-step forecasting with validation...")
    for h in range(1, 6):
        target_col = f'target_w{h}'
        train_df[target_col] = train_df.groupby('region_id')['score'].shift(-h)

        # FIX ISSUE #2: drop rows where the shifted target is on a fake-fill week
        # The shifted row's _is_train must also be 1
        is_train_shifted = (train_df.groupby('region_id')['_is_train']
                            .shift(-h))
        valid_mask = (train_df[target_col].notna()) & (is_train_shifted == 1)
        valid_train = train_df[valid_mask].copy()

        # FIX ISSUE #3: hold out the last 60 weeks per region for early stopping
        max_idx_per_region = valid_train.groupby('region_id')['time_idx'].transform('max')
        is_val = (max_idx_per_region - valid_train['time_idx']) < 60
        tr_data = valid_train[~is_val]
        val_data = valid_train[is_val]

        X_tr, y_tr = tr_data[features], tr_data[target_col]
        X_va, y_va = val_data[features], val_data[target_col]

        print(f"   -> Week {h}: train={len(X_tr):,}, val={len(X_va):,}")

        model = lgb.LGBMRegressor(
            n_estimators=2000,           # high cap; early stopping picks the right number
            learning_rate=0.03,          # slightly lower for more careful learning
            num_leaves=63,               # a bit more capacity
            min_child_samples=50,        # regularization
            colsample_bytree=0.8,
            subsample=0.8,
            subsample_freq=1,
            reg_alpha=0.1,
            reg_lambda=0.1,
            objective='mae',
            random_state=42,
            n_jobs=-1,
            verbose=-1,
        )

        model.fit(
            X_tr, y_tr,
            eval_set=[(X_va, y_va)],
            eval_metric='mae',
            callbacks=[lgb.early_stopping(50, verbose=False)],
        )

        # Predict on test
        X_test = last_test_weeks[features]
        preds_h = model.predict(X_test)
        preds_h = np.clip(preds_h, 0.0, 5.0)
        preds_h = np.where(preds_h < 0.05, 0.0, preds_h)
        predictions[f"pred_week{h}"] = preds_h

        print(f"      best val MAE: {model.best_score_['valid_0']['l1']:.4f}")
        print(f"      best iter: {model.best_iteration_}")

    print("4. Formatting submission...")
    sub = pd.DataFrame({'region_id': last_test_weeks['region_id']})
    for h in range(1, 6):
        sub[f"pred_week{h}"] = np.round(predictions[f"pred_week{h}"], 4)

    out_path = Path("submissions/LGBM_DEEP_MEMORY_V2.csv")
    sub.to_csv(out_path, index=False)
    print(f"Saved → {out_path}")

if __name__ == "__main__":
    main()