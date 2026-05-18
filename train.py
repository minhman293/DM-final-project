# train.py — Train LightGBM model and save to models/
#
# Run: python train.py
#
# Steps:
#   1. Load train.csv and test.csv
#   2. Extract date parts
#   3. Build features via feature_engineering.py
#   4. Region-level train/val split
#   5. Train LightGBM with early stopping
#   6. Evaluate on validation set (MAE)
#   7. Save model + feature list

import pandas as pd
import numpy as np
import lightgbm as lgb
import joblib
import json
from pathlib import Path

from config import (
    TRAIN_PATH, TEST_PATH, MODEL_DIR,
    LGBM_PARAMS, EARLY_STOPPING_ROUNDS, LOG_EVERY_N_ROUNDS,
    TARGET_COL, REGION_COL, SEED, VAL_REGION_COUNT, LAG_WEEKS
)
from utils import extract_date_parts, mae, get_logger
from feature_engineering import build_features, get_feature_cols

logger = get_logger("train")


def region_split(
    train_weekly: pd.DataFrame,
    val_region_count: int,
    seed: int
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Split by region — all weeks of a region go entirely into train or val.
    This avoids data leakage from lag features crossing the split boundary.
    """
    all_regions = train_weekly[REGION_COL].unique()
    rng = np.random.default_rng(seed)
    val_regions = rng.choice(all_regions, size=val_region_count, replace=False)

    val_mask   = train_weekly[REGION_COL].isin(val_regions)
    train_mask = ~val_mask

    return train_weekly[train_mask].copy(), train_weekly[val_mask].copy()


def main():
    # ── 1. Load raw data ──────────────────────────────────────────────────────
    logger.info("Loading raw data...")
    train_raw = pd.read_csv(TRAIN_PATH)
    test_raw  = pd.read_csv(TEST_PATH)
    logger.info(f"Train: {train_raw.shape}, Test: {test_raw.shape}")

    # ── 2. Extract date parts ─────────────────────────────────────────────────
    logger.info("Extracting date parts...")
    train_raw = extract_date_parts(train_raw)
    test_raw  = extract_date_parts(test_raw)

    # ── 3. Build features ─────────────────────────────────────────────────────
    train_weekly, test_weekly = build_features(train_raw, test_raw)

    # ── 4. Train / validation split ───────────────────────────────────────────
    logger.info(f"Splitting: {VAL_REGION_COUNT} regions held out for validation...")
    tr, val = region_split(train_weekly, VAL_REGION_COUNT, SEED)
    logger.info(f"Train weeks: {len(tr):,} | Val weeks: {len(val):,}")

    # ── 5. Define feature columns ─────────────────────────────────────────────
    feature_cols = get_feature_cols(train_weekly)
    logger.info(f"Features ({len(feature_cols)}): {feature_cols}")

    X_tr  = tr[feature_cols]
    y_tr  = tr[TARGET_COL]
    X_val = val[feature_cols]
    y_val = val[TARGET_COL]

    # ── 6. Train LightGBM ─────────────────────────────────────────────────────
    logger.info("Training LightGBM...")

    callbacks = [
        lgb.early_stopping(stopping_rounds=EARLY_STOPPING_ROUNDS, verbose=True),
        lgb.log_evaluation(period=LOG_EVERY_N_ROUNDS),
    ]

    model = lgb.LGBMRegressor(**LGBM_PARAMS)
    model.fit(
        X_tr, y_tr,
        eval_set=[(X_val, y_val)],
        callbacks=callbacks,
    )

    # ── 7. Evaluate ───────────────────────────────────────────────────────────
    val_pred   = model.predict(X_val)
    val_pred   = np.clip(val_pred, 0, 5)   # score is bounded [0, 5]
    val_mae    = mae(y_val.values, val_pred)
    logger.info(f"Validation MAE: {val_mae:.4f}")
    logger.info(f"Best iteration: {model.best_iteration_}")

    # Baselines for reference
    logger.info(f"Baseline 1 (predict mean): {mae(y_val.values, np.full_like(val_pred, y_tr.mean())):.4f}")
    logger.info(f"Baseline 0 (predict zero): {mae(y_val.values, np.zeros_like(val_pred)):.4f}")

    # Feature importance
    fi = pd.Series(model.feature_importances_, index=feature_cols)
    fi = fi.sort_values(ascending=False)
    logger.info(f"Top 15 features:\n{fi.head(15).to_string()}")

    # ── 8. Save model and metadata ────────────────────────────────────────────
    model_path   = MODEL_DIR / "lgbm_model.pkl"
    feature_path = MODEL_DIR / "feature_cols.json"
    meta_path    = MODEL_DIR / "train_meta.json"

    joblib.dump(model, model_path)
    logger.info(f"Model saved: {model_path}")

    with open(feature_path, "w") as f:
        json.dump(feature_cols, f)
    logger.info(f"Feature list saved: {feature_path}")

    meta = {
        "val_mae"        : round(float(val_mae), 4),
        "best_iteration" : int(model.best_iteration_),
        "n_train_weeks"  : int(len(tr)),
        "n_val_weeks"    : int(len(val)),
        "val_regions"    : int(VAL_REGION_COUNT),
        "n_features"     : int(len(feature_cols)),
    }
    with open(meta_path, "w") as f:
        json.dump(meta, f, indent=2)
    logger.info(f"Metadata saved: {meta_path}")
    logger.info("Training complete.")


if __name__ == "__main__":
    main()