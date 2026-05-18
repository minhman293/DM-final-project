# predict.py — Load trained model and generate Kaggle submission
#
# Run: python predict.py
#
# Steps:
#   1. Load test.csv + train.csv (needed for lag scores and region baseline)
#   2. Build features (same pipeline as train.py)
#   3. For each region: predict 5 weeks ahead using autoregressive approach
#   4. Save submissions/submission.csv

import pandas as pd
import numpy as np
import lightgbm as lgb
import joblib
import json
from pathlib import Path

from config import (
    TRAIN_PATH, TEST_PATH, SAMPLE_SUB_PATH,
    MODEL_DIR, SUBMISSION_DIR,
    TARGET_COL, REGION_COL, PRED_COLS, PRED_WEEKS, LAG_WEEKS,
    ROLLING_WINDOWS, ROLLING_FEATURES
)
from utils import extract_date_parts, get_logger, get_last_known_scores_per_lag
from feature_engineering import build_features, get_feature_cols

logger = get_logger("predict")


def predict_autoregressive(
    model: lgb.LGBMRegressor,
    test_weekly: pd.DataFrame,
    feature_cols: list[str],
) -> pd.DataFrame:
    """
    Predict 5 weeks ahead for each region using an autoregressive approach.

    Strategy:
    - Week 1: use lag scores from training data (lag_score_1 = last train score)
    - Week 2-5: use previous predictions as lag_score_1, shift others forward

    Returns DataFrame with columns: region_id, pred_week1..pred_week5
    """
    logger.info("Generating predictions (autoregressive, 5 weeks)...")

    all_predictions = []
    regions = test_weekly[REGION_COL].unique()

    for region in regions:
        region_data = test_weekly[test_weekly[REGION_COL] == region].copy()

        # Use the single test feature row (aggregated from 91 days)
        # It already has lag_score_1..N from training data
        row = region_data.iloc[0].copy()

        preds = []
        for week in range(1, PRED_WEEKS + 1):
            # Build feature vector for this week
            feat_vec = row[feature_cols].values.reshape(1, -1).astype(np.float32)
            feat_df  = pd.DataFrame(feat_vec, columns=feature_cols, dtype=np.float32)
            
            # Predict and clip to valid range [0, 5]
            pred = float(np.clip(model.predict(feat_df)[0], 0, 5))
            preds.append(pred)

            # Autoregressive update: shift lag scores forward
            # lag_score_1 <- this prediction
            # lag_score_2 <- old lag_score_1
            # lag_score_3 <- old lag_score_2
            # lag_score_4 <- old lag_score_3
            for lag in sorted(LAG_WEEKS, reverse=True):
                if lag == 1:
                    row[f"lag_score_1"] = pred
                elif f"lag_score_{lag-1}" in row.index:
                    row[f"lag_score_{lag}"] = row[f"lag_score_{lag-1}"]

        all_predictions.append({REGION_COL: region, **dict(zip(PRED_COLS, preds))})

    result = pd.DataFrame(all_predictions)
    logger.info(f"Predictions generated for {len(regions)} regions.")
    return result


def aggregate_test_to_single_row(test_weekly: pd.DataFrame) -> pd.DataFrame:
    """
    The test set has 91 days = 13 weeks of features per region.
    We aggregate these 13 weeks into a single feature row per region
    by taking the mean of each feature — representing the 'recent context'
    that the model uses to predict the next 5 weeks.

    Lag scores are already attached per region from training data,
    so we just preserve them from the first row (they're identical across all
    13 test weeks for the same region).
    """
    lag_cols      = [f"lag_score_{l}" for l in LAG_WEEKS]
    rolling_cols  = [c for c in test_weekly.columns
                     if any(c.startswith(f) for f in ROLLING_FEATURES)
                     and "roll" in c]
    baseline_cols = ["region_mean_score"] if "region_mean_score" in test_weekly.columns else []

    # Features to average across the 13 test weeks
    meteo_weekly_cols = [c for c in test_weekly.columns
                         if c not in [REGION_COL, "week_id", TARGET_COL,
                                      "month", "date", "year", "day", "jdn", "day_of_week"]
                         and c not in lag_cols
                         and c not in rolling_cols
                         and c not in baseline_cols]

    # Average meteorological weekly features
    agg_dict = {c: "mean" for c in meteo_weekly_cols}

    # For lag scores, rolling features, baseline — take the last row
    # (they represent the most recent state, not an average)
    for c in lag_cols + rolling_cols + baseline_cols:
        if c in test_weekly.columns:
            agg_dict[c] = "last"

    # Month — take last (most recent week's month)
    if "month" in test_weekly.columns:
        agg_dict["month"] = "last"

    aggregated = (test_weekly
                  .groupby(REGION_COL)
                  .agg(agg_dict)
                  .reset_index())

    logger.info(f"Test aggregated to single row per region: {aggregated.shape}")
    return aggregated


def main():
    # ── 1. Load raw data ──────────────────────────────────────────────────────
    logger.info("Loading raw data...")
    train_raw = pd.read_csv(TRAIN_PATH)
    test_raw  = pd.read_csv(TEST_PATH)

    # ── 2. Extract date parts ─────────────────────────────────────────────────
    logger.info("Extracting date parts...")
    train_raw = extract_date_parts(train_raw)
    test_raw  = extract_date_parts(test_raw)

    # ── 3. Build features ─────────────────────────────────────────────────────
    _, test_weekly = build_features(train_raw, test_raw)

    # ── 4. Aggregate test to one row per region ───────────────────────────────
    test_single = aggregate_test_to_single_row(test_weekly)

    # ── 5. Load model and feature list ───────────────────────────────────────
    model_path   = MODEL_DIR / "lgbm_model.pkl"
    feature_path = MODEL_DIR / "feature_cols.json"

    logger.info(f"Loading model from {model_path}...")
    model = joblib.load(model_path)

    with open(feature_path, "r") as f:
        feature_cols = json.load(f)
    logger.info(f"Feature list loaded: {len(feature_cols)} features")

    # Ensure all feature columns exist in test (fill missing with 0)
    for col in feature_cols:
        if col not in test_single.columns:
            logger.warning(f"Feature '{col}' missing in test — filling with 0")
            test_single[col] = 0.0

    # Cast all feature columns to float32 — LightGBM requires numeric dtypes
    test_single[feature_cols] = test_single[feature_cols].astype(np.float32)

    # ── 6. Generate predictions ───────────────────────────────────────────────
    predictions = predict_autoregressive(model, test_single, feature_cols)

    # ── 7. Format submission ──────────────────────────────────────────────────
    logger.info("Formatting submission...")
    sample_sub = pd.read_csv(SAMPLE_SUB_PATH)

    # Merge predictions onto sample submission to ensure correct region order
    submission = sample_sub[[REGION_COL]].merge(predictions, on=REGION_COL, how="left")

    # Fill any missing regions with 0 (fallback)
    submission[PRED_COLS] = submission[PRED_COLS].fillna(0.0)

    # Round to 4 decimal places
    submission[PRED_COLS] = submission[PRED_COLS].round(4)

    # ── 8. Save ───────────────────────────────────────────────────────────────
    model_name = model_path.stem   # e.g. "lgbm_model"
    out_path   = SUBMISSION_DIR / f"submission_{model_name}.csv"
    submission.to_csv(out_path, index=False)
    logger.info(f"Submission saved: {out_path}")
    logger.info(f"Shape: {submission.shape}")
    logger.info(f"Preview:\n{submission.head(10).to_string()}")
    logger.info("Prediction complete.")


if __name__ == "__main__":
    main()