# predict.py — Load trained model and generate Kaggle submission
#
# Run: python predict.py
#
# Strategy:
#   1. Load test.csv + train.csv (needed for lag scores and region baseline)
#   2. Build features (same pipeline as train.py)
#   3. For each region: use the LAST week of the 13 test weeks as input row
#   4. Predict 5 weeks autoregressively:
#        week 1: input = last test week features + lag scores from train end
#        week 2: update lag_score_1 = pred_week1, shift others, predict again
#        week 3..5: same pattern
#   5. Save submissions/submission_lgbm_model.csv

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
)
from utils import extract_date_parts, get_logger
from feature_engineering import build_features, get_feature_cols

logger = get_logger("predict")


def get_last_test_week(test_weekly: pd.DataFrame) -> pd.DataFrame:
    """
    From the 13 test weeks per region, keep only the LAST week (week_id = 12).
    This is the most recent meteorological context before prediction.

    Why last week instead of average:
    - The model was trained on individual weekly rows, not averages
    - The most recent week's features are most relevant for near-future prediction
    - Lag scores already encode the historical drought state
    """
    last_week = (test_weekly
                 .sort_values([REGION_COL, "week_id"])
                 .groupby(REGION_COL)
                 .last()
                 .reset_index())
    logger.info(f"Last test week per region: {last_week.shape}")
    return last_week


def predict_autoregressive(
    model: lgb.LGBMRegressor,
    test_last_week: pd.DataFrame,
    feature_cols: list[str],
) -> pd.DataFrame:
    """
    Predict 5 weeks ahead for each region using autoregressive strategy.

    Week 1: use last test week features + lag scores from train end
    Week 2: shift lag scores — lag_score_1 = pred_week1, others shift forward
    Week 3..5: repeat

    Why autoregressive:
    - Autocorrelation at lag-1 = 0.936 — previous score is the strongest signal
    - We must feed each prediction back as input for the next week
    - Error accumulates slightly over 5 weeks, but the lag-5 autocorrelation
      is still 0.679 so the signal remains meaningful

    Returns DataFrame: region_id, pred_week1 ... pred_week5
    """
    logger.info("Generating predictions (autoregressive, 5 weeks)...")

    all_predictions = []
    regions = test_last_week[REGION_COL].unique()

    for region in regions:
        row = test_last_week[
            test_last_week[REGION_COL] == region
        ].iloc[0].copy()

        preds = []
        for week in range(1, PRED_WEEKS + 1):

            # Build feature vector — cast to float32 to satisfy LightGBM
            feat_vec = (row[feature_cols]
                        .values
                        .reshape(1, -1)
                        .astype(np.float32))
            feat_df  = pd.DataFrame(feat_vec, columns=feature_cols,
                                    dtype=np.float32)

            # Predict and clip to valid score range [0, 5]
            pred = float(np.clip(model.predict(feat_df)[0], 0, 5))
            preds.append(pred)

            # Autoregressive update of lag scores:
            # lag_score_4 <- lag_score_3
            # lag_score_3 <- lag_score_2
            # lag_score_2 <- lag_score_1
            # lag_score_1 <- current prediction
            for lag in sorted(LAG_WEEKS, reverse=True):
                if lag == 1:
                    row["lag_score_1"] = np.float32(pred)
                elif f"lag_score_{lag - 1}" in row.index:
                    row[f"lag_score_{lag}"] = row[f"lag_score_{lag - 1}"]

        all_predictions.append({
            REGION_COL: region,
            **dict(zip(PRED_COLS, preds))
        })

    result = pd.DataFrame(all_predictions)
    logger.info(f"Predictions generated for {len(regions)} regions.")
    return result


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

    # ── 3. Build features (same pipeline as train.py) ─────────────────────────
    _, test_weekly = build_features(train_raw, test_raw)

    # ── 4. Use last test week as prediction input ─────────────────────────────
    test_last_week = get_last_test_week(test_weekly)

    # ── 5. Load model and feature list ───────────────────────────────────────
    model_path   = MODEL_DIR / "lgbm_model.pkl"
    feature_path = MODEL_DIR / "feature_cols.json"

    logger.info(f"Loading model from {model_path}...")
    model = joblib.load(model_path)

    with open(feature_path, "r") as f:
        feature_cols = json.load(f)
    logger.info(f"Feature list loaded: {len(feature_cols)} features")

    # Ensure all expected feature columns exist — fill missing with 0
    for col in feature_cols:
        if col not in test_last_week.columns:
            logger.warning(f"Feature '{col}' missing in test — filling with 0")
            test_last_week[col] = np.float32(0.0)

    # Cast all feature columns to float32
    test_last_week[feature_cols] = (test_last_week[feature_cols]
                                    .apply(pd.to_numeric, errors="coerce")
                                    .astype(np.float32))

    # ── 6. Generate autoregressive predictions ────────────────────────────────
    predictions = predict_autoregressive(model, test_last_week, feature_cols)

    # ── 7. Format submission to match sample_submission.csv ───────────────────
    logger.info("Formatting submission...")
    sample_sub = pd.read_csv(SAMPLE_SUB_PATH)

    # Merge onto sample submission to guarantee correct region order
    submission = sample_sub[[REGION_COL]].merge(
        predictions, on=REGION_COL, how="left"
    )

    # Fill any unmatched regions with 0 (safety fallback)
    submission[PRED_COLS] = submission[PRED_COLS].fillna(0.0)

    # Round to 4 decimal places
    submission[PRED_COLS] = submission[PRED_COLS].round(4)

    # ── 8. Save with model name in filename for version tracking ──────────────
    model_name = model_path.stem   # e.g. "lgbm_model"
    out_path   = SUBMISSION_DIR / f"submission_{model_name}_v2.csv"
    submission.to_csv(out_path, index=False)

    logger.info(f"Submission saved : {out_path}")
    logger.info(f"Shape            : {submission.shape}")
    logger.info(f"Prediction stats :")
    for col in PRED_COLS:
        logger.info(f"  {col}: mean={submission[col].mean():.3f}, "
                    f"std={submission[col].std():.3f}, "
                    f"min={submission[col].min():.3f}, "
                    f"max={submission[col].max():.3f}")
    logger.info(f"Preview:\n{submission.head(10).to_string()}")
    logger.info("Prediction complete.")


if __name__ == "__main__":
    main()