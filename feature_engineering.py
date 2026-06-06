# feature_engineering.py — Build all features for train and test

import pandas as pd
import numpy as np
from utils import extract_date_parts, compute_region_baseline, get_logger
from config import (
    METEO_COLS, WEEKLY_AGG, LAG_WEEKS, ROLLING_WINDOWS,
    ROLLING_FEATURES, TARGET_COL, REGION_COL
)

logger = get_logger("feature_engineering")


# ── Step 1: Aggregate daily → weekly ─────────────────────────────────────────
def aggregate_to_weekly(df: pd.DataFrame, is_train: bool = True) -> pd.DataFrame:
    """
        Aggregate daily rows into weekly rows per region.
        Week boundary: every 7 consecutive rows from the start of each region.

        is_train=True  -> keep only weeks with score (dropna score)
        is_train=False -> keep all 13 test weeks
    """
    logger.info("Aggregating daily data to weekly level...")

    df = df.sort_values([REGION_COL, "jdn"]).reset_index(drop=True)
    df["week_id"] = df.groupby(REGION_COL).cumcount() // 7

    # Build aggregation dict from config
    src_to_outputs = {}
    for out_col, (src_col, func) in WEEKLY_AGG.items():
        if src_col not in src_to_outputs:
            src_to_outputs[src_col] = []
        src_to_outputs[src_col].append((out_col, func))

    # Base columns: region_id, week_id, month
    if is_train:
        base = (df.groupby([REGION_COL, "week_id"])
                .agg(month=(  "month",     "last"),
                     score=(TARGET_COL,   "first"))
                .reset_index())
    else:
        base = (df.groupby([REGION_COL, "week_id"])
                .agg(month=("month", "last"))
                .reset_index())

    # Aggregate meteorological features
    parts = [base]
    for src_col, outputs in src_to_outputs.items():
        if src_col not in df.columns:
            continue
        agg_funcs = {out_col: (src_col, func) for out_col, func in outputs}
        part = (df.groupby([REGION_COL, "week_id"])
                .agg(**agg_funcs)
                .reset_index()
                .drop(columns=[REGION_COL, "week_id"]))
        parts.append(part)

    weekly = pd.concat(parts, axis=1)

    if is_train and TARGET_COL in weekly.columns:
        weekly = weekly.dropna(subset=[TARGET_COL]).reset_index(drop=True)

    feat_cols = [c for c in weekly.columns
                 if c not in [REGION_COL, "week_id", TARGET_COL]]
    weekly[feat_cols] = weekly[feat_cols].apply(
        pd.to_numeric, errors="coerce"
    ).astype(np.float32)

    logger.info(f"Weekly aggregation done: {weekly.shape[0]} rows, {weekly.shape[1]} cols")
    return weekly


# ── Step 2: Add lag score features ─────────────────────
def add_lag_features_train(weekly: pd.DataFrame) -> pd.DataFrame:
    """
        Add lag score features for train data.
        lag_score_1 = score of the immediately preceding week (T-1)
        lag_score_2 = score of 2 weeks ago (T-2), etc.

        These are the most important features — autocorrelation at lag-1 = 0.936.
        Rows with lag_score_1 = NaN (first week of each region) are dropped.
    """
    logger.info("Adding lag score features (train)...")
    weekly = weekly.sort_values([REGION_COL, "week_id"]).copy()

    for lag in LAG_WEEKS:
        weekly[f"lag_score_{lag}"] = (
            weekly.groupby(REGION_COL)[TARGET_COL]
            .shift(lag)
            .astype(np.float32)
        )

    before = len(weekly)
    weekly = weekly.dropna(subset=["lag_score_1"]).reset_index(drop=True)
    logger.info(f"Dropped {before - len(weekly)} rows with missing lag_score_1")
    return weekly


def add_lag_features_test(
    weekly_test: pd.DataFrame,
    weekly_train: pd.DataFrame,
) -> pd.DataFrame:
    """
    Add lag score features for test data.

    Problem: test has no score column, and the gap between train end
    and test start averages ~384 days -> lag scores from train are stale.

    Solution: use the last 4 scores from train as lag features.
    This is best effort — imperfect but better than nothing.
    The model was trained on truly sequential lag scores, so it understands
    lag_score_1 as the most recently known score.
    """
    logger.info("Adding lag features (test) from end of train...")

    scored_train = (weekly_train
                    .dropna(subset=[TARGET_COL])
                    .sort_values([REGION_COL, "week_id"])
                    [[REGION_COL, "week_id", TARGET_COL]]
                    .copy())

    # Build lag scores using tail() per region — avoids nth() index issues
    last_scores_df = scored_train.groupby(REGION_COL).tail(max(LAG_WEEKS))

    # Assign lag rank within each region (1 = most recent)
    last_scores_df = last_scores_df.copy()
    last_scores_df["lag_rank"] = (last_scores_df
                                  .groupby(REGION_COL)
                                  .cumcount(ascending=False) + 1)

    # Keep only the lags we need
    last_scores_df = last_scores_df[
        last_scores_df["lag_rank"].isin(LAG_WEEKS)
    ]

    # Pivot to wide format: one column per lag
    last_scores_wide = last_scores_df.pivot_table(
        index=REGION_COL,
        columns="lag_rank",
        values=TARGET_COL,
        aggfunc="first",
    )
    last_scores_wide.columns = [f"lag_score_{int(c)}" for c in last_scores_wide.columns]
    last_scores_wide = last_scores_wide.reset_index()
    last_scores_wide[REGION_COL] = last_scores_wide[REGION_COL].astype(str)

    weekly_test = weekly_test.copy()
    weekly_test[REGION_COL] = weekly_test[REGION_COL].astype(str)
    weekly_test = weekly_test.merge(last_scores_wide, on=REGION_COL, how="left")

    # Cast lag columns to float32
    lag_cols = [f"lag_score_{l}" for l in LAG_WEEKS]
    for col in lag_cols:
        if col in weekly_test.columns:
            weekly_test[col] = weekly_test[col].astype(np.float32)

    logger.info("Lag features (test) added.")
    return weekly_test


# ── Step 3: Rolling window features ─────────────────
def add_rolling_features(weekly: pd.DataFrame) -> pd.DataFrame:
    """
        Compute rolling mean of key features over 4-week and 13-week windows.

        - Roll 4 weeks : short-term trend
        - Roll 13 weeks: long-term trend (~1 quarter)

        shift(1) before rolling to avoid data leakage — current week's
        values must not leak into the feature of that same week.
    """
    logger.info("Adding rolling window features...")
    weekly = weekly.sort_values([REGION_COL, "week_id"]).copy()

    for feat in ROLLING_FEATURES:
        if feat not in weekly.columns:
            continue
        for window in ROLLING_WINDOWS:
            col_name = f"{feat}_roll{window}w"
            weekly[col_name] = (
                weekly.groupby(REGION_COL)[feat]
                .transform(lambda x: x.shift(1).rolling(window, min_periods=1).mean())
                .astype(np.float32)
            )

    logger.info(f"Rolling features added: {len(ROLLING_FEATURES)} x {len(ROLLING_WINDOWS)} windows")
    return weekly


# ── Step 4: Region baseline ─────────────────
def add_region_baseline(
    weekly: pd.DataFrame,
    region_baseline: pd.Series,
) -> pd.DataFrame:
    """
        Merge per-region historical mean score as a feature.
        Each region has its own drought tendency (range 0.08 to 2.26) —
        this helps the model distinguish chronically dry vs wet regions.
    """
    logger.info("Adding region baseline feature...")
    weekly = weekly.merge(region_baseline, on=REGION_COL, how="left")
    if "region_mean_score" in weekly.columns:
        weekly["region_mean_score"] = weekly["region_mean_score"].astype(np.float32)
    return weekly


# ── Step 5: Seasonal weighting ───────────────────────────
def apply_seasonal_filter(
    weekly: pd.DataFrame,
    summer_months: list[int] = [5, 6, 7, 8, 9, 10],
    weight_multiplier: float = 3.0,
) -> pd.DataFrame:
    """
        Address distribution shift between train and test (EDA section 19).
        Test features are 45-70% warmer than train average — test window
        falls in summer while train covers all 4 seasons over 15 years.

        Use sample_weight to upweight summer weeks:
        - Summer weeks (months 5-10): weight = weight_multiplier (default 3x)
        - Other months             : weight = 1x

        Model pays more attention to summer-like weeks without discarding
        winter data entirely.
    """
    logger.info(f"Applying seasonal weighting: months {summer_months} x{weight_multiplier}")
    weekly = weekly.copy()
    weekly["sample_weight"] = np.where(
        weekly["month"].isin(summer_months),
        weight_multiplier,
        1.0
    ).astype(np.float32)
    summer_count = weekly["month"].isin(summer_months).sum()
    logger.info(f"Summer weeks: {summer_count:,} | Winter weeks: {len(weekly)-summer_count:,}")
    return weekly

def add_phase2_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    # Ensure it is sorted chronologically per region before doing rolling math
    df = df.sort_values(["region_id", "week_id"])
    
    # --- EXISTING FEATURES (Keep whatever was already here like month_sin/cos) ---
    if "month" in df.columns:
        radians = (df["month"].astype(np.float32) - 1.0) * (2.0 * np.pi / 12.0)
        df["month_sin"] = np.sin(radians).astype(np.float32)
        df["month_cos"] = np.cos(radians).astype(np.float32)

    # --- NEW: Exponential Moving Average (EMA) - 4 week span ---
    if "prec_sum" in df.columns:
        df["prec_ema_4w"] = df.groupby("region_id")["prec_sum"].transform(
            lambda x: x.ewm(span=4, adjust=False).mean()
        ).astype(np.float32)
    if "tmp_range_mean" in df.columns:
        df["tmp_range_ema_4w"] = df.groupby("region_id")["tmp_range_mean"].transform(
            lambda x: x.ewm(span=4, adjust=False).mean()
        ).astype(np.float32)

    # --- NEW: Rolling Volatility (Standard Deviation) - 4 week window ---
    if "prec_sum" in df.columns:
        df["prec_std_4w"] = df.groupby("region_id")["prec_sum"].transform(
            lambda x: x.rolling(4, min_periods=1).std().fillna(0)
        ).astype(np.float32)
    if "tmp_range_mean" in df.columns:
        df["tmp_range_std_4w"] = df.groupby("region_id")["tmp_range_mean"].transform(
            lambda x: x.rolling(4, min_periods=1).std().fillna(0)
        ).astype(np.float32)

    # --- NEW: Differencing (1st, 2nd, and 52-week Seasonal) ---
    if "prec_sum" in df.columns:
        df["delta_prec"] = df.groupby("region_id")["prec_sum"].diff().fillna(0).astype(np.float32)
        df["delta2_prec"] = df.groupby("region_id")["delta_prec"].diff().fillna(0).astype(np.float32)
        df["seasonal_diff_prec"] = df.groupby("region_id")["prec_sum"].diff(52).fillna(0).astype(np.float32)

    if "tmp_range_mean" in df.columns:
        df["delta_tmp_range"] = df.groupby("region_id")["tmp_range_mean"].diff().fillna(0).astype(np.float32)
        df["delta2_tmp_range"] = df.groupby("region_id")["delta_tmp_range"].diff().fillna(0).astype(np.float32)
        df["seasonal_diff_tmp_range"] = df.groupby("region_id")["tmp_range_mean"].diff(52).fillna(0).astype(np.float32)

    return df

# ── Master pipeline ───────────────────────────────────────────────────────────
def build_features(
    train_raw: pd.DataFrame,
    test_raw: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
        Full feature engineering pipeline.

        Train pipeline:
            daily -> weekly -> lag scores -> rolling -> region baseline -> seasonal weight

        Test pipeline:
            daily -> weekly (13 weeks) -> lag scores from train end
            -> rolling over 13 test weeks -> region baseline

        Returns:
            train_weekly: feature matrix with score column, ready for LightGBM
            test_weekly:  all 13 test weeks with full features;
                        predict.py will use the last week (week_id=12) as input
    """
    logger.info("=" * 50)
    logger.info("Starting feature engineering pipeline")
    logger.info("=" * 50)

    # ── Train ─────────────────────────────────────────────────────────────────
    train_weekly = aggregate_to_weekly(train_raw, is_train=True)
    train_weekly = add_lag_features_train(train_weekly)
    train_weekly = add_rolling_features(train_weekly)

    # Region baseline
    region_baseline = compute_region_baseline(train_raw)
    train_weekly    = add_region_baseline(train_weekly, region_baseline)

    train_weekly = add_phase2_features(train_weekly, train_raw)

    # Seasonal weighting 
    train_weekly = apply_seasonal_filter(train_weekly)

    # ── Test ──────────────────────────────────────────────────────────────────
    test_weekly = aggregate_to_weekly(test_raw, is_train=False)
    test_weekly = add_lag_features_test(test_weekly, train_weekly)
    test_weekly = add_rolling_features(test_weekly)
    test_weekly = add_region_baseline(test_weekly, region_baseline)

    logger.info(f"Train features shape : {train_weekly.shape}")
    logger.info(f"Test  features shape : {test_weekly.shape}")
    logger.info("Feature engineering complete.")
    logger.info("=" * 50)

    return train_weekly, test_weekly


# ── Feature list ──────────────────────────────────────────────────────────────
def get_feature_cols(df: pd.DataFrame) -> list[str]:
    """
        Return the list of feature columns to feed into LightGBM.
        Excludes non-feature columns: id, target, date columns, and weights.
    """
    exclude = {
        REGION_COL, "week_id", TARGET_COL,
        "date", "year", "month", "day", "jdn", "day_of_week",
        "sample_weight",
    }
    return [c for c in df.columns if c not in exclude]