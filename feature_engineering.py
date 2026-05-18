# feature_engineering.py — Build all features for train and test
#
# Pipeline:
#   raw daily data
#   -> extract date parts
#   -> aggregate to weekly level
#   -> add lag score features
#   -> add rolling window features
#   -> add region baseline feature
#   -> return feature matrix ready for LightGBM

import pandas as pd
import numpy as np
from utils import extract_date_parts, compute_region_baseline, get_last_known_scores_per_lag, get_logger
from config import METEO_COLS, WEEKLY_AGG, LAG_WEEKS, ROLLING_WINDOWS, ROLLING_FEATURES, TARGET_COL, REGION_COL

logger = get_logger("feature_engineering")


# ── Step 1: Aggregate daily data to weekly ────────────────────────────────────
def aggregate_to_weekly(df: pd.DataFrame) -> pd.DataFrame:
    """
    Aggregate daily rows into weekly rows per region.
    Week boundary: every 7 rows starting from the first row of each region
    (consistent with EDA finding that score gaps are exactly 7 days).

    Returns one row per scored week with aggregated meteorological features.
    """
    logger.info("Aggregating daily data to weekly level...")

    df = df.sort_values([REGION_COL, "jdn"]).reset_index(drop=True)

    # Assign week_id within each region using row position
    df["week_id"] = df.groupby(REGION_COL).cumcount() // 7

    # Build aggregation dict from config
    agg_dict = {}
    for out_col, (src_col, func) in WEEKLY_AGG.items():
        agg_dict[src_col] = agg_dict.get(src_col, {})

    # Build proper groupby agg — multiple aggs per source column
    src_to_outputs = {}
    for out_col, (src_col, func) in WEEKLY_AGG.items():
        if src_col not in src_to_outputs:
            src_to_outputs[src_col] = []
        src_to_outputs[src_col].append((out_col, func))

    # Aggregate each source column
    weekly_parts = []

    # Always include: region_id, week_id, month (from scored row), score
    base = (df.groupby([REGION_COL, "week_id"])
            .agg(
                month  = ("month", "last"),   # month of the scored day
                score  = (TARGET_COL, "first") if TARGET_COL in df.columns else ("month", "last"),
            )
            .reset_index())

    # Fix score column if test data (no score col)
    if TARGET_COL not in df.columns:
        base = (df.groupby([REGION_COL, "week_id"])
                .agg(month=("month", "last"))
                .reset_index())
    else:
        base = (df.groupby([REGION_COL, "week_id"])
                .agg(
                    month = ("month", "last"),
                    score = (TARGET_COL, "first"),
                )
                .reset_index())

    weekly_parts.append(base)

    # Aggregate meteorological features
    for src_col, outputs in src_to_outputs.items():
        if src_col not in df.columns:
            continue
        agg_funcs = {out_col: (src_col, func) for out_col, func in outputs}
        part = (df.groupby([REGION_COL, "week_id"])
                .agg(**agg_funcs)
                .reset_index())
        weekly_parts.append(part.drop(columns=[REGION_COL, "week_id"]))

    weekly = pd.concat(weekly_parts, axis=1)

    # Keep only scored weeks (score is not NaN) for train; keep all for test
    if TARGET_COL in df.columns:
        weekly = weekly.dropna(subset=[TARGET_COL]).reset_index(drop=True)

    logger.info(f"Weekly aggregation done: {weekly.shape[0]} rows, {weekly.shape[1]} cols")
    return weekly


# ── Step 2: Add lag score features ───────────────────────────────────────────
def add_lag_features(
    weekly_train: pd.DataFrame,
    weekly_test: pd.DataFrame,
    last_scores_df: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Add lag score features to weekly train and test dataframes.

    For train: shift score within each region's weekly series.
    For test:  use last known scores from training data (from last_scores_df).

    last_scores_df: output of get_last_known_scores_per_lag() from utils.py
                    indexed by region_id, columns: lag_score_1, lag_score_2, ...
    """
    logger.info("Adding lag score features...")

    # ── Train lag features ────────────────────────────────────────────────────
    weekly_train = weekly_train.sort_values([REGION_COL, "week_id"]).copy()
    for lag in LAG_WEEKS:
        weekly_train[f"lag_score_{lag}"] = (
            weekly_train.groupby(REGION_COL)[TARGET_COL]
            .shift(lag)
        )

    # ── Test lag features ─────────────────────────────────────────────────────
    # Test has no score column — use the last N scores from train
    weekly_test = weekly_test.copy()
    weekly_test[REGION_COL] = weekly_test[REGION_COL].astype(str)
    weekly_test = weekly_test.merge(
        last_scores_df,
        on=REGION_COL,
        how="left"
    )

    logger.info(f"Lag features added: lag_score_1 ... lag_score_{max(LAG_WEEKS)}")
    return weekly_train, weekly_test


# ── Step 3: Add rolling window features ──────────────────────────────────────
def add_rolling_features(weekly: pd.DataFrame) -> pd.DataFrame:
    """
    Add rolling mean features over multiple window sizes.
    Computed within each region's weekly series.
    Windows defined in config.ROLLING_WINDOWS (in weeks).
    Features to roll defined in config.ROLLING_FEATURES.
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
            )

    logger.info(f"Rolling features added: {len(ROLLING_FEATURES)} features x {len(ROLLING_WINDOWS)} windows")
    return weekly


# ── Step 4: Add region baseline ───────────────────────────────────────────────
def add_region_baseline(
    weekly: pd.DataFrame,
    region_baseline: pd.Series,
) -> pd.DataFrame:
    """
    Merge per-region historical mean score as a feature.
    region_baseline: Series indexed by region_id with name 'region_mean_score'.
    """
    logger.info("Adding region baseline feature...")
    weekly = weekly.merge(region_baseline, on=REGION_COL, how="left")
    return weekly


# ── Master pipeline ───────────────────────────────────────────────────────────
def build_features(
    train_raw: pd.DataFrame,
    test_raw: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Full feature engineering pipeline.

    Args:
        train_raw: raw train.csv loaded as DataFrame (with date parsed)
        test_raw:  raw test.csv loaded as DataFrame (with date parsed)

    Returns:
        (train_weekly, test_weekly): feature matrices ready for LightGBM
        train_weekly includes TARGET_COL (score)
        test_weekly  does not
    """
    logger.info("=" * 50)
    logger.info("Starting feature engineering pipeline")
    logger.info("=" * 50)

    # Step 1: aggregate to weekly
    train_weekly = aggregate_to_weekly(train_raw)
    test_weekly  = aggregate_to_weekly(test_raw)

    # Step 2: compute region-level stats from training data only
    region_baseline  = compute_region_baseline(train_raw)
    last_scores_df   = get_last_known_scores_per_lag(train_raw, LAG_WEEKS)
    last_scores_df   = last_scores_df.reset_index()
    last_scores_df.columns = [REGION_COL] + [f"lag_score_{l}" for l in LAG_WEEKS]
    last_scores_df[REGION_COL] = last_scores_df[REGION_COL].astype(str)

    # Step 3: lag features
    train_weekly, test_weekly = add_lag_features(
        train_weekly, test_weekly, last_scores_df
    )

    # Step 4: rolling features
    train_weekly = add_rolling_features(train_weekly)
    test_weekly  = add_rolling_features(test_weekly)

    # Step 5: region baseline
    train_weekly = add_region_baseline(train_weekly, region_baseline)
    test_weekly  = add_region_baseline(test_weekly,  region_baseline)

    # Drop rows where all lag features are NaN (first few weeks of train)
    lag_cols = [f"lag_score_{l}" for l in LAG_WEEKS]
    train_weekly = train_weekly.dropna(subset=[lag_cols[0]]).reset_index(drop=True)

    logger.info(f"Train features shape : {train_weekly.shape}")
    logger.info(f"Test  features shape : {test_weekly.shape}")
    logger.info("Feature engineering complete.")
    logger.info("=" * 50)

    return train_weekly, test_weekly


# ── Feature list ──────────────────────────────────────────────────────────────
def get_feature_cols(df: pd.DataFrame) -> list[str]:
    """
    Return the list of feature columns to feed into LightGBM.
    Excludes region_id, week_id, score, and any raw date columns.
    """
    exclude = {REGION_COL, "week_id", TARGET_COL, "date", "year",
               "day", "jdn", "day_of_week"}
    return [c for c in df.columns if c not in exclude]