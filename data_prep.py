"""
data_prep.py — Convert raw daily CSVs into a single weekly long frame
that pytorch-forecasting's TimeSeriesDataSet can consume.

Why a single combined frame?
    pytorch-forecasting expects ONE dataframe whose `time_idx` is contiguous
    per group. We append the 13 test weeks as future rows (score=NaN) to each
    region's train history, with a continuous time_idx. At inference time
    we ask TFT to predict the last 5 weeks of each region's frame.

    Critically, the gap between train end and test start (58–604 days per EDA)
    is collapsed: we do NOT preserve calendar time, only relative ordering.
    This is intentional. The model uses month/sin-cos for seasonality, not
    absolute time.

Output schema (per row, one row per (region, week)):
    region_id          str       — group id
    time_idx           int64     — 0 ... N-1, continuous per region
    score              float32   — target (NaN for test weeks)
    score_known        int8      — 1 if score is known, 0 otherwise (encoder mask)
    is_test            int8      — 1 if this row is one of the 13 test weeks
    month              int8      — 1-12, used for seasonality
    month_sin, month_cos float32 — cyclical encoding of month
    {weekly_features}  float32   — 13 weather aggregates from WEEKLY_FEATURES
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from config_tft import (
    REGION_COL,
    TARGET_COL,
    WEEKLY_AGG_SPEC,
    WEEKLY_FEATURES,
    COMBINED_FRAME_CACHE_STEM,
)
from utils_tft import extract_date_parts, get_logger

log = get_logger("data_prep")


# ─────────────────────────────────────────────────────────────────────────────
#  Daily → weekly aggregation
# ─────────────────────────────────────────────────────────────────────────────
def _aggregate_daily_to_weekly(daily: pd.DataFrame, is_train: bool) -> pd.DataFrame:
    """
    Aggregate 7-day chunks per region into one weekly row.
    """
    daily = daily.sort_values([REGION_COL, "jdn"]).reset_index(drop=True)
    daily["week_id"] = daily.groupby(REGION_COL).cumcount() // 7

    agg_dict = {}
    for out_col, (src_col, func) in WEEKLY_AGG_SPEC.items():
        agg_dict[out_col] = (src_col, func)
    agg_dict["month"] = ("month", "last") 
    if is_train:
        agg_dict["score"] = (TARGET_COL, "min")

    weekly = (
        daily.groupby([REGION_COL, "week_id"], sort=True)
        .agg(**agg_dict)
        .reset_index()
    )

    for c in WEEKLY_FEATURES:
        if c in weekly.columns and not c.startswith("delta_") and not "ema" in c and not "std" in c and not "seasonal" in c:
            weekly[c] = weekly[c].astype(np.float32)
    weekly["month"] = weekly["month"].astype(np.int8)

    # ---> ADD ADVANCED DELTAS, VOLATILITY, AND EMA HERE <---
    weekly = weekly.sort_values([REGION_COL, "week_id"])
    
    # 1. Exponential Moving Average (EMA) - 4 week span
    if "prec_sum" in weekly.columns:
        weekly["prec_ema_4w"] = weekly.groupby(REGION_COL)["prec_sum"].transform(
            lambda x: x.ewm(span=4, adjust=False).mean()
        ).astype(np.float32)
    if "tmp_range_mean" in weekly.columns:
        weekly["tmp_range_ema_4w"] = weekly.groupby(REGION_COL)["tmp_range_mean"].transform(
            lambda x: x.ewm(span=4, adjust=False).mean()
        ).astype(np.float32)

    # 2. Rolling Volatility (Standard Deviation) - 4 week window
    if "prec_sum" in weekly.columns:
        weekly["prec_std_4w"] = weekly.groupby(REGION_COL)["prec_sum"].transform(
            lambda x: x.rolling(4, min_periods=1).std().fillna(0)
        ).astype(np.float32)
    if "tmp_range_mean" in weekly.columns:
        weekly["tmp_range_std_4w"] = weekly.groupby(REGION_COL)["tmp_range_mean"].transform(
            lambda x: x.rolling(4, min_periods=1).std().fillna(0)
        ).astype(np.float32)

    # 3. Differencing (1st, 2nd, and 52-week Seasonal)
    if "prec_sum" in weekly.columns:
        weekly["delta_prec"] = weekly.groupby(REGION_COL)["prec_sum"].diff().fillna(0).astype(np.float32)
        weekly["delta2_prec"] = weekly.groupby(REGION_COL)["delta_prec"].diff().fillna(0).astype(np.float32)
        weekly["seasonal_diff_prec"] = weekly.groupby(REGION_COL)["prec_sum"].diff(52).fillna(0).astype(np.float32)

    if "tmp_range_mean" in weekly.columns:
        weekly["delta_tmp_range"] = weekly.groupby(REGION_COL)["tmp_range_mean"].diff().fillna(0).astype(np.float32)
        weekly["delta2_tmp_range"] = weekly.groupby(REGION_COL)["delta_tmp_range"].diff().fillna(0).astype(np.float32)
        weekly["seasonal_diff_tmp_range"] = weekly.groupby(REGION_COL)["tmp_range_mean"].diff(52).fillna(0).astype(np.float32)

    log.info(
        "Aggregated %s to weekly: %d rows, %d regions",
        "train" if is_train else "test",
        len(weekly),
        weekly[REGION_COL].nunique(),
    )
    return weekly


# ─────────────────────────────────────────────────────────────────────────────
#  Build the combined long frame
# ─────────────────────────────────────────────────────────────────────────────
def _add_calendar_features(df: pd.DataFrame) -> pd.DataFrame:
    """Cyclical month encoding so TFT sees seasonality as continuous."""
    df = df.copy()
    radians = (df["month"].astype(np.float32) - 1.0) * (2.0 * np.pi / 12.0)
    df["month_sin"] = np.sin(radians).astype(np.float32)
    df["month_cos"] = np.cos(radians).astype(np.float32)
    return df


def _fill_missing_weekly_scores(weekly_train: pd.DataFrame) -> pd.DataFrame:
    """
    EDA found exactly 1 missing week per region (the partial Dec-31 start).
    Forward-fill from the next available week so we don't need to drop them.
    This affects the encoder mask, not training targets.
    """
    weekly_train = weekly_train.sort_values([REGION_COL, "week_id"]).copy()
    # We don't fill 'score' — leave it NaN if missing so the mask stays correct.
    return weekly_train


def build_combined_frame(
    train_raw: pd.DataFrame,
    test_raw: pd.DataFrame,
) -> pd.DataFrame:
    """
    Produce one long-format dataframe with both train and test weeks per region.

    Layout per region:
        time_idx 0 .. N_train-1        : train weeks   (score possibly known)
        time_idx N_train .. N_train+12 : test  weeks   (score always NaN)

    The decoder will be asked to predict the LAST 5 of these (weeks 8-12 of
    the test window, i.e. the future from the perspective of week 8 of test).
    """
    log.info("Parsing date parts...")
    train_raw = extract_date_parts(train_raw)
    test_raw = extract_date_parts(test_raw)

    log.info("Aggregating daily → weekly...")
    weekly_train = _aggregate_daily_to_weekly(train_raw, is_train=True)
    weekly_test = _aggregate_daily_to_weekly(test_raw, is_train=False)

    # ---> PROPER DATA TRUNCATION <---
    # log.info("Applying 50% Truncation: Keeping last 391 weeks of train data...")
    weekly_train = weekly_train.groupby(REGION_COL).tail(391).reset_index(drop=True)

    weekly_train = _fill_missing_weekly_scores(weekly_train)

    # ── Offset test week_ids to come AFTER train week_ids per region ─────────
    train_last_week = (
        weekly_train.groupby(REGION_COL)["week_id"].max().rename("train_last_week")
    )
    weekly_test = weekly_test.merge(train_last_week, on=REGION_COL, how="left")
    weekly_test["week_id"] = (
        weekly_test["week_id"].astype(np.int64) + weekly_test["train_last_week"].astype(np.int64) + 1
    ).astype(np.int32)
    weekly_test = weekly_test.drop(columns=["train_last_week"])

    # Tag origin
    weekly_train["is_test"] = np.int8(0)
    weekly_test["is_test"] = np.int8(1)
    weekly_test["score"] = np.float32("nan")

    # ── Build 5 synthetic future weeks per region ────────────────────────────
    log.info("Building 5 synthetic future weeks per region...")
    last_test_month = (
        weekly_test.sort_values([REGION_COL, "week_id"])
        .groupby(REGION_COL)
        .agg(last_week_id=("week_id", "max"), last_month=("month", "last"))
        .reset_index()
    )

    future_rows = []
    for _, row in last_test_month.iterrows():
        for offset in range(1, 6):
            future_rows.append({
                REGION_COL: row[REGION_COL],
                "week_id": int(row["last_week_id"]) + offset,
                "month": int(row["last_month"]),  # approximate; refined below
                "is_test": np.int8(2),            # 2 = future (prediction target)
                "score": np.float32("nan"),
            })
    weekly_future = pd.DataFrame(future_rows)

    # Refine `month` for future weeks: 4 weeks ≈ 1 month
    weekly_future = weekly_future.sort_values([REGION_COL, "week_id"]).reset_index(drop=True)
    weekly_future["offset_in_future"] = weekly_future.groupby(REGION_COL).cumcount() + 1
    month_shift = ((weekly_future["offset_in_future"] - 1) // 4).astype(int)
    weekly_future["month"] = ((weekly_future["month"] - 1 + month_shift) % 12 + 1).astype(np.int8)
    weekly_future = weekly_future.drop(columns=["offset_in_future"])

    # Weather columns NaN for future
    for c in WEEKLY_FEATURES:
        weekly_future[c] = np.float32("nan")

    # ── Concatenate train + test + future per region ─────────────────────────
    combined = pd.concat([weekly_train, weekly_test, weekly_future], ignore_index=True)
    combined = combined.sort_values([REGION_COL, "week_id"]).reset_index(drop=True)

    # Replace the per-region week_id with a contiguous time_idx
    combined["time_idx"] = combined.groupby(REGION_COL).cumcount().astype(np.int32)
    combined = combined.drop(columns=["week_id"])

    # Encoder mask: 1 if score is known, else 0
    combined["score_known"] = (~combined["score"].isna()).astype(np.int8)

    # Fill NaN score with the per-region historical mean 
    region_mean = combined.groupby(REGION_COL)["score"].transform("mean")
    combined["score"] = combined["score"].fillna(region_mean).astype(np.float32)
    combined["score"] = combined["score"].fillna(0.0).astype(np.float32)

    # Calendar features 
    combined = _add_calendar_features(combined)

    # Weather features: forward-fill within region for any NaN in future rows.
    for c in WEEKLY_FEATURES:
        combined[c] = combined.groupby(REGION_COL)[c].transform(
            lambda s: s.ffill().bfill()
        ).astype(np.float32)

    # ---> ADD SUMMER WEIGHT <---
    # Forces TFT to prioritize summer weeks where drought actually occurs
    combined["summer_weight"] = np.where(
        combined["month"].astype(int).isin([5, 6, 7, 8, 9, 10]), 
        3.0, 
        1.0
    ).astype(np.float32)

    # region_id must be a string for pytorch-forecasting's categorical handling
    combined[REGION_COL] = combined[REGION_COL].astype(str)
    combined["month"] = combined["month"].astype(str)

    # ---> DROP ROGUE COLUMNS TO PREVENT NAN CRASHES <---
    active_cols = ["region_id", "time_idx", "score", "score_known", "is_test", 
                   "month", "month_sin", "month_cos", "summer_weight"] + WEEKLY_FEATURES
    combined = combined[[c for c in active_cols if c in combined.columns]]

    log.info("Combined frame: %d rows, %d regions, time_idx range %d-%d",
             len(combined), combined[REGION_COL].nunique(),
             combined["time_idx"].min(), combined["time_idx"].max())

    # Sanity checks
    assert combined.isna().sum().sum() == 0, \
        f"NaNs remain in combined frame:\n{combined.isna().sum()}"
    assert combined.groupby(REGION_COL)["time_idx"].apply(
        lambda s: (s.diff().dropna() == 1).all()
    ).all(), "time_idx is not contiguous per region"

    return combined


# ─────────────────────────────────────────────────────────────────────────────
#  Cached entry point
# ─────────────────────────────────────────────────────────────────────────────
def _pick_cache_path():
    """
    Prefer parquet (faster, columnar, smaller) when pyarrow is installed,
    otherwise fall back to pickle. Either keeps dtypes intact.
    """
    try:
        import pyarrow  # noqa: F401
        return COMBINED_FRAME_CACHE_STEM.with_suffix(".parquet"), "parquet"
    except ImportError:
        return COMBINED_FRAME_CACHE_STEM.with_suffix(".pkl"), "pickle"


def prepare(force_rebuild: bool = False) -> pd.DataFrame:
    """
    Load cached file if present, else build from CSVs.
    Returns the combined long frame.
    """
    from config_tft import TRAIN_PATH, TEST_PATH

    cache_path, cache_kind = _pick_cache_path()

    # Also accept the OTHER cache format if it happens to be on disk
    # (e.g. previously cached as parquet, now no pyarrow).
    alt = (COMBINED_FRAME_CACHE_STEM.with_suffix(".pkl")
           if cache_kind == "parquet"
           else COMBINED_FRAME_CACHE_STEM.with_suffix(".parquet"))

    for candidate, kind in [(cache_path, cache_kind), (alt, "parquet" if cache_kind == "pickle" else "pickle")]:
        if candidate.exists() and not force_rebuild:
            log.info("Loading cached combined frame from %s", candidate)
            try:
                if kind == "parquet":
                    return pd.read_parquet(candidate)
                else:
                    return pd.read_pickle(candidate)
            except Exception as e:
                log.warning("Failed to read %s (%s) — rebuilding", candidate, e)

    log.info("Building combined frame from raw CSVs...")
    train_raw = pd.read_csv(TRAIN_PATH)
    test_raw = pd.read_csv(TEST_PATH)
    log.info("Train raw: %s | Test raw: %s", train_raw.shape, test_raw.shape)

    combined = build_combined_frame(train_raw, test_raw)

    log.info("Caching combined frame to %s (%s)", cache_path, cache_kind)
    if cache_kind == "parquet":
        combined.to_parquet(cache_path, index=False)
    else:
        combined.to_pickle(cache_path)
    return combined


if __name__ == "__main__":
    # Standalone build / sanity check
    combined = prepare(force_rebuild=True)
    print(combined.head(10))
    print("Schema:")
    print(combined.dtypes)
    print(f"\nTotal rows: {len(combined):,}")
    print(f"Per-region row count head:\n{combined.groupby('region_id').size().head()}")