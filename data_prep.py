# data_prep.py — V4 (The 7.5 Year Truncation)
from __future__ import annotations
import numpy as np
import pandas as pd
from config_tft import (REGION_COL, TARGET_COL, WEEKLY_AGG_SPEC, WEEKLY_FEATURES, COMBINED_FRAME_CACHE_STEM)
from utils_tft import extract_date_parts, get_logger

log = get_logger("data_prep")

def _aggregate_daily_to_weekly(daily: pd.DataFrame, is_train: bool) -> pd.DataFrame:
    daily = daily.sort_values([REGION_COL, "jdn"]).reset_index(drop=True)
    daily["week_id"] = daily.groupby(REGION_COL).cumcount() // 7

    agg_dict = {out_col: (src_col, func) for out_col, (src_col, func) in WEEKLY_AGG_SPEC.items()}
    agg_dict["month"] = ("month", "last")
    if is_train:
        agg_dict["score"] = (TARGET_COL, "min")

    weekly = daily.groupby([REGION_COL, "week_id"], sort=True).agg(**agg_dict).reset_index()

    for c in WEEKLY_FEATURES:
        if c in weekly.columns and not c.startswith("delta_"):
            weekly[c] = weekly[c].astype(np.float32)
    weekly["month"] = weekly["month"].astype(np.int8)

    weekly = weekly.sort_values([REGION_COL, "week_id"])
    
    # 1st ORDER DIFF ONLY
    if "prec_sum" in weekly.columns:
        weekly["delta_prec"] = weekly.groupby(REGION_COL)["prec_sum"].diff().fillna(0).astype(np.float32)
    if "tmp_range_mean" in weekly.columns:
        weekly["delta_tmp_range"] = weekly.groupby(REGION_COL)["tmp_range_mean"].diff().fillna(0).astype(np.float32)

    return weekly

def _add_calendar_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    radians = (df["month"].astype(np.float32) - 1.0) * (2.0 * np.pi / 12.0)
    df["month_sin"] = np.sin(radians).astype(np.float32)
    df["month_cos"] = np.cos(radians).astype(np.float32)
    return df

def build_combined_frame(train_raw: pd.DataFrame, test_raw: pd.DataFrame) -> pd.DataFrame:
    train_raw = extract_date_parts(train_raw)
    test_raw = extract_date_parts(test_raw)

    weekly_train = _aggregate_daily_to_weekly(train_raw, is_train=True)
    weekly_test = _aggregate_daily_to_weekly(test_raw, is_train=False)

    # ---> THE GOLDEN TRUNCATION <---
    weekly_train = weekly_train.groupby(REGION_COL).tail(391).reset_index(drop=True)

    train_last_week = weekly_train.groupby(REGION_COL)["week_id"].max().rename("train_last_week")
    weekly_test = weekly_test.merge(train_last_week, on=REGION_COL, how="left")
    weekly_test["week_id"] = (weekly_test["week_id"].astype(np.int64) + weekly_test["train_last_week"].astype(np.int64) + 1).astype(np.int32)
    weekly_test = weekly_test.drop(columns=["train_last_week"])

    weekly_train["is_test"] = np.int8(0)
    weekly_test["is_test"] = np.int8(1)
    weekly_test["score"] = np.float32("nan")

    last_test_month = weekly_test.sort_values([REGION_COL, "week_id"]).groupby(REGION_COL).agg(last_week_id=("week_id", "max"), last_month=("month", "last")).reset_index()

    future_rows = []
    for _, row in last_test_month.iterrows():
        for offset in range(1, 6):
            future_rows.append({
                REGION_COL: row[REGION_COL],
                "week_id": int(row["last_week_id"]) + offset,
                "month": int(row["last_month"]),
                "is_test": np.int8(2),
                "score": np.float32("nan"),
            })
    weekly_future = pd.DataFrame(future_rows)

    weekly_future = weekly_future.sort_values([REGION_COL, "week_id"]).reset_index(drop=True)
    weekly_future["offset_in_future"] = weekly_future.groupby(REGION_COL).cumcount() + 1
    month_shift = ((weekly_future["offset_in_future"] - 1) // 4).astype(int)
    weekly_future["month"] = ((weekly_future["month"] - 1 + month_shift) % 12 + 1).astype(np.int8)
    weekly_future = weekly_future.drop(columns=["offset_in_future"])

    for c in WEEKLY_FEATURES:
        weekly_future[c] = np.float32("nan")

    combined = pd.concat([weekly_train, weekly_test, weekly_future], ignore_index=True)
    combined = combined.sort_values([REGION_COL, "week_id"]).reset_index(drop=True)
    combined["time_idx"] = combined.groupby(REGION_COL).cumcount().astype(np.int32)
    combined = combined.drop(columns=["week_id"])

    combined["score_known"] = (~combined["score"].isna()).astype(np.int8)
    region_mean = combined.groupby(REGION_COL)["score"].transform("mean")
    combined["score"] = combined["score"].fillna(region_mean).astype(np.float32)
    combined["score"] = combined["score"].fillna(0.0).astype(np.float32)

    # Transform the target to log(y+1) before the model ever sees it
    combined["score"] = np.log1p(combined["score"]).astype(np.float32)

    combined = _add_calendar_features(combined)

    for c in WEEKLY_FEATURES:
        combined[c] = combined.groupby(REGION_COL)[c].transform(lambda s: s.ffill().bfill()).astype(np.float32)

    combined["summer_weight"] = np.where(combined["month"].astype(int).isin([5, 6, 7, 8, 9, 10]), 3.0, 1.0).astype(np.float32)
    combined[REGION_COL] = combined[REGION_COL].astype(str)
    combined["month"] = combined["month"].astype(str)

    active_cols = ["region_id", "time_idx", "score", "score_known", "is_test", "month", "month_sin", "month_cos", "summer_weight"] + WEEKLY_FEATURES
    combined = combined[[c for c in active_cols if c in combined.columns]]

    return combined

def _pick_cache_path():
    try:
        import pyarrow 
        return COMBINED_FRAME_CACHE_STEM.with_suffix(".parquet"), "parquet"
    except ImportError:
        return COMBINED_FRAME_CACHE_STEM.with_suffix(".pkl"), "pickle"

def prepare(force_rebuild: bool = False) -> pd.DataFrame:
    from config_tft import TRAIN_PATH, TEST_PATH
    cache_path, cache_kind = _pick_cache_path()
    
    if cache_path.exists() and not force_rebuild:
        if cache_kind == "parquet": return pd.read_parquet(cache_path)
        else: return pd.read_pickle(cache_path)

    train_raw = pd.read_csv(TRAIN_PATH)
    test_raw = pd.read_csv(TEST_PATH)
    combined = build_combined_frame(train_raw, test_raw)

    if cache_kind == "parquet": combined.to_parquet(cache_path, index=False)
    else: combined.to_pickle(cache_path)
    return combined