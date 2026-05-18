# utils.py — Shared helper functions used across all scripts

import pandas as pd
import numpy as np
import logging
import sys
from pathlib import Path


# ── Logging ───────────────────────────────────────────────────────────────────
def get_logger(name: str = "dm") -> logging.Logger:
    """Return a logger that prints to stdout with timestamp."""
    logger = logging.getLogger(name)
    if not logger.handlers:
        handler = logging.StreamHandler(sys.stdout)
        handler.setFormatter(logging.Formatter(
            "[%(asctime)s] %(levelname)s — %(message)s",
            datefmt="%H:%M:%S"
        ))
        logger.addHandler(handler)
    logger.setLevel(logging.INFO)
    return logger


# ── Date parsing ──────────────────────────────────────────────────────────────
def extract_date_parts(df: pd.DataFrame) -> pd.DataFrame:
    """
    Parse date parts from string 'YYYY-MM-DD'.
    Years like 3004-58061 are outside pandas datetime64 range (1678-2262),
    so we extract manually and compute Julian Day Number for day_of_week.

    Adds columns: year, month, day, jdn, day_of_week
    """
    parts = df["date"].str.split("-", expand=True).astype(int)
    parts.columns = ["year", "month", "day"]
    df = df.copy()
    df["year"]  = parts["year"]
    df["month"] = parts["month"]
    df["day"]   = parts["day"]

    # Julian Day Number — works for any year magnitude
    y = parts["year"].copy()
    m = parts["month"].copy()
    d = parts["day"].copy()
    mask = m <= 2
    y[mask] -= 1
    m[mask] += 12
    A   = (y / 100).astype(int)
    B   = 2 - A + (A / 4).astype(int)
    jdn = ((365.25 * (y + 4716)).astype(int)
           + (30.6001 * (m + 1)).astype(int)
           + d + B - 1524)
    df["jdn"]         = jdn
    df["day_of_week"] = jdn % 7
    return df


# ── Region-level utilities ────────────────────────────────────────────────────
def compute_region_baseline(train: pd.DataFrame) -> pd.Series:
    """
    Compute per-region historical mean score from training data.
    Returns a Series indexed by region_id.
    """
    scored = train.dropna(subset=["score"])
    return scored.groupby("region_id")["score"].mean().rename("region_mean_score")


def get_last_known_score(train: pd.DataFrame) -> pd.Series:
    """
    For each region, get the last non-NaN score in training data.
    Returns a Series indexed by region_id.
    """
    scored = train.dropna(subset=["score"])
    last   = scored.sort_values("jdn").groupby("region_id")["score"].last()
    return last.rename("last_known_score")


def get_last_known_scores_per_lag(
    train: pd.DataFrame,
    lags: list[int]
) -> pd.DataFrame:
    """
    For each region, get the last N scored weeks to use as lag features.
    Returns a DataFrame indexed by region_id with columns lag_score_1 ... lag_score_N.

    lag_score_1 = most recent score (lag 1)
    lag_score_2 = second most recent score (lag 2)
    etc.
    """
    scored = (train.dropna(subset=["score"])
              .sort_values(["region_id", "jdn"]))

    result = {}
    for lag in lags:
        # nth-from-last score per region
        lag_scores = (scored
                      .groupby("region_id")["score"]
                      .nth(-lag)
                      .rename(f"lag_score_{lag}"))
        result[f"lag_score_{lag}"] = lag_scores

    return pd.DataFrame(result)


# ── General utilities ─────────────────────────────────────────────────────────
def reduce_mem_usage(df: pd.DataFrame, verbose: bool = True) -> pd.DataFrame:
    """
    Downcast numeric columns to reduce memory footprint.
    Useful when working with 12M row datasets.
    """
    start_mem = df.memory_usage(deep=True).sum() / 1024 ** 2
    for col in df.columns:
        col_type = df[col].dtype
        if col_type != object and str(col_type) != "category":
            c_min = df[col].min()
            c_max = df[col].max()
            if str(col_type).startswith("int"):
                if c_min >= np.iinfo(np.int8).min and c_max <= np.iinfo(np.int8).max:
                    df[col] = df[col].astype(np.int8)
                elif c_min >= np.iinfo(np.int16).min and c_max <= np.iinfo(np.int16).max:
                    df[col] = df[col].astype(np.int16)
                elif c_min >= np.iinfo(np.int32).min and c_max <= np.iinfo(np.int32).max:
                    df[col] = df[col].astype(np.int32)
            else:
                if c_min >= np.finfo(np.float32).min and c_max <= np.finfo(np.float32).max:
                    df[col] = df[col].astype(np.float32)
    end_mem = df.memory_usage(deep=True).sum() / 1024 ** 2
    if verbose:
        print(f"Memory reduced from {start_mem:.1f} MB to {end_mem:.1f} MB "
              f"({100*(start_mem-end_mem)/start_mem:.1f}% reduction)")
    return df


def mae(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Mean Absolute Error."""
    return np.mean(np.abs(y_true - y_pred))