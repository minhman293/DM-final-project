"""
utils_tft.py — Shared helpers: logging, date parsing, MAE.

Date parsing note:
    The dataset uses fictional years (3004-58061) outside pandas' datetime64
    range. We compute Julian Day Number manually to get the within-region
    ordering and the day-of-week (latter unused per EDA but kept for safety).
"""

from __future__ import annotations

import logging
import sys

import numpy as np
import pandas as pd


def get_logger(name: str = "tft") -> logging.Logger:
    """Stdout logger with timestamp; idempotent under reimport."""
    logger = logging.getLogger(name)
    if not logger.handlers:
        h = logging.StreamHandler(sys.stdout)
        h.setFormatter(logging.Formatter(
            "[%(asctime)s] %(levelname)s %(name)s — %(message)s",
            datefmt="%H:%M:%S",
        ))
        logger.addHandler(h)
    logger.setLevel(logging.INFO)
    logger.propagate = False
    return logger


def extract_date_parts(df: pd.DataFrame) -> pd.DataFrame:
    """
    Parse 'YYYY-MM-DD' strings into integer year/month/day/jdn.

    JDN (Julian Day Number) is computed from the formula that works for any
    proleptic Gregorian date — including fictional far-future years that
    pandas can't represent.

    Adds: year (int32), month (int8), day (int8), jdn (int64)
    """
    parts = df["date"].str.split("-", expand=True).astype(np.int64)
    parts.columns = ["year", "month", "day"]
    df = df.copy()
    df["year"] = parts["year"].astype(np.int32)
    df["month"] = parts["month"].astype(np.int8)
    df["day"] = parts["day"].astype(np.int8)

    y = parts["year"].copy()
    m = parts["month"].copy()
    d = parts["day"].copy()
    early = m <= 2
    y[early] -= 1
    m[early] += 12

    A = (y // 100).astype(np.int64)
    B = 2 - A + (A // 4).astype(np.int64)
    jdn = (
        (365.25 * (y + 4716)).astype(np.int64)
        + (30.6001 * (m + 1)).astype(np.int64)
        + d + B - 1524
    )
    df["jdn"] = jdn.astype(np.int64)
    return df


def mae(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Mean Absolute Error."""
    return float(np.mean(np.abs(np.asarray(y_true) - np.asarray(y_pred))))