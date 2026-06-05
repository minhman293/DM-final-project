"""
reproduce_best.py — Rebuild the best submission (ensemble_70tft_30lstm.csv).

Recipe (from ensemble.py "ver 3 / Strategy A"):
    super_tft = mean(TFT seed 42, seed 123, seed 999)   # average the 3 seeds
    final     = 0.70 * super_tft + 0.30 * LSTM_v2        # then blend with LSTM
    round to 4 decimals

This version merges on region_id instead of relying on row order, so the output
scores identically on Kaggle even if the four input CSVs are ordered differently.

Run from your repo root:   python reproduce_best.py
"""

import os
import sys
import pandas as pd

SUB = "submissions"
KEY = "region_id"
PRED = ["pred_week1", "pred_week2", "pred_week3", "pred_week4", "pred_week5"]

TFT_SEEDS = {
    42:  f"{SUB}/submission_tft_v4_relu.csv",
    123: f"{SUB}/submission_tft_v4_relu_123.csv",
    999: f"{SUB}/submission_tft_v4_relu_999.csv",
}
LSTM_PATH = f"{SUB}/submission_lstm_v2.csv"
OUT_PATH = f"{SUB}/ensemble_70tft_30lstm.csv"

# ── 1. Verify all source files exist ─────────────────────────────────────────
required = list(TFT_SEEDS.values()) + [LSTM_PATH]
missing = [p for p in required if not os.path.exists(p)]
if missing:
    print("Cannot reproduce the best blend — these source files are missing:\n")
    for m in missing:
        print("   MISSING:", m)
    print(
        "\nThe best file is 0.70 * mean(3 TFT seeds) + 0.30 * LSTM_v2.\n"
        "If any seed CSV is gone, you must retrain that piece before blending.\n"
        "(See the retrain notes: transformation='relu', DATA_DIR='data', seeds 42/123/999.)"
    )
    sys.exit(1)

# ── 2. Load, keying everything on region_id ──────────────────────────────────
def load(path, suffix):
    d = pd.read_csv(path)
    d[KEY] = d[KEY].astype(str)
    return d[[KEY] + PRED].rename(columns={c: f"{c}__{suffix}" for c in PRED})

merged = load(LSTM_PATH, "lstm")
for seed, path in TFT_SEEDS.items():
    merged = merged.merge(load(path, seed), on=KEY, how="inner")

# ── 3. Build the blend ───────────────────────────────────────────────────────
out = pd.DataFrame({KEY: merged[KEY]})
for c in PRED:
    super_tft = (merged[f"{c}__42"] + merged[f"{c}__123"] + merged[f"{c}__999"]) / 3.0
    out[c] = (0.70 * super_tft + 0.30 * merged[f"{c}__lstm"]).round(4)

# ── 4. Align to sample_submission order if present (cosmetic only) ────────────
if os.path.exists("sample_submission.csv"):
    samp = pd.read_csv("sample_submission.csv")
    samp[KEY] = samp[KEY].astype(str)
    n_before = len(out)
    out = samp[[KEY]].merge(out, on=KEY, how="left")
    if out[PRED].isna().any().any():
        print("WARNING: some sample_submission regions had no prediction after merge.")
    if len(out) != n_before:
        print(f"WARNING: row count changed on align ({n_before} -> {len(out)}); "
              "check that region sets match across all files.")

# ── 5. Write + report ────────────────────────────────────────────────────────
os.makedirs(SUB, exist_ok=True)
out.to_csv(OUT_PATH, index=False)
print(f"Wrote {OUT_PATH}  ({len(out)} rows)\n")
print("Prediction summary (sanity check vs your 0.8273 run):")
print(out[PRED].describe().round(4).to_string())