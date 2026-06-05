"""
run_all.py — End-to-end from-scratch reproduction of ensemble_70tft_30lstm.csv
              (public LB 0.8273).

Pipeline:
    1. Train 3 TFT seeds (42, 123, 999) → 3 prediction CSVs
    2. Train LSTM v2 → 1 prediction CSV
    3. Blend: 0.70 × mean(3 TFT) + 0.30 × LSTM v2

Each step is idempotent — if its output CSV already exists, it's skipped.
Safe to rerun after a partial failure.

Expected runtime on a single modern GPU: ~3–4 hours total.

Usage:
    python run_all.py
"""

from __future__ import annotations

import shutil
import subprocess
import sys
from pathlib import Path


def run(args: list[str], label: str) -> None:
    print("\n" + "=" * 70)
    print(f">>> {label}")
    print("=" * 70)
    result = subprocess.run([sys.executable] + args)
    if result.returncode != 0:
        sys.exit(f"FAILED: {label} (exit code {result.returncode})")


def main():
    # Sanity check: LSTM must point to the same data folder as TFT.
    try:
        from config_lstm import DATA_DIR as LSTM_DATA_DIR
        if str(LSTM_DATA_DIR) != "data":
            sys.exit(
                f"config_lstm.py has DATA_DIR='{LSTM_DATA_DIR}', expected 'data'.\n"
                f"The best LSTM v2 was trained on the full 'data' folder.\n"
                f"Edit config_lstm.py to set DATA_DIR = Path('data') and rerun."
            )
    except ImportError:
        pass  # config_lstm not yet on path; we'll fail later at the LSTM step

    sub = Path("submissions")
    sub.mkdir(exist_ok=True)

    # ── 1. TFT — three seeds ────────────────────────────────────────────────
    tft_files = {
        42:  sub / "submission_tft_v4_relu.csv",
        123: sub / "submission_tft_v4_relu_123.csv",
        999: sub / "submission_tft_v4_relu_999.csv",
    }
    for seed, csv in tft_files.items():
        if csv.exists():
            print(f"[skip] {csv} already exists")
            continue
        run(["run_seed.py", "--seed", str(seed)], f"TFT seed {seed}")
        if not csv.exists():
            sys.exit(f"Expected {csv} after run_seed.py --seed {seed} but it is missing.")

    # ── 2. LSTM v2 ──────────────────────────────────────────────────────────
    lstm_v2 = sub / "submission_lstm_v2.csv"
    if lstm_v2.exists():
        print(f"[skip] {lstm_v2} already exists")
    else:
        run(["train_lstm.py"], "Train LSTM v2")
        run(["predict_lstm.py"], "Predict LSTM v2")

        # predict_lstm.py writes to lstm_local_preds.csv; the blend needs the v2 name.
        raw = sub / "lstm_local_preds.csv"
        if not raw.exists():
            sys.exit(f"Expected {raw} after predict_lstm.py — not found.")
        shutil.copy(raw, lstm_v2)
        print(f"Copied {raw} -> {lstm_v2}")

    # ── 3. Blend ────────────────────────────────────────────────────────────
    run(["reproduce_best.py"], "Build ensemble_70tft_30lstm.csv")

    final = sub / "ensemble_70tft_30lstm.csv"
    print("\n" + "=" * 70)
    print(f"Done. Submit {final} to Kaggle.")
    print("=" * 70)


if __name__ == "__main__":
    main()