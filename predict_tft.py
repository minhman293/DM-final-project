"""
predict_tft.py — Load the trained TFT, predict the 5 future weeks per region,
                  format submission.csv.

Inference approach:
    For each region we use the LAST ENC_LEN known weeks as encoder context
    and ask TFT to predict the 5 synthetic future weeks (those tagged
    is_test == 2 in the combined frame).

    pytorch-forecasting offers two convenient ways to do this:
        1. `TimeSeriesDataSet.from_dataset(..., predict=True)` which keeps
           only the last (encoder + decoder) window per group.
        2. Manual: build a dataset where the decoder horizon aligns with
           the future rows.

    We use option 1 — it's the canonical pattern.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import torch
from pytorch_forecasting import TemporalFusionTransformer, TimeSeriesDataSet

from config_tft import (
    ACCELERATOR,
    ARTIFACT_DIR,
    BATCH_SIZE,
    CHECKPOINT_PATH,
    MEDIAN_QUANTILE_IDX,
    NUM_WORKERS,
    REGION_COL,
    SAMPLE_SUB_PATH,
    SUBMISSION_DIR,
)
from data_prep import prepare
from utils_tft import get_logger

log = get_logger("predict_tft")


def main():
    log.info("=" * 70)
    log.info("STEP 1 — Load combined frame")
    log.info("=" * 70)
    combined = prepare(force_rebuild=False)

    log.info("=" * 70)
    log.info("STEP 2 — Load training dataset spec + best checkpoint")
    log.info("=" * 70)
    training = TimeSeriesDataSet.load(str(ARTIFACT_DIR / "training_dataset.pkl"))

    # Locate the best checkpoint. ModelCheckpoint saves "tft_best.ckpt" but
    # may append a version suffix; we glob defensively.
    ckpt_candidates = list(ARTIFACT_DIR.glob("tft_best*.ckpt"))
    if not ckpt_candidates:
        raise FileNotFoundError(
            f"No checkpoint matching tft_best*.ckpt found in {ARTIFACT_DIR}"
        )
    ckpt = ARTIFACT_DIR / "tft_best.ckpt"
    log.info("Loading model weights from %s", ckpt)
    tft = TemporalFusionTransformer.load_from_checkpoint(str(ckpt))
    tft.eval()

    log.info("=" * 70)
    log.info("STEP 3 — Build prediction dataset (one window per region, ending at the 5 future weeks)")
    log.info("=" * 70)
    # predict=True keeps the LAST (encoder + decoder) window per group_id.
    # Since the combined frame's last 5 weeks per region ARE the prediction
    # targets, this aligns the decoder horizon to those rows automatically.
    pred_ds = TimeSeriesDataSet.from_dataset(
        training,
        combined,
        predict=True,
        stop_randomization=True,
    )
    log.info("Prediction windows: %d (expect = number of regions)", len(pred_ds))

    pred_loader = pred_ds.to_dataloader(
        train=False,
        batch_size=BATCH_SIZE * 2,
        num_workers=NUM_WORKERS,
    )

    log.info("=" * 70)
    log.info("STEP 4 — Predict")
    log.info("=" * 70)
    # mode="raw" returns the full quantile tensor [n_windows, dec_len, n_quantiles]
    # We pick the 0.5 quantile (MAE-optimal point predictor).
    device = "cuda" if torch.cuda.is_available() and ACCELERATOR != "cpu" else "cpu"
    log.info("Running inference on %s", device)
    tft.to(device)

    # `predict` returns a Prediction object; mode="quantiles" gives shape
    # [n_windows, decoder_length, n_quantiles].
    output = tft.predict(
        pred_loader,
        mode="quantiles",
        return_x=True,
        return_index=True,
        trainer_kwargs=dict(accelerator=ACCELERATOR, devices=1),
    )

    # output.output:  Tensor [N, 5, 7]
    # output.index :  DataFrame with region_id and time_idx of decoder start per window
    quantile_preds = output.output.cpu().numpy()
    if quantile_preds.ndim == 3:
        median_preds = quantile_preds[:, :, MEDIAN_QUANTILE_IDX]
    else:
        # Some versions return shape [N, 5] when only one quantile is asked for.
        # Defensive branch.
        median_preds = quantile_preds

    index_df = output.index.reset_index(drop=True)
    log.info("Prediction tensor shape: %s", median_preds.shape)
    log.info("Index sample:\n%s", index_df.head())

    # Clip to valid range [0, 5]
    median_preds = np.clip(median_preds, 0.0, 5.0)

    # Build submission frame
    pred_df = pd.DataFrame({
        REGION_COL: index_df[REGION_COL].astype(str).values,
        "pred_week1": median_preds[:, 0],
        "pred_week2": median_preds[:, 1],
        "pred_week3": median_preds[:, 2],
        "pred_week4": median_preds[:, 3],
        "pred_week5": median_preds[:, 4],
    })

    # Align with sample_submission ordering
    log.info("=" * 70)
    log.info("STEP 5 — Align with sample_submission and write CSV")
    log.info("=" * 70)
    sample = pd.read_csv(SAMPLE_SUB_PATH)
    sample[REGION_COL] = sample[REGION_COL].astype(str)
    submission = sample[[REGION_COL]].merge(pred_df, on=REGION_COL, how="left")

    # Fallback: per-region historical mean for any region we somehow missed
    pred_cols = ["pred_week1", "pred_week2", "pred_week3", "pred_week4", "pred_week5"]
    missing = submission[pred_cols].isna().any(axis=1).sum()
    if missing:
        log.warning("%d regions missing predictions — filling with per-region mean", missing)
        region_mean = combined.groupby(REGION_COL)["score"].mean()
        for c in pred_cols:
            submission[c] = submission.apply(
                lambda r: r[c] if pd.notna(r[c]) else region_mean.get(r[REGION_COL], 0.0),
                axis=1,
            )

    submission[pred_cols] = submission[pred_cols].round(4)

    out_path = SUBMISSION_DIR / "submission_tft_upgrade.csv"
    submission.to_csv(out_path, index=False)
    log.info("Saved → %s", out_path)
    log.info("Submission summary:")
    log.info("\n%s", submission[pred_cols].describe().to_string())

    # Distribution sanity check vs train (per EDA section 4)
    log.info("\nFraction predicted < 0.5 (train: ~59.6%% are score 0):")
    for c in pred_cols:
        frac = (submission[c] < 0.5).mean()
        log.info("  %s: %.3f", c, frac)


if __name__ == "__main__":
    main()