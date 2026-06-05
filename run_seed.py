"""
run_seed.py — Train one TFT seed end-to-end and write its prediction CSV
              with the filename that ensemble.py / reproduce_best.py expect.

Wraps build_datasets() and build_model() from train_tft.py so each seed gets
its own checkpoint and CSV. Bypasses predict_tft.py entirely (and its
hardcoded "tft_best-v5.ckpt" path).

Usage:
    python run_seed.py --seed 42
    python run_seed.py --seed 123
    python run_seed.py --seed 999
    python run_seed.py --seed 42 --skip-train   # regenerate CSV only

Outputs:
    artifacts_tft/tft_best_seed{N}.ckpt
    artifacts_tft/training_dataset.pkl              (written once, shared)
    submissions/submission_tft_v4_relu.csv          (when seed=42)
    submissions/submission_tft_v4_relu_{N}.csv      (other seeds)
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch

# Importing train_tft runs its module init: detects the right Lightning
# namespace, applies the fp16 attention patch, and exposes pl + callbacks
# as module-level names. We re-import those names from there.
from train_tft import (
    build_datasets,
    build_model,
    pl,
    EarlyStopping,
    ModelCheckpoint,
    LearningRateMonitor,
)
from data_prep import prepare
from config_tft import (
    ACCELERATOR,
    ARTIFACT_DIR,
    BATCH_SIZE,
    DEVICES,
    EARLY_STOPPING_PATIENCE,
    GRADIENT_CLIP_VAL,
    MAX_EPOCHS,
    MEDIAN_QUANTILE_IDX,
    NUM_WORKERS,
    PRECISION,
    REGION_COL,
    SAMPLE_SUB_PATH,
    SUBMISSION_DIR,
)
from pytorch_forecasting import TemporalFusionTransformer, TimeSeriesDataSet
from utils_tft import get_logger

log = get_logger("run_seed")
PRED_COLS = ["pred_week1", "pred_week2", "pred_week3", "pred_week4", "pred_week5"]


def output_csv_name(seed: int) -> str:
    """Match the filenames that ensemble.py / reproduce_best.py expect."""
    if seed == 42:
        return "submission_tft_v4_relu.csv"
    return f"submission_tft_v4_relu_{seed}.csv"


# ── Train ────────────────────────────────────────────────────────────────────
def train_one(seed: int) -> Path:
    log.info("=" * 60)
    log.info(f"TFT TRAINING — seed {seed}")
    log.info("=" * 60)

    pl.seed_everything(seed, workers=True)
    torch.set_float32_matmul_precision("high")

    combined = prepare(force_rebuild=False)
    training, validation = build_datasets(combined)

    train_loader = training.to_dataloader(
        train=True,
        batch_size=BATCH_SIZE,
        num_workers=NUM_WORKERS,
        persistent_workers=NUM_WORKERS > 0,
    )
    val_loader = validation.to_dataloader(
        train=False,
        batch_size=BATCH_SIZE * 2,
        num_workers=NUM_WORKERS,
        persistent_workers=NUM_WORKERS > 0,
    )

    tft = build_model(training)

    callbacks = [
        EarlyStopping(
            monitor="val_loss",
            patience=EARLY_STOPPING_PATIENCE,
            mode="min",
            verbose=True,
        ),
        ModelCheckpoint(
            dirpath=str(ARTIFACT_DIR),
            filename=f"tft_best_seed{seed}",
            monitor="val_loss",
            mode="min",
            save_top_k=1,
        ),
        LearningRateMonitor(logging_interval="epoch"),
    ]

    trainer = pl.Trainer(
        max_epochs=MAX_EPOCHS,
        accelerator=ACCELERATOR,
        devices=DEVICES,
        precision=PRECISION,
        gradient_clip_val=GRADIENT_CLIP_VAL,
        callbacks=callbacks,
        enable_progress_bar=True,
        enable_model_summary=False,
        log_every_n_steps=20,
        deterministic=False,
    )
    trainer.fit(tft, train_dataloaders=train_loader, val_dataloaders=val_loader)

    best_path = Path(callbacks[1].best_model_path)
    log.info(f"Seed {seed}: best ckpt = {best_path}")
    log.info(f"Seed {seed}: val_loss  = {float(callbacks[1].best_model_score):.5f}")

    # Spec is identical across seeds; save once.
    spec_path = ARTIFACT_DIR / "training_dataset.pkl"
    if not spec_path.exists():
        training.save(str(spec_path))
        log.info(f"Saved dataset spec: {spec_path}")

    return best_path


# ── Predict ──────────────────────────────────────────────────────────────────
def predict_one(seed: int, ckpt: Path) -> Path:
    log.info("=" * 60)
    log.info(f"TFT PREDICTION — seed {seed}")
    log.info("=" * 60)

    combined = prepare(force_rebuild=False)
    spec_path = ARTIFACT_DIR / "training_dataset.pkl"
    training = TimeSeriesDataSet.load(str(spec_path))

    tft = TemporalFusionTransformer.load_from_checkpoint(str(ckpt))
    tft.eval()

    pred_ds = TimeSeriesDataSet.from_dataset(
        training,
        combined,
        predict=True,
        stop_randomization=True,
    )
    pred_loader = pred_ds.to_dataloader(
        train=False,
        batch_size=BATCH_SIZE * 2,
        num_workers=NUM_WORKERS,
    )

    output = tft.predict(
        pred_loader,
        mode="quantiles",
        return_x=True,
        return_index=True,
        trainer_kwargs=dict(accelerator=ACCELERATOR, devices=1),
    )

    q = output.output.cpu().numpy()
    median = q[:, :, MEDIAN_QUANTILE_IDX] if q.ndim == 3 else q
    median = np.clip(median, 0.0, 5.0)

    index_df = output.index.reset_index(drop=True)
    pred_df = pd.DataFrame({
        REGION_COL: index_df[REGION_COL].astype(str).values,
        **{c: median[:, i] for i, c in enumerate(PRED_COLS)},
    })

    sample = pd.read_csv(SAMPLE_SUB_PATH)
    sample[REGION_COL] = sample[REGION_COL].astype(str)
    submission = sample[[REGION_COL]].merge(pred_df, on=REGION_COL, how="left")

    if submission[PRED_COLS].isna().any().any():
        log.warning("Some regions missing — filling with per-region historical mean")
        region_mean = combined.groupby(REGION_COL)["score"].mean()
        for c in PRED_COLS:
            submission[c] = submission.apply(
                lambda r: r[c] if pd.notna(r[c]) else region_mean.get(r[REGION_COL], 0.0),
                axis=1,
            )

    submission[PRED_COLS] = submission[PRED_COLS].round(4)
    out_path = SUBMISSION_DIR / output_csv_name(seed)
    submission.to_csv(out_path, index=False)
    log.info(f"Wrote {out_path}")
    return out_path


# ── Entry point ──────────────────────────────────────────────────────────────
def main():
    ap = argparse.ArgumentParser(description="Train one TFT seed and write its CSV.")
    ap.add_argument(
        "--seed", type=int, required=True, choices=[42, 123, 999],
        help="random seed for this run (one of the three ensemble seeds)",
    )
    ap.add_argument(
        "--skip-train", action="store_true",
        help="skip training; regenerate CSV from the existing seed-specific checkpoint",
    )
    args = ap.parse_args()

    if args.skip_train:
        ckpt = ARTIFACT_DIR / f"tft_best_seed{args.seed}.ckpt"
        if not ckpt.exists():
            sys.exit(f"No checkpoint at {ckpt} — run without --skip-train first.")
    else:
        ckpt = train_one(args.seed)

    predict_one(args.seed, ckpt)


if __name__ == "__main__":
    main()