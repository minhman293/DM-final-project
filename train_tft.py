"""
train_tft.py — Train the Temporal Fusion Transformer on weekly drought scores.

Pipeline:
    1. Load combined long frame (train + test + 5 future weeks per region).
    2. Build a TimeSeriesDataSet with:
         - group_ids:                [region_id]
         - target:                   score
         - time_varying_known_reals: month_sin, month_cos, time_idx
         - time_varying_known_cats:  month
         - time_varying_unknown_reals: all weather aggregates + score + score_known
         - static_categoricals:      region_id (so each region gets an embedding)
    3. Define a TFT with QuantileLoss over 7 quantiles.
    4. Train with PyTorch Lightning. Validate on last VAL_WEEKS_PER_REGION
       weeks of each region's train history.
    5. Save the best checkpoint.

Why this beats the LSTM:
    - Encoder length is 52 weeks (vs your 13). Sees full annual seasonality.
    - Past score is an input. Lag-1 score autocorrelation = 0.936 per EDA.
    - Region embedding is learned end-to-end inside variable selection,
      not concatenated to LSTM output as an afterthought.
    - Quantile median output is the MAE-optimal point predictor and
      handles the heavily zero-skewed target without a threshold hack.
"""

from __future__ import annotations

import sys
import warnings
import pandas as pd
import numpy as np

# ── Lightning import: must match the one pytorch-forecasting uses internally ─
# pytorch-forecasting >= 1.0 has shifted between `lightning.pytorch` and
# `pytorch_lightning` across versions. If we import the other one,
# trainer.fit() raises "model must be a LightningModule, got
# TemporalFusionTransformer" because the two namespaces ship SEPARATE
# (non-identical) LightningModule classes and Lightning's isinstance check
# fails across the boundary.
#
# Defensive strategy: pull the *exact* Trainer and callback classes that
# pytorch-forecasting's base_model module already imported. Whatever it uses,
# we use. This bypasses the mismatch entirely.
import pytorch_forecasting.models.base_model as _pf_base

# Resolve which Lightning namespace pytorch-forecasting bound to
# (Only print on the main process — DataLoader workers re-import this module
#  on Windows and would otherwise spam this line.)
_is_main = __name__ == "__main__" or "spawn" not in sys.argv[0].lower()
_pl_namespace = sys.modules[_pf_base.LightningModule.__module__.split(".")[0]]
if _pf_base.LightningModule.__module__.startswith("lightning.pytorch"):
    import lightning.pytorch as pl
    from lightning.pytorch.callbacks import (
        EarlyStopping,
        ModelCheckpoint,
        LearningRateMonitor,
    )
    if _is_main:
        print(f"[train_tft] Using lightning.pytorch (matches pytorch-forecasting)")
else:
    import pytorch_lightning as pl
    from pytorch_lightning.callbacks import (
        EarlyStopping,
        ModelCheckpoint,
        LearningRateMonitor,
    )
    if _is_main:
        print(f"[train_tft] Using pytorch_lightning (matches pytorch-forecasting)")

# Sanity assertion — verify our Trainer and pf's LightningModule are in the
# same Lightning namespace. If they're not, predict early what fit() would say.
assert pl.LightningModule is _pf_base.LightningModule, (
    f"Lightning mismatch detected at import time!\n"
    f"  pf_base.LightningModule.__module__ = {_pf_base.LightningModule.__module__}\n"
    f"  pl.LightningModule.__module__      = {pl.LightningModule.__module__}\n"
    f"  These must be identical. Reinstall: \n"
    f"  pip uninstall pytorch_lightning lightning lightning_fabric\n"
    f"  pip install pytorch-forecasting --force-reinstall"
)

import torch
from pytorch_forecasting import TemporalFusionTransformer, TimeSeriesDataSet
from pytorch_forecasting.data import GroupNormalizer
from pytorch_forecasting.metrics import QuantileLoss


# ── fp16 safety patch ────────────────────────────────────────────────────────
# pytorch-forecasting's ScaledDotProductAttention masks with -1e9, which
# overflows in fp16 (max magnitude ~65,504). If we're using 16-mixed and
# the user opted in, patch the attention to use a smaller mask fill that
# still acts as -inf after softmax.
def _maybe_patch_attention_for_fp16():
    from config_tft import PRECISION, PATCH_ATTENTION_FOR_FP16
    if not PATCH_ATTENTION_FOR_FP16 or "16" not in str(PRECISION):
        return

    from pytorch_forecasting.models.temporal_fusion_transformer import sub_modules

    # Faithful copy of pytorch-forecasting's ScaledDotProductAttention.forward
    # with `-1e9` swapped for `-1e4`. Everything else (scale flag, softmax,
    # dropout handling) is preserved byte-for-byte.
    def patched_forward(self, q, k, v, mask=None):
        attn = torch.bmm(q, k.permute(0, 2, 1))  # query-key overlap
        if self.scale:
            dimension = torch.as_tensor(
                k.size(-1), dtype=attn.dtype, device=attn.device
            ).sqrt()
            attn = attn / dimension
        if mask is not None:
            attn = attn.masked_fill(mask, -1e4)   # fp16-safe; was -1e9
        attn = self.softmax(attn)
        if self.dropout is not None:
            attn = self.dropout(attn)
        output = torch.bmm(attn, v)
        return output, attn

    sub_modules.ScaledDotProductAttention.forward = patched_forward
    print("[train_tft] Patched ScaledDotProductAttention for fp16 (-1e9 → -1e4)")


_maybe_patch_attention_for_fp16()

from config_tft import (
    ACCELERATOR,
    ARTIFACT_DIR,
    ATTENTION_HEAD_SIZE,
    BATCH_SIZE,
    CHECKPOINT_PATH,
    DEC_LEN,
    DEVICES,
    DROPOUT,
    EARLY_STOPPING_PATIENCE,
    ENC_LEN,
    GRADIENT_CLIP_VAL,
    HIDDEN_CONTINUOUS_SIZE,
    HIDDEN_SIZE,
    LEARNING_RATE,
    LSTM_LAYERS,
    MAX_EPOCHS,
    NUM_WORKERS,
    PRECISION,
    QUANTILES,
    REGION_COL,
    SEED,
    TARGET_COL,
    TRAIN_WINDOW_STRIDE,
    VAL_WEEKS_PER_REGION,
    WEEKLY_FEATURES,
)
from data_prep import prepare
from utils_tft import get_logger

warnings.filterwarnings("ignore", category=UserWarning)
log = get_logger("train_tft")


# ─────────────────────────────────────────────────────────────────────────────
#  Dataset construction
# ─────────────────────────────────────────────────────────────────────────────
def build_datasets(combined: pd.DataFrame) -> tuple[TimeSeriesDataSet, TimeSeriesDataSet]:
    """
    Build (training_dataset, validation_dataset).

    Training cutoff: each region has TRAIN_END = (last train time_idx - VAL_WEEKS_PER_REGION).
    Training windows are drawn so that the decoder ends ON OR BEFORE TRAIN_END.
    Validation windows are drawn where decoder ends within the held-out tail.

    pytorch-forecasting builds windows implicitly: a window of length
    (max_encoder_length + max_prediction_length) is created for every valid
    starting time_idx in the dataset, then filtered by the predict-on/training
    selectors. We pass `min_prediction_idx` / `training_cutoff` to control
    which side a window lands on.
    """
    # Per-region last KNOWN-score time_idx (i.e. before the test rows)
    known_mask = combined["score_known"] == 1
    last_known_idx = (
        combined[known_mask]
        .groupby(REGION_COL)["time_idx"]
        .max()
    )
    log.info("Last known-score time_idx — min: %d, median: %d, max: %d",
             last_known_idx.min(), int(last_known_idx.median()), last_known_idx.max())

    # Global training cutoff: take the MIN across regions, minus the validation window.
    # Using the minimum guarantees we never leak any region's validation labels.
    global_train_cutoff = int(last_known_idx.min() - VAL_WEEKS_PER_REGION)
    log.info("Global training cutoff (time_idx): %d", global_train_cutoff)

    # Common dataset kwargs
    common_kwargs = dict(
        time_idx="time_idx",
        target=TARGET_COL,
        weight="summer_weight",
        group_ids=[REGION_COL],
        min_encoder_length=ENC_LEN // 2,
        max_encoder_length=ENC_LEN,
        min_prediction_length=DEC_LEN,
        max_prediction_length=DEC_LEN,
        static_categoricals=[REGION_COL],
        time_varying_known_categoricals=["month"],
        time_varying_known_reals=["time_idx", "month_sin", "month_cos"],
        time_varying_unknown_reals=[
            TARGET_COL,
            "score_known",
            *WEEKLY_FEATURES,
        ],
        target_normalizer=GroupNormalizer(
            groups=[REGION_COL],
            transformation="relu",   # changed from "softplus" to allow zero-skewed target without log transform
        ),
        add_relative_time_idx=True,
        add_target_scales=True,
        add_encoder_length=True,
        allow_missing_timesteps=False,
        categorical_encoders={
            REGION_COL: pytorch_forecasting_encoder_for(REGION_COL),
            "month": pytorch_forecasting_encoder_for("month"),
        },
    )

    log.info("Building training TimeSeriesDataSet...")
    training = TimeSeriesDataSet(
        combined[combined["time_idx"] <= global_train_cutoff].copy(),
        **common_kwargs,
    )

    log.info("Building validation TimeSeriesDataSet from training spec...")
    # `from_dataset` clones the spec; we change only the data window.
    # We include enough lead-in encoder context plus the validation tail.
    val_min_idx = global_train_cutoff - ENC_LEN  # ensure full encoder context
    validation = TimeSeriesDataSet.from_dataset(
        training,
        combined[combined["time_idx"] >= val_min_idx].copy(),
        predict=False,
        stop_randomization=True,
        min_prediction_idx=global_train_cutoff + 1,
    )

    log.info("Train windows: %d | Val windows: %d", len(training), len(validation))
    return training, validation


def pytorch_forecasting_encoder_for(col: str):
    """
    pytorch-forecasting's NaNLabelEncoder handles unseen categories at
    inference (we set add_nan=True). All test regions exist in train per EDA,
    but defensive coding never hurts.
    """
    from pytorch_forecasting.data.encoders import NaNLabelEncoder
    return NaNLabelEncoder(add_nan=True)


# ─────────────────────────────────────────────────────────────────────────────
#  Model
# ─────────────────────────────────────────────────────────────────────────────
def build_model(training: TimeSeriesDataSet) -> TemporalFusionTransformer:
    """
    Instantiate TFT from the training dataset spec.
    All architecture knobs come from config_tft.
    """
    tft = TemporalFusionTransformer.from_dataset(
        training,
        learning_rate=LEARNING_RATE,
        hidden_size=HIDDEN_SIZE,
        lstm_layers=LSTM_LAYERS,
        attention_head_size=ATTENTION_HEAD_SIZE,
        dropout=DROPOUT,
        hidden_continuous_size=HIDDEN_CONTINUOUS_SIZE,
        loss=QuantileLoss(quantiles=QUANTILES),
        log_interval=50,
        reduce_on_plateau_patience=3,
    )
    log.info("Model size: %.2fM params", sum(p.numel() for p in tft.parameters()) / 1e6)
    return tft


# ─────────────────────────────────────────────────────────────────────────────
#  Train
# ─────────────────────────────────────────────────────────────────────────────
def train():
    pl.seed_everything(SEED, workers=True)
    torch.set_float32_matmul_precision("high")

    log.info("=" * 70)
    log.info("STEP 1 — Prepare data")
    log.info("=" * 70)
    combined = prepare(force_rebuild=True) 

    log.info("=" * 70)
    log.info("STEP 2 — Build datasets")
    log.info("=" * 70)
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

    log.info("=" * 70)
    log.info("STEP 3 — Build model")
    log.info("=" * 70)
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
            filename="tft_best",
            monitor="val_loss",
            mode="min",
            save_top_k=1,
            save_weights_only=False,
        ),
        LearningRateMonitor(logging_interval="epoch"),
    ]

    log.info("=" * 70)
    log.info("STEP 4 — Train")
    log.info("=" * 70)
    trainer = pl.Trainer(
        max_epochs=MAX_EPOCHS,
        accelerator=ACCELERATOR,
        devices=DEVICES,
        precision=PRECISION,
        gradient_clip_val=GRADIENT_CLIP_VAL,
        callbacks=callbacks,
        enable_progress_bar=True,
        enable_model_summary=True,
        log_every_n_steps=20,
        deterministic=False,   # `True` slows training a lot on GPU; we set seed anyway
    )

    trainer.fit(tft, train_dataloaders=train_loader, val_dataloaders=val_loader)

    # The ModelCheckpoint callback saves to <ARTIFACT_DIR>/tft_best.ckpt
    best_path = callbacks[1].best_model_path
    log.info("Best checkpoint: %s", best_path)
    log.info("Best val_loss : %.5f", float(callbacks[1].best_model_score))

    # Save the training dataset spec — predict.py will need it to construct
    # a matching prediction dataset for the test rows.
    training.save(str(ARTIFACT_DIR / "training_dataset.pkl"))
    log.info("Saved training dataset spec to %s", ARTIFACT_DIR / "training_dataset.pkl")


if __name__ == "__main__":
    train()