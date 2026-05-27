# train_lstm.py — LSTM training with proper validation split and early stopping
#
# Phase 1 improvements over previous version:
# 1. Hold out 200 regions as validation set (region-level split, no leakage)
# 2. Track validation MAE every epoch
# 3. Early stopping based on validation MAE
# 4. Save BEST model (lowest val MAE), not the last epoch's model
# 5. Detailed per-epoch logging

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from sklearn.preprocessing import StandardScaler, LabelEncoder
import pandas as pd
import numpy as np
import joblib
import json
from pathlib import Path

from feature_engineering import aggregate_to_weekly, add_phase2_features
from dataset import DisasterSequenceDataset
from model import DisasterLSTM
from config_lstm import (
    TRAIN_PATH, MODEL_DIR,
    SEQ_LEN, PRED_LEN, FEATURES,
    BATCH_SIZE, EPOCHS, LEARNING_RATE,
    HIDDEN_SIZE, NUM_LAYERS, EMBED_DIM,
    VAL_REGION_COUNT, EARLY_STOPPING_PATIENCE, SEED,
    SUMMER_WEIGHT_MULTIPLIER,
)
from utils import extract_date_parts, compute_region_baseline, get_logger

logger = get_logger("train_lstm")


def split_regions(all_regions: np.ndarray, val_count: int, seed: int):
    """
    Split region_ids into train and validation lists.
    All weeks of a region go entirely into train OR val — never both.
    This prevents data leakage from sliding window samples.
    """
    rng = np.random.default_rng(seed)
    val_regions = rng.choice(all_regions, size=val_count, replace=False)
    val_set = set(val_regions.tolist())
    train_regions = [r for r in all_regions if r not in val_set]
    return train_regions, val_regions.tolist()


def evaluate(model, val_loader, device) -> float:
    """
    Compute unweighted MAE on validation set, BUT ONLY FOR SUMMER WEEKS.
    This aligns local validation with the Kaggle test set distribution.
    """
    model.eval()
    total_abs_error = 0.0
    total_count     = 0

    with torch.no_grad():
        for batch_X, batch_r, batch_b, batch_y, batch_summer in val_loader:
            batch_X = batch_X.to(device)
            batch_r = batch_r.to(device)
            batch_b = batch_b.to(device)
            batch_y = batch_y.to(device)
            batch_summer = batch_summer.to(device) 

            preds = model(batch_X, batch_r, batch_b)
            preds = torch.clamp(preds, 0.0, 5.0)

            # Filter for summer samples only
            summer_mask = (batch_summer == 1.0).squeeze()
            
            if summer_mask.sum() > 0:
                summer_preds = preds[summer_mask]
                summer_y = batch_y[summer_mask]
                
                total_abs_error += torch.abs(summer_preds - summer_y).sum().item()
                total_count     += summer_y.numel()

    # Fallback just in case a batch has no summer (rare)
    if total_count == 0:
        return float('inf')
        
    return total_abs_error / total_count


def main():
    torch.manual_seed(SEED)
    np.random.seed(SEED)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Training on {device}")

    # ── 1. Load and prepare data ──────────────────────────────────────────────
    logger.info("Loading training data...")
    train_raw = pd.read_csv(TRAIN_PATH)
    train_raw = extract_date_parts(train_raw)

    logger.info("Computing region baselines...")
    baselines = compute_region_baseline(train_raw)

    logger.info("Aggregating to weekly...")
    train_weekly = aggregate_to_weekly(train_raw, is_train=True)

    train_weekly = add_phase2_features(train_weekly)

    train_weekly = train_weekly.merge(baselines, on="region_id", how="left")
    train_weekly["region_mean_score"] = train_weekly["region_mean_score"].fillna(0.0)

    # ── 2. Split regions into train and validation ────────────────────────────
    all_regions = train_weekly["region_id"].unique()
    train_regions, val_regions = split_regions(all_regions, VAL_REGION_COUNT, SEED)
    logger.info(f"Train regions: {len(train_regions):,} | Val regions: {len(val_regions):,}")

    # ── 3. Fit encoders and scalers on TRAIN ONLY (avoid leakage) ─────────────
    region_encoder = LabelEncoder()
    region_encoder.fit(all_regions)   # fit on all so test inference works

    region_encoder_dict = dict(zip(
        region_encoder.classes_,
        region_encoder.transform(region_encoder.classes_)
    ))

    scaler = StandardScaler()
    train_only_df = train_weekly[train_weekly["region_id"].isin(train_regions)]
    scaler.fit(train_only_df[FEATURES])

    joblib.dump(region_encoder, MODEL_DIR / "region_encoder.pkl")
    joblib.dump(scaler,         MODEL_DIR / "scaler.pkl")
    logger.info("Saved encoders and scaler.")

    # ── 4. Build datasets and dataloaders ─────────────────────────────────────
    logger.info("Building training dataset...")
    train_dataset = DisasterSequenceDataset(
        train_weekly,
        region_encoder=region_encoder_dict,
        scaler=scaler,
        is_train=True,
        region_filter=train_regions,
    )
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    logger.info(f"Training samples : {len(train_dataset):,}")

    logger.info("Building validation dataset...")
    val_dataset = DisasterSequenceDataset(
        train_weekly,
        region_encoder=region_encoder_dict,
        scaler=scaler,
        is_train=True,                  # still need targets to compute val MAE
        region_filter=val_regions,
    )
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)
    logger.info(f"Validation samples: {len(val_dataset):,}")

    # ── 5. Initialize model ───────────────────────────────────────────────────
    num_regions = len(region_encoder.classes_)
    model = DisasterLSTM(
        num_regions=num_regions,
        num_features=len(FEATURES),
        embed_dim=EMBED_DIM,
        hidden_size=HIDDEN_SIZE,
        num_layers=NUM_LAYERS,
        pred_len=PRED_LEN,
    ).to(device)

    n_params = sum(p.numel() for p in model.parameters())
    logger.info(f"Model parameters: {n_params:,}")

    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
    criterion = nn.L1Loss(reduction='none')   # MAE with manual weighting

    # ── 6. Training loop with early stopping ──────────────────────────────────
    best_val_mae       = float("inf")
    best_epoch         = -1
    epochs_no_improve  = 0
    history            = []

    for epoch in range(1, EPOCHS + 1):
        model.train()
        total_loss = 0.0
        n_batches  = 0

        for batch_X, batch_r, batch_b, batch_y, batch_summer in train_loader:
            batch_X      = batch_X.to(device)
            batch_r      = batch_r.to(device)
            batch_b      = batch_b.to(device)
            batch_y      = batch_y.to(device)
            batch_summer = batch_summer.to(device)

            optimizer.zero_grad()
            preds = model(batch_X, batch_r, batch_b)

            # Seasonal weighted loss
            raw_loss = criterion(preds, batch_y)
            weights  = torch.where(
                batch_summer == 1.0,
                SUMMER_WEIGHT_MULTIPLIER,
                1.0
            ).unsqueeze(1)
            weighted_loss = (raw_loss * weights).mean()

            weighted_loss.backward()
            optimizer.step()

            total_loss += weighted_loss.item()
            n_batches  += 1

        train_loss = total_loss / n_batches
        val_mae    = evaluate(model, val_loader, device)

        history.append({
            "epoch"      : epoch,
            "train_loss" : round(train_loss, 4),
            "val_mae"    : round(val_mae, 4),
        })

        # Best model tracking
        is_best = val_mae < best_val_mae
        marker  = " *" if is_best else ""
        logger.info(
            f"Epoch {epoch:2d}/{EPOCHS} | "
            f"Train weighted MAE: {train_loss:.4f} | "
            f"Val MAE: {val_mae:.4f}{marker}"
        )

        if is_best:
            best_val_mae      = val_mae
            best_epoch        = epoch
            epochs_no_improve = 0
            torch.save(model.state_dict(), MODEL_DIR / "lstm_weights_best.pth")
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= EARLY_STOPPING_PATIENCE:
                logger.info(
                    f"Early stopping at epoch {epoch} "
                    f"(no improvement for {EARLY_STOPPING_PATIENCE} epochs)"
                )
                break

    # ── 7. Save final metadata ────────────────────────────────────────────────
    # Also save the LAST model in case the user wants to compare
    torch.save(model.state_dict(), MODEL_DIR / "lstm_weights_last.pth")

    meta = {
        "best_val_mae"        : round(best_val_mae, 4),
        "best_epoch"          : best_epoch,
        "total_epochs_run"    : len(history),
        "n_train_samples"     : len(train_dataset),
        "n_val_samples"       : len(val_dataset),
        "n_train_regions"     : len(train_regions),
        "n_val_regions"       : len(val_regions),
        "n_model_parameters"  : int(n_params),
        "features"            : FEATURES,
        "history"             : history,
    }
    with open(MODEL_DIR / "train_meta.json", "w") as f:
        json.dump(meta, f, indent=2)

    logger.info("=" * 60)
    logger.info(f"BEST val MAE: {best_val_mae:.4f} at epoch {best_epoch}")
    logger.info(f"Best model saved to: {MODEL_DIR / 'lstm_weights_best.pth'}")
    logger.info(f"Metadata saved to  : {MODEL_DIR / 'train_meta.json'}")
    logger.info("Training complete.")
    logger.info("=" * 60)


if __name__ == "__main__":
    main()