import numpy as np
import pandas as pd
import torch
from pytorch_forecasting import TemporalFusionTransformer, TimeSeriesDataSet
from config_tft import *
from data_prep import prepare

def main():
    combined = prepare(force_rebuild=False)
    training = TimeSeriesDataSet.load(str(ARTIFACT_DIR / "training_dataset.pkl"))

    # Auto-find the newest checkpoint
    ckpt_candidates = list(ARTIFACT_DIR.glob("tft_best*.ckpt"))
    ckpt_candidates.sort(key=lambda x: x.stat().st_mtime, reverse=True)
    ckpt = ckpt_candidates[0]
    
    print(f"Loading model: {ckpt}")
    tft = TemporalFusionTransformer.load_from_checkpoint(str(ckpt))
    tft.eval()

    pred_ds = TimeSeriesDataSet.from_dataset(training, combined, predict=True, stop_randomization=True)
    pred_loader = pred_ds.to_dataloader(train=False, batch_size=BATCH_SIZE * 2, num_workers=NUM_WORKERS)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    tft.to(device)

    output = tft.predict(pred_loader, mode="quantiles", return_index=True, trainer_kwargs=dict(accelerator=ACCELERATOR, devices=1))
    
    quantile_preds = output.output.cpu().numpy()
    median_preds = quantile_preds[:, :, MEDIAN_QUANTILE_IDX]
    index_df = output.index.reset_index(drop=True)

    # This brings the tiny logarithmic numbers back to the normal 0 to 5 scale
    median_preds = np.expm1(median_preds)

    # THE RELU CLAMP
    median_preds = np.clip(median_preds, 0.0, 5.0)

    pred_df = pd.DataFrame({
        REGION_COL: index_df[REGION_COL].astype(str).values,
        "pred_week1": median_preds[:, 0], "pred_week2": median_preds[:, 1],
        "pred_week3": median_preds[:, 2], "pred_week4": median_preds[:, 3], "pred_week5": median_preds[:, 4],
    })

    sample = pd.read_csv(SAMPLE_SUB_PATH)
    sample[REGION_COL] = sample[REGION_COL].astype(str)
    submission = sample[[REGION_COL]].merge(pred_df, on=REGION_COL, how="left")

    pred_cols = ["pred_week1", "pred_week2", "pred_week3", "pred_week4", "pred_week5"]
    submission[pred_cols] = submission[pred_cols].round(4)

    # DYNAMIC FILE NAMING BASED ON SEED
    if SEED == 42:
        out_name = "submission_tft_v4_relu_42_anomaly.csv"
    else:
        out_name = f"submission_tft_v4_relu_{SEED}_anomaly.csv"

    out_path = SUBMISSION_DIR / out_name
    submission.to_csv(out_path, index=False)
    print(f"Saved -> {out_path}")

if __name__ == "__main__":
    main()