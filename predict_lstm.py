import torch
import pandas as pd
import numpy as np
import joblib

from config_lstm import *
from feature_engineering import aggregate_to_weekly
from utils import extract_date_parts, compute_region_baseline, get_logger
from model import DisasterLSTM

logger = get_logger("predict_lstm")

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Running inference on {device}...")

    # 1. Load Train Raw just to compute the true baselines
    logger.info("Computing region baselines from training data...")
    train_raw = pd.read_csv(TRAIN_PATH)
    baselines = compute_region_baseline(train_raw)

    # 2. Load Test Data and Aggregate
    logger.info("Loading test data...")
    test_raw = pd.read_csv(TEST_PATH)
    test_raw = extract_date_parts(test_raw)
    
    test_weekly = aggregate_to_weekly(test_raw, is_train=False)
    
    # Merge Baselines into Test
    test_weekly = test_weekly.merge(baselines, on="region_id", how="left")
    test_weekly["region_mean_score"] = test_weekly["region_mean_score"].fillna(0.0)

    # 3. Load Encoders, Scalers, and Model Weights
    logger.info("Loading pre-trained encoders and scalers...")
    region_encoder = joblib.load(MODEL_DIR / "region_encoder.pkl")
    scaler = joblib.load(MODEL_DIR / "scaler.pkl")
    
    num_regions = len(region_encoder.classes_)
    
    logger.info("Loading LSTM model...")
    model = DisasterLSTM(
        num_regions=num_regions,
        num_features=len(FEATURES),
        embed_dim=EMBED_DIM,
        hidden_size=HIDDEN_SIZE,
        num_layers=NUM_LAYERS,
        pred_len=PRED_LEN
    ).to(device)
    
    model.load_state_dict(torch.load(MODEL_DIR / "lstm_weights.pth", map_location=device))
    model.eval()

    # 4. Generate Predictions
    logger.info("Generating predictions...")
    predictions = []
    
    with torch.no_grad():
        for region_id, group in test_weekly.groupby("region_id"):
            group = group.sort_values("week_id").reset_index(drop=True)
            
            seq_x = scaler.transform(group[FEATURES].values[-SEQ_LEN:])
            r_idx = region_encoder.transform([region_id])[0]
            b_val = group["region_mean_score"].iloc[0]
            
            seq_x_tensor = torch.tensor(seq_x, dtype=torch.float32).unsqueeze(0).to(device)
            r_idx_tensor = torch.tensor([r_idx], dtype=torch.long).to(device)
            b_tensor = torch.tensor([b_val], dtype=torch.float32).to(device)
            
            pred = model(seq_x_tensor, r_idx_tensor, b_tensor)
            
            pred_np = pred.cpu().numpy().flatten()
            pred_np = np.clip(pred_np, 0.0, 5.0)
            
            predictions.append({
                "region_id": region_id,
                "pred_week1": pred_np[0],
                "pred_week2": pred_np[1],
                "pred_week3": pred_np[2],
                "pred_week4": pred_np[3],
                "pred_week5": pred_np[4]
            })

    pred_df = pd.DataFrame(predictions)

    # 5. Format and Apply Threshold Squeezer
    logger.info("Formatting submission file and applying Threshold Squeezer...")
    sample_sub = pd.read_csv(SAMPLE_SUB_PATH)
    
    pred_cols = ["pred_week1", "pred_week2", "pred_week3", "pred_week4", "pred_week5"]
    submission = sample_sub[["region_id"]].merge(pred_df, on="region_id", how="left")
    submission[pred_cols] = submission[pred_cols].fillna(0.0)
    
    # --- THRESHOLD SQUEEZER ---
    # Force any prediction below 0.4 straight to absolute 0.0
    submission[pred_cols] = submission[pred_cols].where(submission[pred_cols] >= 0.4, 0.0)
    
    submission[pred_cols] = submission[pred_cols].round(4)

    out_path = SUBMISSION_DIR / "submission_lstm_v2.csv"
    submission.to_csv(out_path, index=False)
    
    logger.info(f"Success! Submission saved to: {out_path}")

if __name__ == "__main__":
    main()