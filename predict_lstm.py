import torch
import pandas as pd
import numpy as np
import joblib
import json

from config_lstm import *
from feature_engineering import aggregate_to_weekly
from utils import extract_date_parts, get_logger
from model import DisasterLSTM

logger = get_logger("predict_lstm")

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Running inference on {device}...")

    # 1. Load Raw Data and Aggregate
    logger.info("Loading test data...")
    test_raw = pd.read_csv(TEST_PATH)
    test_raw = extract_date_parts(test_raw)
    
    # is_train=False ensures we keep all weeks (since test has no score column)
    test_weekly = aggregate_to_weekly(test_raw, is_train=False)

    # 2. Load Encoders, Scalers, and Model Weights
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
    model.eval() # Set model to evaluation mode

    # 3. Generate Predictions
    logger.info("Generating predictions...")
    predictions = []
    
    # Process each region one by one
    with torch.no_grad():
        for region_id, group in test_weekly.groupby("region_id"):
            # Ensure chronological order
            group = group.sort_values("week_id").reset_index(drop=True)
            
            # The test set has exactly 13 weeks (91 days) per region. 
            # We take the scaled features for this sequence.
            seq_x = scaler.transform(group[FEATURES].values[-SEQ_LEN:])
            r_idx = region_encoder.transform([region_id])[0]
            
            # Convert to tensors and add batch dimension (unsqueeze)
            seq_x_tensor = torch.tensor(seq_x, dtype=torch.float32).unsqueeze(0).to(device)
            r_idx_tensor = torch.tensor([r_idx], dtype=torch.long).to(device)
            
            # Predict
            pred = model(seq_x_tensor, r_idx_tensor)
            
            # Move to CPU, convert to numpy, and flatten
            pred_np = pred.cpu().numpy().flatten()
            
            # Clip predictions to valid Kaggle score range [0, 5]
            pred_np = np.clip(pred_np, 0.0, 5.0)
            
            # Store results
            predictions.append({
                "region_id": region_id,
                "pred_week1": pred_np[0],
                "pred_week2": pred_np[1],
                "pred_week3": pred_np[2],
                "pred_week4": pred_np[3],
                "pred_week5": pred_np[4]
            })

    pred_df = pd.DataFrame(predictions)

    # 4. Format for Kaggle Submission
    logger.info("Formatting submission file...")
    sample_sub = pd.read_csv(SAMPLE_SUB_PATH)
    
    # Merge to guarantee the exact row order Kaggle expects
    pred_cols = ["pred_week1", "pred_week2", "pred_week3", "pred_week4", "pred_week5"]
    submission = sample_sub[["region_id"]].merge(pred_df, on="region_id", how="left")
    
    # Safety check: fill any missing regions with 0.0
    submission[pred_cols] = submission[pred_cols].fillna(0.0)
    
    # --- OPTIONAL THRESHOLDING TRICK ---
    # Because 59.6% of scores are 0, we can force low-confidence predictions to 0
    # Uncomment the line below if your model over-predicts drought
    # submission[pred_cols] = submission[pred_cols].where(submission[pred_cols] >= 0.4, 0.0)
    
    # Round to 4 decimal places for clean output
    submission[pred_cols] = submission[pred_cols].round(4)

    # 5. Save Submission
    out_path = SUBMISSION_DIR / "submission_lstm.csv"
    submission.to_csv(out_path, index=False)
    
    logger.info(f"Success! Submission saved to: {out_path}")
    logger.info("Preview:")
    logger.info("\n" + submission.head().to_string())

if __name__ == "__main__":
    main()