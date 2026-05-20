import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from sklearn.preprocessing import StandardScaler, LabelEncoder
import pandas as pd
import joblib

from feature_engineering import aggregate_to_weekly
from dataset import DisasterSequenceDataset
from model import DisasterLSTM
from config_lstm import *
from utils import extract_date_parts, compute_region_baseline

def train_loop():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Training on {device}")

    # 1. Load, Extract Dates, and Compute Baselines
    train_raw = pd.read_csv(TRAIN_PATH)
    train_raw = extract_date_parts(train_raw)
    
    # Calculate Region Baselines (Anchor Fix)
    baselines = compute_region_baseline(train_raw)
    
    # Aggregate and Merge
    train_weekly = aggregate_to_weekly(train_raw, is_train=True)
    train_weekly = train_weekly.merge(baselines, on="region_id", how="left")
    train_weekly["region_mean_score"] = train_weekly["region_mean_score"].fillna(0.0)

    # 2. Encoders & Scalers
    region_encoder = LabelEncoder()
    train_weekly["region_idx"] = region_encoder.fit_transform(train_weekly["region_id"])
    
    scaler = StandardScaler()
    scaler.fit(train_weekly[FEATURES])
    
    joblib.dump(region_encoder, MODEL_DIR / "region_encoder.pkl")
    joblib.dump(scaler, MODEL_DIR / "scaler.pkl")

    # 3. Create Dataset
    train_dataset = DisasterSequenceDataset(
        train_weekly, 
        dict(zip(region_encoder.classes_, region_encoder.transform(region_encoder.classes_))), 
        scaler, 
        is_train=True
    )
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)

    # 4. Initialize Model
    model = DisasterLSTM(
        num_regions=len(region_encoder.classes_),
        num_features=len(FEATURES),
        embed_dim=EMBED_DIM,
        hidden_size=HIDDEN_SIZE,
        num_layers=NUM_LAYERS,
        pred_len=PRED_LEN
    ).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
    
    # We use reduction='none' so we can apply our custom seasonal weights before averaging
    criterion = nn.L1Loss(reduction='none') 

    # 5. Training Loop
    for epoch in range(EPOCHS):
        model.train()
        total_loss = 0
        
        for batch_X, batch_r, batch_b, batch_y, batch_summer in train_loader:
            batch_X = batch_X.to(device)
            batch_r = batch_r.to(device)
            batch_b = batch_b.to(device)
            batch_y = batch_y.to(device)
            batch_summer = batch_summer.to(device)
            
            optimizer.zero_grad()
            preds = model(batch_X, batch_r, batch_b)
            
            # Architectural Fix: Custom Seasonal Loss
            raw_loss = criterion(preds, batch_y)
            # Apply 3.0x multiplier if summer, 1.0x if winter. unsqueeze(1) aligns the dimensions.
            weights = torch.where(batch_summer == 1.0, 3.0, 1.0).unsqueeze(1)
            weighted_loss = (raw_loss * weights).mean()
            
            weighted_loss.backward()
            optimizer.step()
            
            total_loss += weighted_loss.item()
            
        print(f"Epoch {epoch+1}/{EPOCHS} | Train Weighted MAE: {total_loss/len(train_loader):.4f}")

    torch.save(model.state_dict(), MODEL_DIR / "lstm_weights.pth")
    print("Training Complete.")

if __name__ == "__main__":
    train_loop()