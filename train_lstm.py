import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from sklearn.preprocessing import StandardScaler, LabelEncoder
import pandas as pd
import numpy as np
import joblib

# Import your weekly aggregation function from feature_engineering.py
from feature_engineering import aggregate_to_weekly
from dataset import DisasterSequenceDataset
from model import DisasterLSTM
from config_lstm import *

def train_loop():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Training on {device}")

    # 1. Load & Aggregate
    train_raw = pd.read_csv(TRAIN_PATH)
    from utils import extract_date_parts
    train_raw = extract_date_parts(train_raw)
    train_weekly = aggregate_to_weekly(train_raw, is_train=True)

    # 2. Encoders & Scalers
    region_encoder = LabelEncoder()
    train_weekly["region_idx"] = region_encoder.fit_transform(train_weekly["region_id"])
    
    scaler = StandardScaler()
    scaler.fit(train_weekly[FEATURES])
    
    # Save encoders for inference
    joblib.dump(region_encoder, MODEL_DIR / "region_encoder.pkl")
    joblib.dump(scaler, MODEL_DIR / "scaler.pkl")

    # 3. Create Dataset
    # Note: For a real run, split val by regions using your region_split function
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
    criterion = nn.L1Loss() # MAE Loss

    # 5. Training Loop
    for epoch in range(EPOCHS):
        model.train()
        total_loss = 0
        
        for batch_X, batch_r, batch_y in train_loader:
            batch_X, batch_r, batch_y = batch_X.to(device), batch_r.to(device), batch_y.to(device)
            
            optimizer.zero_grad()
            preds = model(batch_X, batch_r)
            
            loss = criterion(preds, batch_y)
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            
        print(f"Epoch {epoch+1}/{EPOCHS} | Train MAE: {total_loss/len(train_loader):.4f}")

    torch.save(model.state_dict(), MODEL_DIR / "lstm_weights.pth")
    print("Training Complete.")

if __name__ == "__main__":
    train_loop()