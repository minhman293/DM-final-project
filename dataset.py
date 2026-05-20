import torch
from torch.utils.data import Dataset
import numpy as np

from config_lstm import SEQ_LEN, PRED_LEN, FEATURES

class DisasterSequenceDataset(Dataset):
    def __init__(self, weekly_df, region_encoder, scaler, is_train=True):
        self.is_train = is_train
        self.X_seqs = []
        self.y_seqs = []
        self.region_idxs = []
        self.baselines = []
        self.summers = []
        
        # Group by region to extract sequences safely
        for region_id, group in weekly_df.groupby("region_id"):
            group = group.sort_values("week_id").reset_index(drop=True)
            r_idx = region_encoder[region_id]
            
            # Scale features
            features_scaled = scaler.transform(group[FEATURES])
            baseline = group["region_mean_score"].iloc[0]
            
            n_weeks = len(group)
            
            if is_train:
                scores = group["score"].values
                months = group["month"].values
                
                # Sliding window to create massive training set
                for i in range(n_weeks - SEQ_LEN - PRED_LEN + 1):
                    self.X_seqs.append(features_scaled[i : i + SEQ_LEN])
                    self.y_seqs.append(scores[i + SEQ_LEN : i + SEQ_LEN + PRED_LEN])
                    self.region_idxs.append(r_idx)
                    self.baselines.append(baseline)
                    
                    # EDA Insight: Check if the prediction window starts in Summer (Months 5-10)
                    target_month = months[i + SEQ_LEN]
                    is_summer = 1.0 if target_month in [5, 6, 7, 8, 9, 10] else 0.0
                    self.summers.append(is_summer)
            else:
                # For test set, we only have exactly 13 weeks per region
                self.X_seqs.append(features_scaled[-SEQ_LEN:])
                self.region_idxs.append(r_idx)
                self.baselines.append(baseline)

    def __len__(self):
        return len(self.X_seqs)

    def __getitem__(self, idx):
        x = torch.tensor(self.X_seqs[idx], dtype=torch.float32)
        r = torch.tensor(self.region_idxs[idx], dtype=torch.long)
        b = torch.tensor(self.baselines[idx], dtype=torch.float32)
        
        if self.is_train:
            y = torch.tensor(self.y_seqs[idx], dtype=torch.float32)
            s = torch.tensor(self.summers[idx], dtype=torch.float32)
            return x, r, b, y, s
            
        return x, r, b