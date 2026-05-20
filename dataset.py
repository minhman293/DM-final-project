import torch
from torch.utils.data import Dataset
import numpy as np
import pandas as pd
from config_lstm import SEQ_LEN, PRED_LEN, FEATURES

class DisasterSequenceDataset(Dataset):
    def __init__(self, weekly_df, region_encoder, scaler, is_train=True):
        self.is_train = is_train
        self.X_seqs = []
        self.y_seqs = []
        self.region_idxs = []
        
        # Group by region to extract sequences safely
        for region_id, group in weekly_df.groupby("region_id"):
            group = group.sort_values("week_id").reset_index(drop=True)
            r_idx = region_encoder[region_id]
            
            # Scale features
            features_scaled = scaler.transform(group[FEATURES])
            scores = group["score"].values if "score" in group.columns else None
            
            n_weeks = len(group)
            
            if is_train:
                # Sliding window to create massive training set
                for i in range(n_weeks - SEQ_LEN - PRED_LEN + 1):
                    # 13 weeks input
                    seq_x = features_scaled[i : i + SEQ_LEN]
                    # 5 weeks target
                    seq_y = scores[i + SEQ_LEN : i + SEQ_LEN + PRED_LEN]
                    
                    self.X_seqs.append(seq_x)
                    self.y_seqs.append(seq_y)
                    self.region_idxs.append(r_idx)
            else:
                # For test set, we only have exactly 13 weeks per region
                seq_x = features_scaled[-SEQ_LEN:]
                self.X_seqs.append(seq_x)
                self.region_idxs.append(r_idx)

    def __len__(self):
        return len(self.X_seqs)

    def __getitem__(self, idx):
        x = torch.tensor(self.X_seqs[idx], dtype=torch.float32)
        r = torch.tensor(self.region_idxs[idx], dtype=torch.long)
        
        if self.is_train:
            y = torch.tensor(self.y_seqs[idx], dtype=torch.float32)
            return x, r, y
        return x, r