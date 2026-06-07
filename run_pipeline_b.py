# run_pipeline_b.py
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import numpy as np
import pandas as pd
import math
import joblib

from config_pipeline_b import *

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# =====================================================================
# 1. DATASET
# =====================================================================
class NumpyDroughtDataset(Dataset):
    def __init__(self, features, targets=None, is_test=False):
        self.features = features
        self.targets = targets
        self.is_test = is_test
        self.anchors = []
        num_regions, num_weeks, _ = self.features.shape
        
        if is_test:
            for r in range(num_regions):
                self.anchors.append((r, num_weeks - 1))
        else:
            for r in range(num_regions):
                for t in range(HISTORY - 1, num_weeks - HORIZONS):
                    self.anchors.append((r, t))

    def __len__(self): return len(self.anchors)

    def __getitem__(self, idx):
        r, t = self.anchors[idx]
        X = self.features[r, t - HISTORY + 1 : t + 1, :]
        if self.is_test:
            return torch.FloatTensor(X), torch.LongTensor([r]), torch.zeros(HORIZONS)
        else:
            y = self.targets[r, t + 1 : t + 1 + HORIZONS]
            return torch.FloatTensor(X), torch.LongTensor([r]), torch.FloatTensor(y)

# =====================================================================
# 2. MODELS
# =====================================================================
class AutoGRU(nn.Module):
    def __init__(self):
        super().__init__()
        H = AUTOGRU_HIDDEN; R = AUTOGRU_REGION_DIM; D = AUTOGRU_DROPOUT
        self.encoder  = nn.GRU(N_VARS, H, batch_first=True, bidirectional=True, num_layers=2, dropout=D)
        self.enc_proj = nn.Sequential(nn.Linear(H*2,H), nn.LayerNorm(H), nn.SiLU())
        self.reg_emb  = nn.Embedding(N_REGIONS, R)
        self.decoder  = nn.GRUCell(1+R, H)
        self.out_proj = nn.Sequential(nn.Linear(H,32), nn.SiLU(), nn.Linear(32,1))
        self.drop     = nn.Dropout(D)

    def encode(self, seq):
        out, _ = self.encoder(seq)          
        return self.enc_proj(out[:,-1,:])   

    def forward(self, seq, region):
        B = seq.size(0)
        h = self.encode(seq)               
        rg = self.reg_emb(region).squeeze(1)          
        prev = torch.zeros(B,1, device=seq.device)
        preds = []
        for t in range(HORIZONS):
            inp  = torch.cat([prev, rg], 1)  
            h    = self.decoder(inp, h)
            pred = self.out_proj(self.drop(h))           
            preds.append(pred)
            prev = pred.detach()
        return torch.cat(preds,1)

class DLinear(nn.Module):
    def __init__(self):
        super().__init__()
        k = DLINEAR_MA_KERNEL; R = DLINEAR_REGION_DIM
        self.ma  = nn.AvgPool1d(k,1,k//2)
        dim      = N_VARS*HISTORY
        self.lt  = nn.Linear(dim, HORIZONS)
        self.ls  = nn.Linear(dim, HORIZONS)
        self.reg_emb  = nn.Embedding(N_REGIONS, R)
        self.reg_proj = nn.Linear(R, HORIZONS)
    def forward(self, seq, region):
        x  = seq.permute(0,2,1)           
        tr = self.ma(x)[:,:,:HISTORY]     
        se = x - tr
        reg = self.reg_emb(region).squeeze(1)
        return self.lt(tr.flatten(1)) + self.ls(se.flatten(1)) + self.reg_proj(reg)

class CausalBlock(nn.Module):
    def __init__(self, ic, oc, k=TCN_KERNEL, d=1):
        super().__init__()
        p = (k-1)*d
        self.c1   = nn.Conv1d(ic,oc,k,dilation=d,padding=p)
        self.c2   = nn.Conv1d(oc,oc,k,dilation=d,padding=p)
        self.n1   = nn.BatchNorm1d(oc); self.n2 = nn.BatchNorm1d(oc)
        self.act  = nn.GELU(); self.drop = nn.Dropout(TCN_DROPOUT)
        self.skip = nn.Conv1d(ic,oc,1) if ic!=oc else nn.Identity()
        self._p   = p
    def _trim(self, x, conv): return conv(x)[:,:,:-self._p] if self._p else conv(x)
    def forward(self, x):
        o = self.drop(self.act(self.n1(self._trim(x, self.c1))))
        o = self.drop(self.act(self.n2(self._trim(o, self.c2))))
        return self.act(o + self.skip(x))

class TCN(nn.Module):
    def __init__(self):
        super().__init__()
        ch=TCN_CHANNELS; R=TCN_REGION_DIM
        self.proj   = nn.Conv1d(N_VARS, ch[0], 1)
        self.blocks = nn.ModuleList([CausalBlock(ch[i],ch[i+1],d=2**i) for i in range(len(ch)-1)])
        self.reg_emb= nn.Embedding(N_REGIONS, R)
        self.head   = nn.Sequential(nn.Linear(ch[-1]+R,64), nn.LayerNorm(64), nn.SiLU(), nn.Dropout(TCN_DROPOUT), nn.Linear(64,HORIZONS))
    def forward(self, seq, region):
        x = self.proj(seq.permute(0,2,1))
        for b in self.blocks: x=b(x)
        return self.head(torch.cat([x[:,:,-1], self.reg_emb(region).squeeze(1)],1))

# =====================================================================
# 3. TRAINING ENGINE
# =====================================================================
def train_model(model_name, model, train_loader, val_loader):
    print(f"\n--- Training {model_name} ---")
    model = model.to(DEVICE)
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
    criterion = nn.SmoothL1Loss()
    best_loss = float('inf')
    
    for epoch in range(1, EPOCHS + 1):
        model.train()
        for X, r, y in train_loader:
            X, r, y = X.to(DEVICE), r.to(DEVICE), y.to(DEVICE)
            optimizer.zero_grad()
            loss = criterion(model(X, r), y)
            loss.backward()
            optimizer.step()
            
        model.eval()
        val_loss = 0
        with torch.no_grad():
            for X, r, y in val_loader:
                X, r, y = X.to(DEVICE), r.to(DEVICE), y.to(DEVICE)
                preds = torch.clamp(model(X, r), 0.0, 5.0)
                val_loss += torch.abs(preds - y).mean().item()
                
        val_loss /= len(val_loader)
        if val_loss < best_loss:
            best_loss = val_loss
            torch.save(model.state_dict(), MODEL_DIR / f"{model_name}.pth")
        print(f"Epoch {epoch} | Val MAE: {val_loss:.4f}")

def main():
    print("1. Loading Numpy Arrays...")
    train_features = np.load(MODEL_DIR / "train_features.npy")
    train_targets = np.load(MODEL_DIR / "train_targets.npy")
    test_features = np.load(MODEL_DIR / "test_features.npy")
    region_to_idx = joblib.load(MODEL_DIR / "region_to_idx.pkl")
    
    split_idx = int(train_features.shape[1] * 0.8)
    val_features, val_targets = train_features[:, split_idx:, :], train_targets[:, split_idx:]
    train_features, train_targets = train_features[:, :split_idx, :], train_targets[:, :split_idx]
    
    train_loader = DataLoader(NumpyDroughtDataset(train_features, train_targets), batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(NumpyDroughtDataset(val_features, val_targets), batch_size=BATCH_SIZE, shuffle=False)
    test_loader = DataLoader(NumpyDroughtDataset(test_features, is_test=True), batch_size=BATCH_SIZE, shuffle=False)
    
    # Excluded PatchTST/FusedTCN for speed, but you can add them back if you have hours!
    models = { "AutoGRU": AutoGRU(), "DLinear": DLinear(), "TCN": TCN() }
    idx_to_region = {v: k for k, v in region_to_idx.items()}
    
    for name, model in models.items():
        train_model(name, model, train_loader, val_loader)
        
        model.load_state_dict(torch.load(MODEL_DIR / f"{name}.pth"))
        model.eval()
        all_preds, all_regions = [], []
        with torch.no_grad():
            for X, r, _ in test_loader:
                X, r = X.to(DEVICE), r.to(DEVICE)
                all_preds.append(torch.clamp(model(X, r), 0.0, 5.0).cpu().numpy())
                all_regions.extend([idx_to_region[idx.item()] for idx in r])
                
        df = pd.DataFrame(np.concatenate(all_preds, axis=0), columns=[f"pred_week{i+1}" for i in range(5)])
        df.insert(0, "region_id", all_regions)
        df.to_csv(SUB_DIR / f"submission_{name.lower()}.csv", index=False)
        print(f"Saved {name} predictions!")

if __name__ == "__main__":
    main()