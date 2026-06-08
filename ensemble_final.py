import pandas as pd
import numpy as np
from pathlib import Path

def main():
    # print("1. Loading all submission CSVs...")
    # sub_dir = Path("submissions")
    
    # # 1. Load the models (Ensure these filenames match exactly what is in your folder)
    # tft_42  = pd.read_csv(sub_dir / "submission_tft_v4_relu.csv")
    # tft_123 = pd.read_csv(sub_dir / "submission_tft_v4_relu_123.csv")
    # tft_999 = pd.read_csv(sub_dir / "submission_tft_v4_relu_999.csv")
    
    # lstm    = pd.read_csv(sub_dir / "submission_lstm_v2.csv") # Your golden 0.90 anchor
    
    # # The new Pipeline B models
    # autogru = pd.read_csv(sub_dir / "submission_autogru.csv")
    # dlinear = pd.read_csv(sub_dir / "submission_dlinear.csv")
    # tcn     = pd.read_csv(sub_dir / "submission_tcn.csv")
    
    # pred_cols = ["pred_week1", "pred_week2", "pred_week3", "pred_week4", "pred_week5"]
    
    # print("2. Building the Super-TFT...")
    # super_tft = tft_42.copy()
    # for col in pred_cols:
    #     super_tft[col] = (tft_42[col] + tft_123[col] + tft_999[col]) / 3.0
        
    # print("3. Applying Master Heterogeneous Blend...")
    # # WEIGHTING STRATEGY:
    # # Super-TFT is our strongest model -> 55%
    # # LSTM V2 is our zero-anchor -> 15%
    # # DLinear catches mathematical trends -> 10%
    # # TCN catches sudden spikes -> 10%
    # # AutoGRU provides alternative recurrence -> 10%
    
    # weights = {
    #     "SuperTFT": 0.55,
    #     "LSTM":     0.15,
    #     "DLinear":  0.10,
    #     "TCN":      0.10,
    #     "AutoGRU":  0.10
    # }
    
    # assert sum(weights.values()) == 1.0, "Weights must sum to 1.0!"
    
    # master_sub = super_tft.copy()
    
    # for col in pred_cols:
    #     master_sub[col] = (
    #         (super_tft[col] * weights["SuperTFT"]) +
    #         (lstm[col]      * weights["LSTM"]) +
    #         (dlinear[col]   * weights["DLinear"]) +
    #         (tcn[col]       * weights["TCN"]) +
    #         (autogru[col]   * weights["AutoGRU"])
    #     )
    #     # Optional safe-clamp just in case
    #     master_sub[col] = np.clip(master_sub[col], 0.0, 5.0)
    #     master_sub[col] = master_sub[col].round(4)
        
    # out_path = sub_dir / "SUBMISSION.csv"
    # master_sub.to_csv(out_path, index=False)
    # print(f"\nSuccess! Master blend saved to: {out_path}")
    # print("Upload this to Kaggle. Let's break Baseline 3!")

    # ver 2: tft + autogru + lstm 
    # Load your existing CSVs
    tft_42  = pd.read_csv("submissions/submission_tft_v4_relu.csv")
    tft_123 = pd.read_csv("submissions/submission_tft_v4_relu_123.csv")
    tft_999 = pd.read_csv("submissions/submission_tft_v4_relu_999.csv")
    
    lstm    = pd.read_csv("submissions/submission_lstm_v2.csv")
    autogru = pd.read_csv("submissions/submission_autogru.csv")
    
    pred_cols = ["pred_week1", "pred_week2", "pred_week3", "pred_week4", "pred_week5"]
    
    # 1. Build the Super-TFT (Averages the 3 seeds)
    super_tft = tft_42.copy()
    for col in pred_cols:
        super_tft[col] = (tft_42[col] + tft_123[col] + tft_999[col]) / 3.0
        
    # 2. The Pruned Blend
    # TFT is so strong it needs 75% of the vote.
    # AutoGRU gets 15% to add Recurrent variance.
    # LSTM gets 10% to pull extreme values down towards zero.
    weights = {
        "SuperTFT": 0.75,
        "AutoGRU":  0.15,
        "LSTM":     0.10
    }
    
    pruned_sub = super_tft.copy()
    
    for col in pred_cols:
        pruned_sub[col] = (
            (super_tft[col] * weights["SuperTFT"]) +
            (autogru[col]   * weights["AutoGRU"]) +
            (lstm[col]      * weights["LSTM"])
        )
        # Squeeze values very close to zero to EXACTLY zero (Kaggle MAE Hack)
        # If the ensemble predicts 0.05, it's basically a zero-drought week.
        pruned_sub[col] = np.where(pruned_sub[col] < 0.05, 0.0, pruned_sub[col])
        pruned_sub[col] = np.clip(pruned_sub[col], 0.0, 5.0).round(4)
        
    pruned_sub.to_csv("submissions/ensemble_75tft_15autogru_10lstm.csv", index=False)
    print("Pruned ensemble saved! Submit this.")

if __name__ == "__main__":
    main()