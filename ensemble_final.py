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
    # # Load your existing CSVs
    # tft_42  = pd.read_csv("submissions/submission_tft_v4_relu.csv")
    # tft_123 = pd.read_csv("submissions/submission_tft_v4_relu_123.csv")
    # tft_999 = pd.read_csv("submissions/submission_tft_v4_relu_999.csv")
    
    # lstm    = pd.read_csv("submissions/submission_lstm_v2.csv")
    # autogru = pd.read_csv("submissions/submission_autogru.csv")
    
    # pred_cols = ["pred_week1", "pred_week2", "pred_week3", "pred_week4", "pred_week5"]
    
    # # 1. Build the Super-TFT (Averages the 3 seeds)
    # super_tft = tft_42.copy()
    # for col in pred_cols:
    #     super_tft[col] = (tft_42[col] + tft_123[col] + tft_999[col]) / 3.0
        
    # # 2. The Pruned Blend
    # # TFT is so strong it needs 75% of the vote.
    # # AutoGRU gets 15% to add Recurrent variance.
    # # LSTM gets 10% to pull extreme values down towards zero.
    # weights = {
    #     "SuperTFT": 0.75,
    #     "AutoGRU":  0.15,
    #     "LSTM":     0.10
    # }
    
    # pruned_sub = super_tft.copy()
    
    # for col in pred_cols:
    #     pruned_sub[col] = (
    #         (super_tft[col] * weights["SuperTFT"]) +
    #         (autogru[col]   * weights["AutoGRU"]) +
    #         (lstm[col]      * weights["LSTM"])
    #     )
    #     # Squeeze values very close to zero to EXACTLY zero (Kaggle MAE Hack)
    #     # If the ensemble predicts 0.05, it's basically a zero-drought week.
    #     pruned_sub[col] = np.where(pruned_sub[col] < 0.05, 0.0, pruned_sub[col])
    #     pruned_sub[col] = np.clip(pruned_sub[col], 0.0, 5.0).round(4)
        
    # pruned_sub.to_csv("submissions/ensemble_75tft_15autogru_10lstm.csv", index=False)
    # print("Pruned ensemble saved! Submit this.")

    # ver 3: tft + lgbm + lstm
    # # 1. Load the models
    # tft_42  = pd.read_csv("submissions/submission_tft_v4_relu.csv")
    # tft_123 = pd.read_csv("submissions/submission_tft_v4_relu_123.csv")
    # tft_999 = pd.read_csv("submissions/submission_tft_v4_relu_999.csv")
    
    # lgbm    = pd.read_csv("submissions/submission_lgbm_trees.csv")
    # lstm    = pd.read_csv("submissions/submission_lstm_v2.csv") # The old 0.90 anchor
    
    # pred_cols = ["pred_week1", "pred_week2", "pred_week3", "pred_week4", "pred_week5"]
    
    # # 2. Build Super-TFT
    # super_tft = tft_42.copy()
    # for col in pred_cols:
    #     super_tft[col] = (tft_42[col] + tft_123[col] + tft_999[col]) / 3.0
        
    # # 3. The Ultimate Orthogonal Blend
    # # 60% TFT (The Deep Learning Brain)
    # # 30% LGBM (The Tree-Based Brain)
    # # 10% LSTM (The Zero-Gravity Anchor)
    
    # final_sub = super_tft.copy()
    # for col in pred_cols:
    #     final_sub[col] = (super_tft[col] * 0.60) + (lgbm[col] * 0.30) + (lstm[col] * 0.10)
        
    #     # Aggressive Kaggle MAE Squeezer
    #     final_sub[col] = np.where(final_sub[col] < 0.05, 0.0, final_sub[col])
    #     final_sub[col] = np.clip(final_sub[col], 0.0, 5.0).round(4)
        
    # final_sub.to_csv("submissions/tft_lgbm_lstm.csv", index=False)
    # print("Ultimate blend saved. Submit this to Kaggle!")

    # ver 4: tft + lstm gated
    # print("1. Loading the Champions...")
    # sub_dir = Path("submissions")
    
    # # Load your pure temporal sequence models
    # tft_42  = pd.read_csv(sub_dir / "submission_tft_v4_relu.csv")
    # tft_123 = pd.read_csv(sub_dir / "submission_tft_v4_relu_123.csv")
    # tft_999 = pd.read_csv(sub_dir / "submission_tft_v4_relu_999.csv")
    # lstm    = pd.read_csv(sub_dir / "submission_lstm_v2.csv") # The 0.90 anchor
    
    # pred_cols = ["pred_week1", "pred_week2", "pred_week3", "pred_week4", "pred_week5"]
    
    # print("2. Building the Super-TFT...")
    # super_tft = tft_42.copy()
    # for col in pred_cols:
    #     super_tft[col] = (tft_42[col] + tft_123[col] + tft_999[col]) / 3.0
        
    # print("3. Executing the LSTM Gate & MAE Squeeze...")
    # final_sub = super_tft.copy()
    
    # for col in pred_cols:
    #     # Base state: We trust the Super TFT with a tiny 10% anchor from the LSTM 
    #     # to gently smooth out variance.
    #     blended = (super_tft[col] * 0.90) + (lstm[col] * 0.10)
        
    #     # --- THE LSTM ZERO-GATE ---
    #     # If the LSTM is screaming "NO DROUGHT" (predicting < 0.15), 
    #     # it is usually right. We override the blend and crush it to exactly 0.0.
    #     gated = np.where(lstm[col] < 0.15, 0.0, blended)
        
    #     # --- THE MICRO-SQUEEZE ---
    #     # Kaggle MAE heavily punishes predicting 0.04 when the truth is 0.0.
    #     # If the gated result is still tiny (< 0.05), it is mathematical noise. 
    #     # Force it to absolute zero.
    #     gated = np.where(gated < 0.05, 0.0, gated)
        
    #     # Safe boundary clamp and round for Kaggle formatting
    #     final_sub[col] = np.clip(gated, 0.0, 5.0).round(4)
        
    # out_path = sub_dir / "tft_lstm_GATED.csv"
    # final_sub.to_csv(out_path, index=False)
    # print(f"\nSuccess! Saved to {out_path}")

    # ver 5: different weights tft + lstm
    # sub_dir = Path("submissions")
    
    # # 1. Load the two champions
    # tft_42  = pd.read_csv(sub_dir / "submission_tft_v4_relu.csv")
    # tft_123 = pd.read_csv(sub_dir / "submission_tft_v4_relu_123.csv")
    # tft_999 = pd.read_csv(sub_dir / "submission_tft_v4_relu_999.csv")
    
    # lstm = pd.read_csv(sub_dir / "submission_lstm_v2.csv") # 0.9029
    
    # pred_cols = ["pred_week1", "pred_week2", "pred_week3", "pred_week4", "pred_week5"]
    
    # # 2. Build the Super-TFT (0.8293)
    # super_tft = tft_42.copy()
    # for col in pred_cols:
    #     super_tft[col] = (tft_42[col] + tft_123[col] + tft_999[col]) / 3.0
        
    # # 3. The Golden Ratio Sweep (PURE LINEAR BLENDING ONLY)
    # ratios = [
    #     (0.80, 0.20), # 80% TFT / 20% LSTM
    #     (0.85, 0.15), # 85% TFT / 15% LSTM
    #     (0.90, 0.10)  # 90% TFT / 10% LSTM
    # ]
    
    # for tft_weight, lstm_weight in ratios:
    #     blend = super_tft.copy()
    #     for col in pred_cols:
    #         blend[col] = (super_tft[col] * tft_weight) + (lstm[col] * lstm_weight)
            
    #         # The only rule: The Kaggle target cannot mathematically go below 0 or above 5.
    #         blend[col] = np.clip(blend[col], 0.0, 5.0).round(4)
            
    #     file_name = f"ENSEMBLE_{int(tft_weight*100)}TFT_{int(lstm_weight*100)}LSTM.csv"
    #     blend.to_csv(sub_dir / file_name, index=False)
    #     print(f"Generated: {file_name}") 

    # ver 6: same as v5
    # sub_dir = Path("submissions")
    
    # # Load your models
    # tft_42  = pd.read_csv(sub_dir / "submission_tft_v4_relu.csv")
    # tft_123 = pd.read_csv(sub_dir / "submission_tft_v4_relu_123.csv")
    # tft_999 = pd.read_csv(sub_dir / "submission_tft_v4_relu_999.csv")
    # lstm    = pd.read_csv(sub_dir / "submission_lstm_v2.csv") 
    
    # pred_cols = ["pred_week1", "pred_week2", "pred_week3", "pred_week4", "pred_week5"]
    
    # # Build Super-TFT
    # super_tft = tft_42.copy()
    # for col in pred_cols:
    #     super_tft[col] = (tft_42[col] + tft_123[col] + tft_999[col]) / 3.0
        
    # # The "Down-Slope" Sweep
    # ratios = [
    #     (0.60, 0.40), # 60% TFT / 40% LSTM
    #     (0.50, 0.50), # 50% TFT / 50% LSTM
    #     (0.40, 0.60)  # 40% TFT / 60% LSTM
    # ]
    
    # for tft_weight, lstm_weight in ratios:
    #     blend = super_tft.copy()
    #     for col in pred_cols:
    #         blend[col] = (super_tft[col] * tft_weight) + (lstm[col] * lstm_weight)
            
    #         # The Kaggle MAE Squeezer (Crucial for pushing the score down)
    #         # If the blend is extremely close to zero, force it to absolute zero.
    #         blend[col] = np.where(blend[col] < 0.05, 0.0, blend[col])
    #         blend[col] = np.clip(blend[col], 0.0, 5.0).round(4)
            
    #     file_name = f"ENSEMBLE_{int(tft_weight*100)}TFT_{int(lstm_weight*100)}LSTM.csv"
    #     blend.to_csv(sub_dir / file_name, index=False)
    #     print(f"Generated: {file_name}")

    # ver 7: same as v6
    sub_dir = Path("submissions")
    
    # Load the two champions
    tft_42  = pd.read_csv(sub_dir / "submission_tft_v4_relu.csv")
    tft_123 = pd.read_csv(sub_dir / "submission_tft_v4_relu_123.csv")
    tft_999 = pd.read_csv(sub_dir / "submission_tft_v4_relu_999.csv")
    lstm    = pd.read_csv(sub_dir / "submission_lstm_v2.csv") 
    
    pred_cols = ["pred_week1", "pred_week2", "pred_week3", "pred_week4", "pred_week5"]
    
    # Build Super-TFT
    # super_tft = tft_42.copy()
    # for col in pred_cols:
    #     super_tft[col] = (tft_42[col] + tft_123[col] + tft_999[col]) / 3.0
        
    # # The Mathematical Sweet Spot: 64% TFT and 36% LSTM
    # tft_weight = 0.64
    # lstm_weight = 0.36
    
    # final_blend = super_tft.copy()
    # for col in pred_cols:
    #     final_blend[col] = (super_tft[col] * tft_weight) + (lstm[col] * lstm_weight)
        
    #     # # Post-processing squeeze
    #     # final_blend[col] = np.where(final_blend[col] < 0.05, 0.0, final_blend[col])
    #     # final_blend[col] = np.clip(final_blend[col], 0.0, 5.0).round(4)

    #     # Shift down by 0.05, then clamp. 
    #     # This turns 0.04 -> 0.00, but keeps 1.5 -> 1.45 (preserving the drought)
    #     final_blend[col] = np.clip(final_blend[col] - 0.05, 0.0, 5.0).round(4)
        
    # file_name = "ENSEMBLE_64TFT_36LSTM_v2.csv"
    # final_blend.to_csv(sub_dir / file_name, index=False)
    # print(f"Generated your final submission file: {file_name}")

    # ver 8
    # super_tft = tft_42.copy()
    # for col in pred_cols:
    #     super_tft[col] = (tft_42[col] + tft_123[col] + tft_999[col]) / 3.0
        
    # # The Mathematical Sweet Spot: 64% TFT and 36% LSTM
    # tft_weight = 0.85
    # lstm_weight = 0.15
    
    # final_blend = super_tft.copy()
    # for col in pred_cols:
    #     final_blend[col] = (super_tft[col] * tft_weight) + (lstm[col] * lstm_weight)
        
    #     # # Post-processing squeeze
    #     # final_blend[col] = np.where(final_blend[col] < 0.05, 0.0, final_blend[col])
    #     # final_blend[col] = np.clip(final_blend[col], 0.0, 5.0).round(4)

    #     # Shift down by 0.05, then clamp. 
    #     # This turns 0.04 -> 0.00, but keeps 1.5 -> 1.45 (preserving the drought)
    #     final_blend[col] = np.clip(final_blend[col] - 0.05, 0.0, 5.0).round(4)
        
    # file_name = "ENSEMBLE_85TFT_15LSTM_log.csv"
    # final_blend.to_csv(sub_dir / file_name, index=False)
    # print(f"Generated your final submission file: {file_name}")

    #ver 9
    # # Load your 0.8192 champion file
    # final_blend = pd.read_csv("submissions/ENSEMBLE_64TFT_36LSTM.csv")
    # pred_cols = ["pred_week1", "pred_week2", "pred_week3", "pred_week4", "pred_week5"]

    # for col in pred_cols:
    #     # A tiny power exponent (e.g., 1.05) pushes 0.5 down to 0.47, 
    #     # but leaves 4.0 mostly intact at 4.2 (which will get clipped back down)
    #     final_blend[col] = np.power(final_blend[col], 1.05)
    #     final_blend[col] = np.clip(final_blend[col], 0.0, 5.0).round(4)
        
    # final_blend.to_csv("submissions/ENSEMBLE_SQUEEZE_1.05.csv", index=False)

    # ver 10 currently best
    # Load your current 0.8192 champion
    sub = pd.read_csv("submissions/ENSEMBLE_64TFT_36LSTM.csv")
    
    smoothed = sub.copy()
    
    # Extract the prediction arrays
    w1 = sub["pred_week1"].values
    w2 = sub["pred_week2"].values
    w3 = sub["pred_week3"].values
    w4 = sub["pred_week4"].values
    w5 = sub["pred_week5"].values
    
    # Apply a mild 20% smoothing from neighboring weeks
    # Week 1 only has Week 2 as a future neighbor
    smoothed["pred_week1"] = (w1 * 0.8) + (w2 * 0.2)
    
    # Middle weeks pull 10% from the past and 10% from the future
    smoothed["pred_week2"] = (w2 * 0.8) + (w1 * 0.1) + (w3 * 0.1)
    smoothed["pred_week3"] = (w3 * 0.8) + (w2 * 0.1) + (w4 * 0.1)
    smoothed["pred_week4"] = (w4 * 0.8) + (w3 * 0.1) + (w5 * 0.1)
    
    # Week 5 only has Week 4 as a past neighbor
    smoothed["pred_week5"] = (w5 * 0.8) + (w4 * 0.2)
    
    # Re-apply the baseline Kaggle constraints
    pred_cols = ["pred_week1", "pred_week2", "pred_week3", "pred_week4", "pred_week5"]
    for col in pred_cols:
        smoothed[col] = np.where(smoothed[col] < 0.05, 0.0, smoothed[col])
        smoothed[col] = np.clip(smoothed[col], 0.0, 5.0).round(4)
        
    smoothed.to_csv("submissions/SMOOTHED_64TFT_36LSTM.csv", index=False)
    print("Saved smoothed predictions.")

    
if __name__ == "__main__":
    main()