import pandas as pd
import numpy as np

def ensemble_submissions():
    # ver 1: simple 50/50 blend of the best LSTM and best TFT
    # # Load your two best distinct models
    # lstm_sub = pd.read_csv("submissions/submission_lstm_v2.csv")
    # tft_sub = pd.read_csv("submissions/submission_tft_v4_relu.csv")
    
    # # Create a copy for the ensemble
    # ensemble = lstm_sub.copy()
    
    # pred_cols = ["pred_week1", "pred_week2", "pred_week3", "pred_week4", "pred_week5"]
    
    # # 50/50 Blend
    # for col in pred_cols:
    #     ensemble[col] = (lstm_sub[col] * 0.5) + (tft_sub[col] * 0.5)
    #     ensemble[col] = ensemble[col].round(4)
        
    # ensemble.to_csv("submissions/ensemble_lstm_v2_tft_v4.csv", index=False)
    # print("Ensemble saved!")

    # ver 2: average the 3 ReLU TFT seeds first, then blend with LSTM
    # # 1. Load the best LSTM
    # lstm = pd.read_csv("submissions/submission_lstm_v2.csv")
    
    # # 2. Load the 3 ReLU TFT seeds
    # tft_42 = pd.read_csv("submissions/submission_tft_v4_relu.csv")
    # tft_123 = pd.read_csv("submissions/submission_tft_v4_relu_123.csv")
    # tft_999 = pd.read_csv("submissions/submission_tft_v4_relu_999.csv")
    
    # final_sub = lstm.copy()
    # pred_cols = ["pred_week1", "pred_week2", "pred_week3", "pred_week4", "pred_week5"]
    
    # for col in pred_cols:
    #     # Average the TFT seeds first to create a pristine signal
    #     super_tft = (tft_42[col] + tft_123[col] + tft_999[col]) / 3.0
        
    #     # Blend 50/50 with the LSTM to trigger the error cancellation
    #     final_sub[col] = (super_tft * 0.50) + (lstm[col] * 0.50)
    #     final_sub[col] = final_sub[col].round(4)
        
    # final_sub.to_csv("submissions/ensemble_lstm_v2_tft_relu_seed_avg.csv", index=False)
    # print("The submission is ready!")

    # ver 3: more aggressive 70% TFT / 30% LSTM blend, plus a time-decay variant that gives LSTM more weight on Week 1 and less on Week 5.
    # 1. Load the models
    # lstm = pd.read_csv("submissions/submission_lstm_v2.csv")
    # tft_42 = pd.read_csv("submissions/submission_tft_v4_relu.csv")
    # tft_123 = pd.read_csv("submissions/submission_tft_v4_relu_123.csv")
    # tft_999 = pd.read_csv("submissions/submission_tft_v4_relu_999.csv")
    
    # pred_cols = ["pred_week1", "pred_week2", "pred_week3", "pred_week4", "pred_week5"]
    
    # # 2. Build the Super TFT
    # super_tft = lstm.copy()
    # for col in pred_cols:
    #     super_tft[col] = (tft_42[col] + tft_123[col] + tft_999[col]) / 3.0
        
    # # =========================================================
    # # STRATEGY A: Static Asymmetric (70% TFT / 30% LSTM) THIS IS THE BEST ONE SO FAR
    # # =========================================================
    # sub_70_30 = lstm.copy()
    # for col in pred_cols:
    #     sub_70_30[col] = (super_tft[col] * 0.70) + (lstm[col] * 0.30)
    #     sub_70_30[col] = sub_70_30[col].round(4)
        
    # sub_70_30.to_csv("submissions/ensemble_70tft_30lstm.csv", index=False)
    
    # # =========================================================
    # # STRATEGY B: Time-Decay Blending (The 0.80 Breaker)
    # # =========================================================
    # # LSTM gets 50% on Week 1, but drops to 10% by Week 5.
    # decay_weights = {
    #     "pred_week1": {"tft": 0.50, "lstm": 0.50},
    #     "pred_week2": {"tft": 0.60, "lstm": 0.40},
    #     "pred_week3": {"tft": 0.70, "lstm": 0.30},
    #     "pred_week4": {"tft": 0.80, "lstm": 0.20},
    #     "pred_week5": {"tft": 0.90, "lstm": 0.10},
    # }
    
    # sub_decay = lstm.copy()
    # for col in pred_cols:
    #     w_tft = decay_weights[col]["tft"]
    #     w_lstm = decay_weights[col]["lstm"]
        
    #     sub_decay[col] = (super_tft[col] * w_tft) + (lstm[col] * w_lstm)
    #     sub_decay[col] = sub_decay[col].round(4)
        
    # sub_decay.to_csv("submissions/ensemble_time_decay.csv", index=False)
    # print("Smart ensembles generated successfully!")

    
    # ver 5
    # print("1. Loading the Champion Models...")
    # # Load your best LSTM
    # lstm = pd.read_csv("submissions/submission_lstm_v2.csv")
    
    # # Load the 3 Median TFT models
    # tft_42 = pd.read_csv("submissions/submission_tft_v4_relu.csv")
    # tft_123 = pd.read_csv("submissions/submission_tft_v4_relu_123.csv")
    # tft_999 = pd.read_csv("submissions/submission_tft_v4_relu_999.csv")

    # pred_cols = ["pred_week1", "pred_week2", "pred_week3", "pred_week4", "pred_week5"]

    # print("2. Building the Super TFT...")
    # super_tft = lstm.copy()
    # for col in pred_cols:
    #     super_tft[col] = (tft_42[col] + tft_123[col] + tft_999[col]) / 3.0

    # print("3. Applying the LSTM-Gated Squeezer...")
    # final_sub = lstm.copy()
    
    # for col in pred_cols:
    #     # Calculate the standard 70/30 blend
    #     standard_blend = (super_tft[col] * 0.70) + (lstm[col] * 0.30)
        
    #     # Apply the Gate: If LSTM is highly confident it's a zero (<0.10), force to 0.0
    #     # Otherwise, keep the 70/30 blend
    #     final_sub[col] = np.where(lstm[col] < 0.10, 0.0, standard_blend)
        
    #     # Round to 4 decimals for Kaggle
    #     final_sub[col] = final_sub[col].round(4)

    # out_path = "submissions/ensemble_70tft_30lstm_GATED.csv"
    # final_sub.to_csv(out_path, index=False)
    # print(f"Success! Saved to {out_path}")

    # ver 6
    # print("1. Loading the Champion Models...")
    # # Load your best LSTM
    # lstm = pd.read_csv("submissions/submission_lstm_v2.csv")
    
    # # Load the 3 Median TFT models
    # tft_42 = pd.read_csv("submissions/submission_tft_v4_relu.csv")
    # tft_123 = pd.read_csv("submissions/submission_tft_v4_relu_123.csv")
    # tft_999 = pd.read_csv("submissions/submission_tft_v4_relu_999.csv")

    # print("2. Calculating true historical baseline per region...")
    # train = pd.read_csv("data/train.csv")
    # region_means = train.dropna(subset=["score"]).groupby("region_id")["score"].mean().to_dict()
    # global_mean = train["score"].mean()

    # print("3. Building the Super TFT...")
    # pred_cols = ["pred_week1", "pred_week2", "pred_week3", "pred_week4", "pred_week5"]
    # super_tft = lstm.copy()
    # for col in pred_cols:
    #     super_tft[col] = (tft_42[col] + tft_123[col] + tft_999[col]) / 3.0

    # print("4. Applying the 55/25/20 Triple Blend...")
    # final_sub = lstm.copy()
    
    # for idx, row in final_sub.iterrows():
    #     reg = row["region_id"]
    #     # Handle string or int region_ids safely
    #     try:
    #         reg_key = int(reg)
    #     except ValueError:
    #         reg_key = reg
            
    #     baseline = region_means.get(reg_key, global_mean)

    #     for col in pred_cols:
    #         tft_val = super_tft.at[idx, col]
    #         lstm_val = lstm.at[idx, col]
            
    #         # The Formula: 55% TFT + 25% LSTM + 20% Region Mean
    #         blended_val = (tft_val * 0.55) + (lstm_val * 0.25) + (baseline * 0.20)
    #         final_sub.at[idx, col] = blended_val

    # for col in pred_cols:
    #     final_sub[col] = final_sub[col].round(4)

    # out_path = "submissions/ensemble_TRIPLE_BLEND.csv"
    # final_sub.to_csv(out_path, index=False)
    # print(f"Success! Saved to {out_path}")

    # Load your two best distinct models
    lstm_sub = pd.read_csv("submissions/submission_lstm_upgrade.csv")
    tft_sub = pd.read_csv("submissions/submission_tft_upgrade.csv")
    
    # Create a copy for the ensemble
    ensemble = lstm_sub.copy()
    
    pred_cols = ["pred_week1", "pred_week2", "pred_week3", "pred_week4", "pred_week5"]
    
    # 30/70 Blend
    for col in pred_cols:
        ensemble[col] = (lstm_sub[col] * 0.30) + (tft_sub[col] * 0.70)
        ensemble[col] = ensemble[col].round(4)
        
    ensemble.to_csv("submissions/ensemble_70tft_30lstm_upgrade.csv", index=False)
    print("Ensemble saved!")


if __name__ == "__main__":
    ensemble_submissions()