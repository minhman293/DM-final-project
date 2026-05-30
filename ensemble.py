import pandas as pd

def ensemble_submissions():
    # Load your two best distinct models
    lstm_sub = pd.read_csv("submissions/submission_lstm_v2.csv")
    tft_sub = pd.read_csv("submissions/submission_tft_v4_relu.csv")
    
    # Create a copy for the ensemble
    ensemble = lstm_sub.copy()
    
    pred_cols = ["pred_week1", "pred_week2", "pred_week3", "pred_week4", "pred_week5"]
    
    # 50/50 Blend
    for col in pred_cols:
        ensemble[col] = (lstm_sub[col] * 0.5) + (tft_sub[col] * 0.5)
        ensemble[col] = ensemble[col].round(4)
        
    ensemble.to_csv("submissions/ensemble_lstm_v2_tft_v4.csv", index=False)
    print("Ensemble saved!")

if __name__ == "__main__":
    ensemble_submissions()