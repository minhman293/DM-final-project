import pandas as pd
import numpy as np

def region_mean_blend():
    print("1. Calculating true historical baseline per region...")
    # Load train to get the true historical baseline
    train = pd.read_csv("data/train.csv")
    region_means = train.dropna(subset=["score"]).groupby("region_id")["score"].mean().to_dict()
    global_mean = train["score"].mean() # Fallback

    print("2. Loading the 3 Median TFT models...")
    # Ensure these are the files generated with MEDIAN_QUANTILE_IDX = 3
    tft_42 = pd.read_csv("submissions/submission_tft_v4_relu.csv")
    tft_123 = pd.read_csv("submissions/submission_tft_v4_relu_123.csv")
    tft_999 = pd.read_csv("submissions/submission_tft_v4_relu_999.csv")

    print("3. Building the Super TFT...")
    super_tft = tft_42.copy()
    pred_cols = ["pred_week1", "pred_week2", "pred_week3", "pred_week4", "pred_week5"]
    for col in pred_cols:
        super_tft[col] = (tft_42[col] + tft_123[col] + tft_999[col]) / 3.0

    print("4. Applying the 70/30 Region-Mean anchor...")
    final_sub = super_tft.copy()
    
    # We iterate to map the correct region baseline to each row
    for idx, row in final_sub.iterrows():
        reg = row["region_id"]
        # Handle string or int region_ids safely
        try:
            reg_key = int(reg)
        except ValueError:
            reg_key = reg
            
        baseline = region_means.get(reg_key, global_mean)

        for col in pred_cols:
            # 70% Super TFT / 30% Historical Baseline
            final_sub.at[idx, col] = (super_tft.at[idx, col] * 0.70) + (baseline * 0.30)

    for col in pred_cols:
        final_sub[col] = final_sub[col].round(4)

    out_path = "submissions/ensemble_70tft_30region_mean.csv"
    final_sub.to_csv(out_path, index=False)
    print(f"Success! Saved to {out_path}")

if __name__ == "__main__":
    region_mean_blend()