import pandas as pd
from pathlib import Path
import numpy as np

def create_local_validation():
    print("1. Loading original train.csv (This may take a moment)...")
    train_raw = pd.read_csv("data/train.csv")
    
    # Ensure chronological order
    train_raw = train_raw.sort_values(["region_id", "date"]).reset_index(drop=True)
    
    print("2. Slicing the timeline...")
    # 5 future weeks = 35 days, 13 test weeks = 91 days
    # We rank the days from newest (0) to oldest
    train_raw["day_rank"] = train_raw.groupby("region_id").cumcount(ascending=False)
    
    # The Target (Last 35 days)
    local_truth = train_raw[train_raw["day_rank"] < 35].copy()
    
    # The Test Context (The 91 days before the target)
    local_test = train_raw[(train_raw["day_rank"] >= 35) & (train_raw["day_rank"] < 126)].copy()
    
    # The Training Data (Everything else)
    local_train = train_raw[train_raw["day_rank"] >= 126].copy()
    
    # Clean up the helper column
    for df in [local_truth, local_test, local_train]:
        df.drop(columns=["day_rank"], inplace=True)
        
    # Mimic the real Kaggle test set by blanking out the score
    local_test["score"] = np.nan
    
    print("3. Saving simulated Kaggle environment...")
    out_dir = Path("data_local")
    out_dir.mkdir(exist_ok=True)
    
    local_train.to_csv(out_dir / "train.csv", index=False)
    local_test.to_csv(out_dir / "test.csv", index=False)
    
    # Format the Truth file to match Kaggle submission style for easy grading
    local_truth["week_id"] = local_truth.groupby("region_id").cumcount() // 7
    truth_weekly = local_truth.groupby(["region_id", "week_id"])["score"].first().reset_index()
    truth_pivot = truth_weekly.pivot(index="region_id", columns="week_id", values="score").reset_index()
    truth_pivot.columns = ["region_id", "pred_week1", "pred_week2", "pred_week3", "pred_week4", "pred_week5"]
    truth_pivot.to_csv(out_dir / "local_truth.csv", index=False)
    
    print("Done! Simulated dataset saved in 'data_local/' folder.")

if __name__ == "__main__":
    create_local_validation()