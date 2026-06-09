import numpy as np
from pathlib import Path

def main():
    # Load your training targets
    target_path = Path("models_pipeline_b/train_targets.npy")
    predict_path = Path("submissions/ENSEMBLE_64TFT_36LSTM.csv")
    
    # if not target_path.exists():
    #     print(f"Error: Could not find {target_path}")
    #     return
        
    # train_targets = np.load(target_path)
    
    # # Calculate the stats
    # total_samples = train_targets.size
    # zero_count = np.sum(train_targets == 0.0)
    # zero_percentage = (zero_count / total_samples) * 100
    
    # print("--- Training Data Target Distribution ---")
    # print(f"Total rows (samples):  {total_samples:,}")
    # print(f"Rows with exactly 0.0: {zero_count:,}")
    # print(f"Percentage of zeros:   {zero_percentage:.2f}%")

    if not predict_path.exists():
        print(f"Error: Could not find {predict_path}")
        return

    train_targets = np.load(predict_path)
    
    # Calculate the stats
    total_samples = train_targets.size
    zero_count = np.sum(train_targets == 0.0)
    zero_percentage = (zero_count / total_samples) * 100
    
    print("--- Prediction Data Target Distribution ---")
    print(f"Total rows (samples):  {total_samples:,}")
    print(f"Rows with exactly 0.0: {zero_count:,}")
    print(f"Percentage of zeros:   {zero_percentage:.2f}%")

if __name__ == "__main__":
    main()