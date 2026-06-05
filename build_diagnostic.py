import pandas as pd

def build_diagnostic_file():
    print("Loading local predictions and truth...")
    truth = pd.read_csv("data_local/local_truth.csv")
    tft = pd.read_csv("submissions/tft_local_preds.csv")
    lstm = pd.read_csv("submissions/lstm_local_preds.csv")

    print("Formatting for diagnostics...")
    # Melt from wide format (5 columns) to long format (1 column)
    truth_long = truth.melt(id_vars="region_id", value_name="true_score", var_name="week")
    tft_long = tft.melt(id_vars="region_id", value_name="tft_pred", var_name="week")
    lstm_long = lstm.melt(id_vars="region_id", value_name="lstm_pred", var_name="week")

    # Combine everything side-by-side
    merged = truth_long.merge(tft_long, on=["region_id", "week"]).merge(lstm_long, on=["region_id", "week"])
    
    merged.to_csv("validation_predictions.csv", index=False)
    print("Success! 'validation_predictions.csv' is ready for the diagnostic script.")

if __name__ == "__main__":
    build_diagnostic_file()