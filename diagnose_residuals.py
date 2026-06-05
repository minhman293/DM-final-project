import pandas as pd
import numpy as np
from scipy.stats import pearsonr
from sklearn.metrics import mean_absolute_error

def diagnose_residuals(csv_path="validation_predictions.csv"):
    # Load your validation predictions
    df = pd.read_csv(csv_path)
    
    # Calculate Residuals (Error = Predicted - True)
    # Positive residual = Model over-predicted
    # Negative residual = Model under-predicted
    df['tft_error'] = df['tft_pred'] - df['true_score']
    df['lstm_error'] = df['lstm_pred'] - df['true_score']
    
    print("="*50)
    print("1. OVERALL STATS")
    print("="*50)
    print(f"TFT MAE:  {mean_absolute_error(df['true_score'], df['tft_pred']):.4f}")
    print(f"LSTM MAE: {mean_absolute_error(df['true_score'], df['lstm_pred']):.4f}")
    
    # Bias: is the model systematically predicting too high or too low?
    print(f"TFT Bias:  {df['tft_error'].mean():.4f}")
    print(f"LSTM Bias: {df['lstm_error'].mean():.4f}")
    
    print("\n"+"="*50)
    print("2. ENSEMBLE SYNERGY (RESIDUAL CORRELATION)")
    print("="*50)
    corr, _ = pearsonr(df['tft_error'], df['lstm_error'])
    print(f"Pearson Correlation between errors: {corr:.4f}")
    if corr > 0.80:
        print("-> WARNING: Models are making the same mistakes. Blending won't help much more.")
    elif corr < 0.60:
        print("-> EXCELLENT: Models are highly uncorrelated. Blending is highly effective.")
        
    print("\n"+"="*50)
    print("3. MAE LEAKAGE BY TRUE SCORE")
    print("="*50)
    # Group by the true score (rounded to nearest integer for grouping)
    df['true_rounded'] = df['true_score'].round()
    
    summary = []
    for score in sorted(df['true_rounded'].unique()):
        subset = df[df['true_rounded'] == score]
        count = len(subset)
        pct_of_data = (count / len(df)) * 100
        
        tft_mae = mean_absolute_error(subset['true_score'], subset['tft_pred'])
        tft_bias = subset['tft_error'].mean()
        
        lstm_mae = mean_absolute_error(subset['true_score'], subset['lstm_pred'])
        lstm_bias = subset['lstm_error'].mean()
        
        summary.append({
            "True": score,
            "% Data": f"{pct_of_data:.1f}%",
            "TFT MAE": round(tft_mae, 4),
            "TFT Bias": round(tft_bias, 4),
            "LSTM MAE": round(lstm_mae, 4),
            "LSTM Bias": round(lstm_bias, 4)
        })
        
    summary_df = pd.DataFrame(summary)
    print(summary_df.to_string(index=False))

if __name__ == "__main__":
    # Replace with your actual validation dataframe/csv
    # Make sure it has columns: true_score, tft_pred, lstm_pred
    diagnose_residuals("validation_predictions.csv")