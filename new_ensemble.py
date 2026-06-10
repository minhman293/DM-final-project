# import pandas as pd
# import numpy as np

# # Load your best blend and the new LightGBM
# blend = pd.read_csv("submissions/ENSEMBLE_64TFT_36LSTM.csv")
# lgbm = pd.read_csv("submissions/LGBM_DEEP_MEMORY_V2.csv")

# # Ensure same row order
# blend['region_id'] = blend['region_id'].astype(str)
# lgbm['region_id'] = lgbm['region_id'].astype(str)
# merged = blend.merge(lgbm, on='region_id', suffixes=('_blend', '_lgbm'))

# pred_cols = ["pred_week1", "pred_week2", "pred_week3", "pred_week4", "pred_week5"]

# # 70% blend (proven strong) + 30% LightGBM (new, different architecture)
# out = pd.DataFrame({'region_id': merged['region_id']})
# for col in pred_cols:
#     out[col] = (0.70 * merged[f'{col}_blend'] + 0.30 * merged[f'{col}_lgbm']).round(4)

# out.to_csv("submissions/MEGA_BLEND_70TFTLSTM_30LGBM.csv", index=False)
# print("Saved.")

# import pandas as pd

# # Your TFT-only submission (need to confirm filename — likely submission_tft_v4_relu.csv at 0.8891)
# tft = pd.read_csv("submissions/submission_tft_v4_relu_bk.csv")
# # Your LSTM-only submission
# lstm = pd.read_csv("submissions/submission_lstm_v2.csv")
# # Your LightGBM
# lgbm = pd.read_csv("submissions/LGBM_DEEP_MEMORY_V2.csv")

# for d in [tft, lstm, lgbm]:
#     d['region_id'] = d['region_id'].astype(str)

# pred_cols = ["pred_week1", "pred_week2", "pred_week3", "pred_week4", "pred_week5"]

# # 50% TFT + 25% LSTM + 25% LightGBM
# # Three architecturally distinct models in equal-ish weights
# merged = tft.merge(lstm, on='region_id', suffixes=('_tft', '_lstm'))
# merged = merged.merge(lgbm, on='region_id')
# # After second merge, lgbm columns will be named normally (without suffix)

# out = pd.DataFrame({'region_id': merged['region_id']})
# for col in pred_cols:
#     out[col] = (
#         0.50 * merged[f'{col}_tft'] +
#         0.25 * merged[f'{col}_lstm'] +
#         0.25 * merged[col]  # lgbm
#     ).round(4)

# out.to_csv("submissions/TRIPLE_50TFT_25LSTM_25LGBM.csv", index=False)
# print("Saved.")

# import pandas as pd
# import numpy as np

# def horizon_blend_and_smooth():
#     print("Loading Finalists for Horizon Blending...")
#     # Make sure you are loading the SMOOTHED TFT, not the raw one
#     blend = pd.read_csv("submissions/SMOOTHED_64TFT_36LSTM.csv")
#     lgbm = pd.read_csv("submissions/LGBM_DEEP_MEMORY_V2.csv")

#     blend['region_id'] = blend['region_id'].astype(str)
#     lgbm['region_id'] = lgbm['region_id'].astype(str)
#     merged = blend.merge(lgbm, on='region_id', suffixes=('_dl', '_tree'))

#     out = pd.DataFrame({'region_id': merged['region_id']})
    
#     # 1. Horizon-Specific Blending
#     # Week 1 favors the Tree more (short-term lag)
#     # Week 5 favors the DL more (long-term sequence)
#     weights_dl = [0.60, 0.65, 0.70, 0.75, 0.80]
    
#     print("Applying Dynamic Weights...")
#     for i in range(5):
#         w = i + 1
#         col = f"pred_week{w}"
#         dl_w = weights_dl[i]
#         tree_w = 1.0 - dl_w
#         out[col] = (merged[f"{col}_dl"] * dl_w) + (merged[f"{col}_tree"] * tree_w)

#     print("Re-applying Temporal Smoother to the Mega Blend...")
#     # 2. Temporal Smoothing
#     w1, w2, w3, w4, w5 = out["pred_week1"], out["pred_week2"], out["pred_week3"], out["pred_week4"], out["pred_week5"]
    
#     out["pred_week1"] = (w1 * 0.8) + (w2 * 0.2)
#     out["pred_week2"] = (w2 * 0.8) + (w1 * 0.1) + (w3 * 0.1)
#     out["pred_week3"] = (w3 * 0.8) + (w2 * 0.1) + (w4 * 0.1)
#     out["pred_week4"] = (w4 * 0.8) + (w3 * 0.1) + (w5 * 0.1)
#     out["pred_week5"] = (w5 * 0.8) + (w4 * 0.2)

#     # 3. Final constraints
#     for i in range(1, 6):
#         col = f"pred_week{i}"
#         out[col] = np.where(out[col] < 0.05, 0.0, out[col])
#         out[col] = np.clip(out[col], 0.0, 5.0).round(4)

#     out_path = "submissions/HORIZON_MEGA_BLEND.csv"
#     out.to_csv(out_path, index=False)
#     print(f"Done! Final dart saved to: {out_path}")

# if __name__ == "__main__":
#     horizon_blend_and_smooth()

import pandas as pd
import numpy as np

blend = pd.read_csv("submissions/SMOOTHED_64TFT_36LSTM.csv")
lgbm = pd.read_csv("submissions/LGBM_DEEP_MEMORY_V2.csv")

blend['region_id'] = blend['region_id'].astype(str)
lgbm['region_id'] = lgbm['region_id'].astype(str)
merged = blend.merge(lgbm, on='region_id', suffixes=('_dl', '_tree'))

# More aggressive horizon gradient — push LGBM harder where it has signal (week 1)
# Less aggressive in middle (week 3) where signals are most balanced
weights_dl = [0.50, 0.60, 0.70, 0.78, 0.85]

out = pd.DataFrame({'region_id': merged['region_id']})
for i in range(5):
    w = i + 1
    col = f"pred_week{w}"
    out[col] = (merged[f"{col}_dl"] * weights_dl[i]) + (merged[f"{col}_tree"] * (1.0 - weights_dl[i]))

# Same smoothing
w1, w2, w3, w4, w5 = [out[f"pred_week{i}"].copy() for i in range(1, 6)]
out["pred_week1"] = w1 * 0.8 + w2 * 0.2
out["pred_week2"] = w2 * 0.8 + w1 * 0.1 + w3 * 0.1
out["pred_week3"] = w3 * 0.8 + w2 * 0.1 + w4 * 0.1
out["pred_week4"] = w4 * 0.8 + w3 * 0.1 + w5 * 0.1
out["pred_week5"] = w5 * 0.8 + w4 * 0.2

for i in range(1, 6):
    col = f"pred_week{i}"
    out[col] = np.where(out[col] < 0.05, 0.0, out[col])
    out[col] = np.clip(out[col], 0.0, 5.0).round(4)

out.to_csv("submissions/HORIZON_AGGRESSIVE.csv", index=False)
print("Saved.")