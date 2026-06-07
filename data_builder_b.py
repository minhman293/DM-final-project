# data_builder_b.py
import pandas as pd
import numpy as np
import joblib
from pathlib import Path
from config_pipeline_b import ARTIFACT_DIR, MODEL_DIR

def build_3d_numpy_dataset():
    print("1. Loading TFT's pre-processed Combined Frame...")
    
    # Try parquet first, fallback to pickle
    parquet_path = ARTIFACT_DIR / "combined_frame.parquet"
    pkl_path = ARTIFACT_DIR / "combined_frame.pkl"
    
    if parquet_path.exists():
        combined = pd.read_parquet(parquet_path)
    else:
        combined = pd.read_pickle(pkl_path)
        
    features = [
        "surf_pre_mean", "tmp_range_mean", "wind_mean", "wind_max_max",
        "prec_sum", "humidity_mean", "delta_prec", "delta_tmp_range",
        "month_sin", "month_cos"
    ]
    
    print("2. Splitting into Train and Test...")
    # is_test == 0 (Train), is_test == 1 (Test), is_test == 2 (Future/NaNs)
    train_w = combined[combined["is_test"] == 0].sort_values(['region_id', 'time_idx'])
    test_w = combined[combined["is_test"] == 1].sort_values(['region_id', 'time_idx'])
    
    print("3. Building Region Index...")
    regions = sorted(train_w['region_id'].unique())
    region_to_idx = {r: i for i, r in enumerate(regions)}
    joblib.dump(region_to_idx, MODEL_DIR / "region_to_idx.pkl")
    
    # 4. Create 3D Numpy Arrays [Regions, Weeks, Features]
    num_regions = len(regions)
    num_train_weeks = train_w.groupby('region_id').size().max() # Should be 391
    num_test_weeks = test_w.groupby('region_id').size().max()   # Should be 13
    num_features = len(features)
    
    train_array = np.zeros((num_regions, num_train_weeks, num_features), dtype=np.float32)
    target_array = np.zeros((num_regions, num_train_weeks), dtype=np.float32)
    test_array = np.zeros((num_regions, num_test_weeks, num_features), dtype=np.float32)
    
    print("4. Filling Tensors (This takes a few seconds)...")
    for r, idx in region_to_idx.items():
        r_train = train_w[train_w['region_id'] == r]
        r_test = test_w[test_w['region_id'] == r]
        
        train_array[idx, :len(r_train), :] = r_train[features].values
        target_array[idx, :len(r_train)] = r_train['score'].values
        test_array[idx, :len(r_test), :] = r_test[features].values

    print("5. Normalizing Features...")
    mean = np.nanmean(train_array, axis=(0, 1))
    std = np.nanstd(train_array, axis=(0, 1)) + 1e-8
    train_array = (train_array - mean) / std
    test_array = (test_array - mean) / std
    
    print("6. Saving Numpy files...")
    np.save(MODEL_DIR / "train_features.npy", train_array)
    np.save(MODEL_DIR / "train_targets.npy", target_array)
    np.save(MODEL_DIR / "test_features.npy", test_array)
    
    print(f"Done! Train Array Shape: {train_array.shape} (Regions, Weeks, Features)")

if __name__ == "__main__":
    build_3d_numpy_dataset()