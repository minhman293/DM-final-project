# DM-final-project

This repository contains code and artifacts for the time-series forecasting project using Temporal Fusion Transformer (TFT), LSTM variants, and LightGBM models. The project trains, evaluates, and blends multiple models and produces Kaggle-ready submission CSVs.

**Quick links**
- Training scripts: `train_tft.py`, `train_lstm.py`, `train.py`
- Predict runner: `predict_tft.py`, `predict_lstm.py`, `lgbm_deep_memory_v2.py`
- Ensemble helpers: `new_ensemble.py`, `ensemble.py`, `ensemble_final.py`
- Configs: `config_tft.py`, `config_lstm.py`
- Artifacts: `artifacts_tft/`, `models/`, `lightning_logs/`, `submissions/`

**Repository structure (important files/folders)**
- `data/` — input CSVs: `train.csv`, `test.csv`
- `artifacts_tft/` — saved TFT checkpoints (tft_best*.ckpt) and cached dataset spec
- `lightning_logs/` — PyTorch Lightning logs (each `version_*` folder contains `metrics.csv` and `hparams.yaml`)
- `models/`, `models_lstm/`, `models_pipeline_b/` — stored model artifacts and metadata
- `submissions/` — generated submission CSVs
- `requirements.txt` — Python dependencies

Prerequisites
- OS: tested on Windows and Linux
- Python 3.8+ (Conda recommended)
- GPU recommended for TFT training (NVIDIA + CUDA)

Recommended Conda environment (example)
```bash
conda create -n gpu_env python=3.10 -y
conda activate gpu_env
pip install -r requirements.txt
```
If you use an existing environment, ensure PyTorch, PyTorch Lightning and PyTorch Forecasting versions in `requirements.txt` are satisfied.

Configuration
- Main TFT config is in `config_tft.py`. Important values:
	- `ARTIFACT_DIR` — where checkpoints and `training_dataset.pkl` are saved (default `artifacts_tft`)
	- `CHECKPOINT_PATH` — default path for the best checkpoint
	- Batch sizes, seed, accelerator, devices, and precision options are also in the config file
- To change hyperparameters, edit `config_tft.py` or other relevant config files before running.

Training (TFT)
1. Prepare data (build combined frame and cached frames):
```bash
python train_tft.py

```
2. The `train_tft.py` scripts construct datasets with `utils_tft` and `data_prep.prepare()` and then call PyTorch Lightning `Trainer.fit()`.
3. Checkpoints: ModelCheckpoint callback saves the best checkpoint to `artifacts_tft/tft_best*.ckpt` or `artifacts_tft/tft_best_seed{seed}.ckpt`.
4. Logs: PyTorch Lightning writes human-readable logs to `lightning_logs/version_*/metrics.csv` and `hparams.yaml`.
	 - Example: `lightning_logs/version_9/metrics.csv` contains per-step/epoch metrics like `train_loss`, `val_loss`, `val_MAE`, `val_RMSE`, `val_SMAPE`.
	 - To find the latest training metrics CSV:
```bash
ls -1t lightning_logs | head
# then open the newest version folder, e.g. lightning_logs/version_13/metrics.csv
```

Training (LSTM, LightGBM)
- LSTM training: `train_lstm.py` (weights in `models_lstm/`)
- LightGBM / tree-based training artifacts: produced by `lgbm_deep_memory_v2.py` and saved under `submissions/` as CSV predictions.

Producing predictions and submissions
- TFT: `predict_tft.py` loads the latest `tft_best*.ckpt` from `artifacts_tft/`, loads `training_dataset.pkl`, builds a predict dataset and writes a submission CSV to `submissions/`.
- LSTM: `predict_lstm.py` writes `submissions/submission_lstm_v2.csv` or similar.
- Ensemble/Blend: `reproduce_best.py` and `run_all.py` show the expected blending strategy and how to combine model outputs into final ensemble submission files.

Approaches and results

The table below summarizes the main model families and blends explored during the project, together with the public leaderboard score where available.

| # | Approach | Public LB |
|---|---|---:|
| 1 | Original LightGBM (no temporal features) | 1.0911 |
| 2 | Pure TFT + anomaly features | 1.1962 |
| 3 | Pure TFT with $q = 0.25$ quantile | 0.9393 |
| 4 | Pure TFT, full training data | 0.9296 |
| 5 | Pure LSTM | 0.9029 |
| 6 | Pure TFT (truncated 391w, ReLU) | 0.8891 |
| 7 | TFT + region_mean (70/30) | 0.8666 |
| 8 | LightGBM + anomaly features | 0.8402 |
| 9 | TFT + LSTM + region_mean (triple) | 0.8378 |
| 10 | LightGBM + anomaly + memory features | 0.8295 |
| 11 | TFT + LSTM (70/30) | 0.8273 |
| 12 | TFT + LSTM + LGBM (50/25/25) | 0.8220 |
| 13 | TFT + LSTM (64/36, smoothed) | 0.8189 |
| 14 | MEGA_BLEND (DL 70 + LGBM 30) | 0.8142 |
| 15 | HORIZON_MEGA_BLEND (gradient 0.60 to 0.80) | 0.8133 |
| 16 | **HORIZON_AGGRESSIVE (0.50 to 0.85)** | **0.8129** |

Notes:
- The best overall public result in this list is **HORIZON_AGGRESSIVE (0.50 to 0.85)** at **0.8129**.
- The earlier canonical blend used by `reproduce_best.py` is **TFT + LSTM (70/30)** at **0.8273**.
- The approach names match the experimental lineage used in `new_ensemble.py` and the saved submission files in `submissions/`.

Reproducing the official pipeline (full end-to-end)
1. Ensure `data/train.csv` and `data/test.csv` are present.
2. Steps:
- Update SEED=42 in `config_tft.py`
- Train three TFT seeds (example seeds used: 42, 123, 999):
```bash
python train_tft.py
python predict_tft.py
```
- Train LSTM and predict:
```bash
python train_lstm.py
python predict_lstm.py
```
- Train LGBM and predict:
```bash
python lgbm_deep_memory_v2.py
```
- Build the ensemble:
```bash
python ensemble_final.py
python new_ensemble.py
```

Where to find logs and readable metrics
- Check `lightning_logs/version_*/metrics.csv` for CSV-formatted training metrics.
- Hyperparameters and dataset encodings are in `lightning_logs/version_*/hparams.yaml` (useful to recreate data encoders and categorical mappings).
- Checkpoints (binary) are in `artifacts_tft/` and `models/`.

Debugging and tips
- CUDA OOM: reduce `BATCH_SIZE` or set `precision=16` in `config_tft.py`.
- If metrics appear in a different Lightning `version_*` folder, open that folder and read `metrics.csv`.
- To inspect the most recent metrics CSV programmatically:
```python
import pandas as pd
from pathlib import Path
v = sorted(Path('lightning_logs').iterdir(), key=lambda p: p.stat().st_mtime, reverse=True)[0]
print(v)
print(pd.read_csv(v / 'metrics.csv').tail())
```