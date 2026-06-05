# Drought Severity Prediction — Group 17

Final project for Data Mining, Spring 2026.
Public LB: **0.8273** — 70/30 blend of a Temporal Fusion Transformer and an LSTM.

> Replace `{ID}` above with your group number before submitting.

## Environment

```
pip install -r requirements.txt
```

Tested on Python 3.10 with a single CUDA GPU. The pipeline runs on CPU as well but training takes substantially longer.

## Data layout

Place the Kaggle files exactly here before running anything:

```
data/train.csv
data/test.csv
sample_submission.csv
```

## Reproduction — quick path (no training, ~1 minute)

The four prediction CSVs that produced the 0.8273 blend are committed under `submissions/`:

- `submission_tft_v4_relu.csv` — TFT seed 42
- `submission_tft_v4_relu_123.csv` — TFT seed 123
- `submission_tft_v4_relu_999.csv` — TFT seed 999
- `submission_lstm_v2.csv` — LSTM v2

To rebuild the ensemble file:

```
python reproduce_best.py
```

Output: `submissions/ensemble_70tft_30lstm.csv`. This is the file that scored 0.8273 on the public leaderboard.

## Reproduction — full path (from scratch, ~3–4 hours on GPU)

Before running:

1. Confirm `data/train.csv` and `data/test.csv` are in place.
2. Open `config_lstm.py` and ensure `DATA_DIR = Path("data")` (not `data_local`).

Then:

```
python run_all.py
```

This script will:

1. Train 3 TFT seeds (42, 123, 999), writing one prediction CSV per seed.
2. Train LSTM v2, copying its prediction into `submission_lstm_v2.csv`.
3. Build the final blend: `0.70 × mean(3 TFT seeds) + 0.30 × LSTM v2`.

Each step skips itself if its output CSV already exists, so the script is safe to rerun after partial failures.

## Locked configuration

| Component | Setting | Location |
|---|---|---|
| TFT target normalizer | `GroupNormalizer(transformation="relu")` | `train_tft.py` |
| TFT encoder length | 26 weeks | `config_tft.py` |
| TFT decoder length | 5 weeks | `config_tft.py` |
| TFT training truncation | last 391 weeks per region | `data_prep.py` |
| TFT seeds in ensemble | 42, 123, 999 | `run_all.py` |
| LSTM data directory | `data` | `config_lstm.py` |
| LSTM threshold squeezer | 0.4 | `config_lstm.py` |
| Blend formula | `0.70 × mean(3 TFT) + 0.30 × LSTM`, rounded to 4 dp | `reproduce_best.py` |

## File index

```
.
├── data/                  Kaggle CSVs (train.csv, test.csv)
├── submissions/           output CSVs (4 source files + final blend committed)
├── artifacts_tft/         TFT cached frame + checkpoints (created at runtime)
├── models_lstm/           LSTM encoders, scaler, weights (created at runtime)
│
├── config_tft.py          TFT hyperparameters
├── data_prep.py           daily → weekly + combined frame builder for TFT
├── train_tft.py           single-seed TFT training (called via run_seed.py)
├── predict_tft.py         single-seed TFT inference — see Known issues below
├── utils_tft.py           logging helpers
│
├── config_lstm.py         LSTM hyperparameters
├── feature_engineering.py weekly aggregation + lag/rolling features for LSTM
├── dataset.py             PyTorch Dataset for LSTM
├── model.py               LSTM architecture
├── train_lstm.py          LSTM training
├── predict_lstm.py        LSTM inference
├── utils.py               shared utilities
│
├── ensemble.py            original blending experiments (kept for transparency)
├── reproduce_best.py      rebuild the best submission from the 4 saved CSVs
├── run_seed.py            train one TFT seed end-to-end (seed-aware naming)
├── run_all.py             full from-scratch reproduction
├── README.md              this file
└── requirements.txt       Python dependencies
```

## Known issues

`predict_tft.py` contains a hardcoded checkpoint path (`tft_best-v5.ckpt`) from our local development environment. **Use `run_seed.py` instead** — it selects the right seed-specific checkpoint automatically.

If you want to run `predict_tft.py` directly, change line ~57 from:

```python
ckpt = ARTIFACT_DIR / "tft_best-v5.ckpt"
```

to:

```python
ckpt = max(ckpt_candidates, key=lambda p: p.stat().st_mtime)
```

## Reference

Temporal Fusion Transformer architecture: Lim, Bryan, et al. "Temporal fusion transformers for interpretable multi-horizon time series forecasting." *International Journal of Forecasting* 37.4 (2021): 1748–1764.