"""
Generates all figures for the report.
Run from the project root: python report_evidence/make_figures.py
Outputs: report_evidence/fig_*.png
"""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from pathlib import Path

OUT = Path("report_evidence")
OUT.mkdir(exist_ok=True)

# ── Figure 1: Ablation bar chart ────────────────────────────────────────────
approaches = [
    ("Original LightGBM (no temporal features)", 1.0911, "tab:red"),
    ("Pure TFT + anomaly features", 1.1962, "tab:red"),
    ("Pure TFT with q=0.25 quantile", 0.9393, "tab:orange"),
    ("Pure TFT (full training data)", 0.9296, "tab:orange"),
    ("Pure LSTM", 0.9029, "tab:orange"),
    ("Pure TFT (truncated 391w, ReLU)", 0.8891, "tab:orange"),
    ("TFT + region_mean (70/30)", 0.8666, "tab:gray"),
    ("LightGBM + anomaly features", 0.8402, "tab:gray"),
    ("TFT + LSTM + region_mean (triple)", 0.8378, "tab:gray"),
    ("LightGBM + anomaly + memory features", 0.8295, "tab:gray"),
    ("TFT + LSTM (70/30)", 0.8273, "tab:green"),
    ("TFT + LSTM + LGBM (50/25/25)", 0.8220, "tab:green"),
    ("TFT + LSTM (64/36, smoothed)", 0.8189, "tab:green"),
    ("MEGA_BLEND (DL 70 + LGBM 30)", 0.8142, "tab:blue"),
    ("HORIZON_MEGA_BLEND (gradient 0.60->0.80)", 0.8133, "tab:blue"),
    ("HORIZON_AGGRESSIVE (0.50->0.85)", 0.8129, "tab:blue"),
]
approaches.sort(key=lambda x: -x[1])  # worst first, best at top
labels = [a[0] for a in approaches]
scores = [a[1] for a in approaches]
colors = [a[2] for a in approaches]

fig, ax = plt.subplots(figsize=(8, 7))
bars = ax.barh(labels, scores, color=colors, edgecolor='black', linewidth=0.5)
ax.axvline(0.9117, color='red', linestyle='--', linewidth=1, label='Baseline 1 (0.9117)')
ax.axvline(0.8623, color='orange', linestyle='--', linewidth=1, label='Baseline 2 (0.8623)')
ax.axvline(0.8056, color='green', linestyle='--', linewidth=1, label='Baseline 3 (0.8056)')
ax.set_xlabel("Public LB MAE (lower is better)")
ax.set_title("Ablation Study: All Tested Approaches")
ax.legend(loc='lower right', fontsize=8)
ax.set_xlim(0.74, 1.22)
for bar, s in zip(bars, scores):
    ax.text(s + 0.003, bar.get_y() + bar.get_height()/2,
            f"{s:.4f}", va='center', fontsize=7)
plt.tight_layout()
plt.savefig(OUT / "fig_ablation.png", dpi=200)
plt.close()
print("Saved fig_ablation.png")

# ── Figure 2: Per-horizon blend weights ─────────────────────────────────────
horizons = [1, 2, 3, 4, 5]
dl_weights = [0.50, 0.60, 0.70, 0.78, 0.85]
lgbm_weights = [1 - w for w in dl_weights]

fig, ax = plt.subplots(figsize=(6, 3.5))
ax.plot(horizons, dl_weights, 'o-', label='Deep Learning (TFT+LSTM)', color='tab:blue', linewidth=2)
ax.plot(horizons, lgbm_weights, 's-', label='LightGBM', color='tab:orange', linewidth=2)
ax.set_xticks(horizons)
ax.set_xlabel("Forecast Horizon (weeks ahead)")
ax.set_ylabel("Blend Weight")
ax.set_title("Horizon-Dependent Ensemble Weights")
ax.legend()
ax.grid(alpha=0.3)
ax.set_ylim(0.1, 0.9)
plt.tight_layout()
plt.savefig(OUT / "fig_blend_weights.png", dpi=200)
plt.close()
print("Saved fig_blend_weights.png")

# ── Figure 3: Score distribution (showing zero-spike) ───────────────────────
# This requires train.csv — replace path if needed
try:
    train = pd.read_csv("data/train.csv")
    scores = train['score'].dropna().values
    fig, ax = plt.subplots(figsize=(6, 3.5))
    ax.hist(scores, bins=np.arange(-0.25, 5.75, 0.5),
            color='steelblue', edgecolor='black')
    ax.set_xlabel("Drought severity score")
    ax.set_ylabel("Number of scored weeks (log scale)")
    ax.set_yscale('log')
    ax.set_title("Target Distribution: Heavy Zero-Inflation")
    pct_zero = (scores == 0).mean() * 100
    ax.text(2.5, ax.get_ylim()[1] * 0.5,
            f"{pct_zero:.1f}% of scored weeks are exactly 0",
            fontsize=10, bbox=dict(boxstyle='round', facecolor='lightyellow'))
    plt.tight_layout()
    plt.savefig(OUT / "fig_score_distribution.png", dpi=200)
    plt.close()
    print(f"Saved fig_score_distribution.png — {pct_zero:.1f}% zeros")
except FileNotFoundError:
    print("WARNING: data/train.csv not found, skipping fig_score_distribution.png")
    print("Run this script from the project root with data/ folder present.")

# ── Figure 4: Architecture diagram (text-based, you'll re-create in LaTeX) ──
# I'll write this as a TikZ block in the report itself; no PNG needed.

print("\nAll figures generated in report_evidence/")