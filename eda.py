"""
EDA Script — Natural Disaster Severity Prediction
Data Mining Spring 2026
Run: python eda.py
Outputs: eda_output/ directory with plots + eda_summary.txt
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from scipy import stats as scipy_stats
import warnings
warnings.filterwarnings("ignore")

# ── Config ────────────────────────────────────────────────────────────────────
TRAIN_PATH = "./data/train.csv"
TEST_PATH  = "./data/test.csv"
OUT_DIR    = Path("eda_output")
OUT_DIR.mkdir(exist_ok=True)

METEO_COLS = [
    "prec", "surf_pre", "humidity", "tmp", "dp_tmp", "wb_tmp",
    "tmp_max", "tmp_min", "tmp_range", "surf_tmp",
    "wind", "wind_max", "wind_min", "wind_range",
]

plt.rcParams.update({
    "figure.facecolor": "white",
    "axes.facecolor":   "white",
    "axes.spines.top":  False,
    "axes.spines.right": False,
    "font.size": 11,
})

summary_lines = []

def log(msg):
    print(msg)
    summary_lines.append(str(msg))

def conclusion(msg):
    """Print and log a conclusion block."""
    border = "-" * 60
    log(border)
    log(f"CONCLUSION: {msg}")
    log(border)

# ── Helper: extract date parts ────────────────────────────────────────────────
def extract_date_parts(df):
    """
    Parse date parts from string 'YYYY-MM-DD'.
    Years like 8133, 10004 are outside pandas datetime64 range (1678-2262),
    so we extract manually and compute Julian Day Number for day_of_week.
    """
    parts = df["date"].str.split("-", expand=True).astype(int)
    parts.columns = ["year", "month", "day"]
    df["year"]  = parts["year"]
    df["month"] = parts["month"]
    df["day"]   = parts["day"]
    y = parts["year"].copy()
    m = parts["month"].copy()
    d = parts["day"].copy()
    mask = m <= 2
    y[mask] -= 1
    m[mask] += 12
    A = (y / 100).astype(int)
    B = 2 - A + (A / 4).astype(int)
    jdn = (365.25 * (y + 4716)).astype(int) + (30.6001 * (m + 1)).astype(int) + d + B - 1524
    df["jdn"]         = jdn
    df["day_of_week"] = jdn % 7
    return df

# ══════════════════════════════════════════════════════════════════════════════
# 1. LOAD DATA
# ══════════════════════════════════════════════════════════════════════════════
log("=" * 60)
log("1. LOADING DATA")
log("=" * 60)

train = pd.read_csv(TRAIN_PATH)
test  = pd.read_csv(TEST_PATH)

log(f"Train shape : {train.shape}")
log(f"Test  shape : {test.shape}")
log(f"Train columns: {list(train.columns)}")
log(f"Test  columns: {list(test.columns)}")

conclusion(
    "Dataset loaded successfully. Train has 17 columns including the target 'score'; "
    "test has 16 columns (no score). Both share the same 14 meteorological features."
)

# ══════════════════════════════════════════════════════════════════════════════
# 2. DATE ANALYSIS
# ══════════════════════════════════════════════════════════════════════════════
log("\n" + "=" * 60)
log("2. DATE ANALYSIS")
log("=" * 60)

log(f"Train date range (raw): {train['date'].min()} -> {train['date'].max()}")
log(f"Test  date range (raw): {test['date'].min()}  -> {test['date'].max()}")
log("NOTE: Years are outside pandas datetime range — extracting parts from string.")

train = extract_date_parts(train)
test  = extract_date_parts(test)

log(f"Year  range (train): {train['year'].min()} -> {train['year'].max()}")
log(f"Month distribution (train):\n{train['month'].value_counts().sort_index().to_string()}")

conclusion(
    "Years like 3004-58061 are fictional and far outside pandas datetime64 range. "
    "We parse date parts manually using string splitting and compute day_of_week via "
    "Julian Day Number formula. The 'year' column cannot be used as a feature since "
    "it is arbitrary per region. Useful temporal features are: month (seasonality) "
    "and day_of_week (computed via JDN)."
)

# ══════════════════════════════════════════════════════════════════════════════
# 3. REGION ANALYSIS
# ══════════════════════════════════════════════════════════════════════════════
log("\n" + "=" * 60)
log("3. REGION ANALYSIS")
log("=" * 60)

region_counts_train = train.groupby("region_id").size()
region_counts_test  = test.groupby("region_id").size()

log(f"Train regions : {train['region_id'].nunique()}")
log(f"Test  regions : {test['region_id'].nunique()}")
log(f"\nTrain rows per region:\n{region_counts_train.describe().to_string()}")
log(f"\nTest  rows per region:\n{region_counts_test.describe().to_string()}")

n_full  = (region_counts_train == 5480).sum()
n_short = (region_counts_train < 5480).sum()
log(f"\nRegions with full 5480 days : {n_full}")
log(f"Regions with fewer days      : {n_short}")

fig, axes = plt.subplots(1, 2, figsize=(13, 4))
axes[0].hist(region_counts_train.values, bins=40, color="#378ADD", edgecolor="white", linewidth=0.5)
axes[0].set_title("Train: rows per region")
axes[0].set_xlabel("Row count")
axes[0].set_ylabel("Number of regions")
axes[0].axvline(5480, color="#E24B4A", linestyle="--", linewidth=1.5, label="Expected 5,480")
axes[0].legend(fontsize=10)

axes[1].hist(region_counts_test.values, bins=20, color="#1D9E75", edgecolor="white", linewidth=0.5)
axes[1].set_title("Test: rows per region")
axes[1].set_xlabel("Row count")
axes[1].set_ylabel("Number of regions")
axes[1].axvline(91, color="#E24B4A", linestyle="--", linewidth=1.5, label="Expected 91")
axes[1].legend(fontsize=10)

plt.tight_layout()
plt.savefig(OUT_DIR / "01_region_row_counts.png", dpi=150)
plt.close()
log("Saved: 01_region_row_counts.png")

conclusion(
    "Dataset is perfectly balanced: all 2,248 regions have exactly 5,480 days in train "
    "and exactly 91 days in test. No missing regions, no truncated timeseries. "
    "All 2,248 test regions also exist in train, meaning we can always look up "
    "historical data for any region in test."
)

# ══════════════════════════════════════════════════════════════════════════════
# 4. SCORE (LABEL) ANALYSIS
# ══════════════════════════════════════════════════════════════════════════════
log("\n" + "=" * 60)
log("4. SCORE (LABEL) ANALYSIS")
log("=" * 60)

scored = train.dropna(subset=["score"])
log(f"Rows with score    : {len(scored):,}  ({len(scored)/len(train)*100:.1f}%)")
log(f"Rows without score : {len(train)-len(scored):,}")
log(f"\nScore stats:\n{scored['score'].describe().to_string()}")

score_counts = scored["score"].value_counts().sort_index()
log(f"\nScore distribution:\n{score_counts.to_string()}")
log(f"\nScore distribution (%):\n{(score_counts / score_counts.sum() * 100).round(2).to_string()}")

fig, axes = plt.subplots(1, 2, figsize=(13, 4))
colors = ["#639922", "#FAC775", "#EF9F27", "#D85A30", "#993C1D", "#4A1B0C"]
bars = axes[0].bar(score_counts.index.astype(str), score_counts.values,
                   color=colors[:len(score_counts)], edgecolor="white", linewidth=0.5)
axes[0].set_title("Score distribution (count)")
axes[0].set_xlabel("Score")
axes[0].set_ylabel("Count")
for bar, val in zip(bars, score_counts.values):
    axes[0].text(bar.get_x() + bar.get_width()/2, bar.get_height() + score_counts.max()*0.01,
                 f"{val/score_counts.sum()*100:.1f}%", ha="center", va="bottom", fontsize=9)

axes[1].hist(scored["score"], bins=30, color="#378ADD", edgecolor="white", linewidth=0.5)
axes[1].set_title("Score distribution (histogram)")
axes[1].set_xlabel("Score value")
axes[1].set_ylabel("Count")

plt.tight_layout()
plt.savefig(OUT_DIR / "02_score_distribution.png", dpi=150)
plt.close()
log("Saved: 02_score_distribution.png")

conclusion(
    "Score is heavily imbalanced: 59.6% of weeks have score=0 (no drought). "
    "A naive model that always predicts 0 would achieve MAE ~0.84, which already "
    "beats Baseline 1 (0.9117). This imbalance means the model must be carefully "
    "designed to avoid always predicting 0. Class weighting or loss adjustment "
    "may be needed to properly predict drought weeks (score >= 1)."
)

# ══════════════════════════════════════════════════════════════════════════════
# 5. MISSING VALUE ANALYSIS
# ══════════════════════════════════════════════════════════════════════════════
log("\n" + "=" * 60)
log("5. MISSING VALUE ANALYSIS")
log("=" * 60)

miss_train = train.isnull().sum()
miss_test  = test.isnull().sum()

log(f"\nTrain missing values:\n{miss_train[miss_train > 0].to_string()}")
log(f"\nTest  missing values:\n{miss_test[miss_test > 0].to_string()}")

miss_pct = (train[METEO_COLS].isnull().sum() / len(train) * 100).sort_values(ascending=False)
log(f"\nMeteo feature missing % (train):\n{miss_pct.to_string()}")

fig, ax = plt.subplots(figsize=(11, 4))
miss_all = train.isnull().sum() / len(train) * 100
miss_all = miss_all[miss_all > 0].sort_values(ascending=False)
ax.barh(miss_all.index, miss_all.values, color="#378ADD", edgecolor="white")
ax.set_xlabel("Missing (%)")
ax.set_title("Missing value rate per column (train)")
ax.axvline(50, color="#E24B4A", linestyle="--", linewidth=1, label="50%")
ax.legend()
plt.tight_layout()
plt.savefig(OUT_DIR / "03_missing_values.png", dpi=150)
plt.close()
log("Saved: 03_missing_values.png")

conclusion(
    "All 14 meteorological features have zero missing values in both train and test. "
    "The only missing column is 'score' (85.7% NaN in train), which is by design — "
    "score is recorded once per week, not daily. No imputation is needed for features. "
    "This is a very clean dataset from a data quality perspective."
)

# ══════════════════════════════════════════════════════════════════════════════
# 6. METEOROLOGICAL FEATURE STATS
# ══════════════════════════════════════════════════════════════════════════════
log("\n" + "=" * 60)
log("6. METEOROLOGICAL FEATURE STATS")
log("=" * 60)
log(f"\n{train[METEO_COLS].describe().T.to_string()}")

fig, axes = plt.subplots(4, 4, figsize=(16, 13))
axes = axes.flatten()
palette = ["#378ADD", "#1D9E75", "#D85A30", "#7F77DD",
           "#378ADD", "#1D9E75", "#D85A30", "#7F77DD",
           "#378ADD", "#1D9E75", "#D85A30", "#7F77DD",
           "#378ADD", "#1D9E75"]

for i, col in enumerate(METEO_COLS):
    data = train[col].dropna()
    axes[i].hist(data, bins=50, color=palette[i], edgecolor="white", linewidth=0.3, alpha=0.9)
    axes[i].set_title(col, fontsize=11)
    axes[i].set_ylabel("Count", fontsize=9)
    axes[i].tick_params(labelsize=8)
    med = data.median()
    axes[i].axvline(med, color="#2C2C2A", linestyle="--", linewidth=1, alpha=0.7)
    axes[i].text(0.97, 0.93, f"med={med:.2f}", transform=axes[i].transAxes,
                 ha="right", va="top", fontsize=8, color="#5F5E5A")

for j in range(len(METEO_COLS), len(axes)):
    axes[j].set_visible(False)

plt.suptitle("Meteorological feature distributions (train)", fontsize=13, y=1.01)
plt.tight_layout()
plt.savefig(OUT_DIR / "04_feature_distributions.png", dpi=150, bbox_inches="tight")
plt.close()
log("Saved: 04_feature_distributions.png")

conclusion(
    "Temperature features (tmp, tmp_max, tmp_min, surf_tmp, dp_tmp, wb_tmp) span wide "
    "ranges (-45 to +50 deg C), suggesting the dataset covers diverse climatic regions. "
    "Precipitation (prec) is heavily right-skewed (median=0.14, max=243), typical for "
    "rainfall data. dp_tmp and wb_tmp are very close in distribution — likely collinear. "
    "Normalization will be needed before feeding into distance-based models."
)

# ══════════════════════════════════════════════════════════════════════════════
# 7. FEATURE-SCORE CORRELATIONS
# ══════════════════════════════════════════════════════════════════════════════
log("\n" + "=" * 60)
log("7. FEATURE-SCORE CORRELATIONS")
log("=" * 60)

corr_df = scored[METEO_COLS + ["score"]].copy()
pearson_corr  = corr_df.corr(method="pearson")["score"].drop("score").sort_values()
spearman_corr = corr_df.corr(method="spearman")["score"].drop("score").sort_values()

log(f"\nPearson correlation with score:\n{pearson_corr.to_string()}")
log(f"\nSpearman correlation with score:\n{spearman_corr.to_string()}")

fig, axes = plt.subplots(1, 2, figsize=(14, 5))
colors_p = ["#E24B4A" if v > 0 else "#378ADD" for v in pearson_corr.values]
colors_s = ["#E24B4A" if v > 0 else "#378ADD" for v in spearman_corr.values]

axes[0].barh(pearson_corr.index, pearson_corr.values, color=colors_p, edgecolor="white")
axes[0].axvline(0, color="#888780", linewidth=0.8)
axes[0].set_title("Pearson correlation with score")
axes[0].set_xlabel("Correlation")

axes[1].barh(spearman_corr.index, spearman_corr.values, color=colors_s, edgecolor="white")
axes[1].axvline(0, color="#888780", linewidth=0.8)
axes[1].set_title("Spearman correlation with score")
axes[1].set_xlabel("Correlation")

plt.tight_layout()
plt.savefig(OUT_DIR / "05_feature_score_correlation.png", dpi=150)
plt.close()
log("Saved: 05_feature_score_correlation.png")

conclusion(
    "No single feature strongly predicts score — the highest is tmp_range at 0.222 "
    "(Spearman), followed by surf_pre at -0.135 and tmp_max at 0.125. "
    "All others below 0.11. Spearman > Pearson for most features, indicating nonlinear "
    "relationships. Linear regression is unsuitable; tree-based models (LightGBM, "
    "XGBoost) are strongly preferred. Feature combinations and lag features will matter "
    "far more than any individual raw feature."
)

# ══════════════════════════════════════════════════════════════════════════════
# 8. FEATURE CORRELATION MATRIX
# ══════════════════════════════════════════════════════════════════════════════
log("\n" + "=" * 60)
log("8. FEATURE CORRELATION MATRIX")
log("=" * 60)

corr_matrix = corr_df.corr(method="spearman")
fig, ax = plt.subplots(figsize=(13, 11))
mask = np.triu(np.ones_like(corr_matrix, dtype=bool))
sns.heatmap(
    corr_matrix, mask=mask, ax=ax,
    cmap="RdBu_r", center=0, vmin=-1, vmax=1,
    annot=True, fmt=".2f", annot_kws={"size": 8},
    linewidths=0.4, linecolor="white",
    cbar_kws={"shrink": 0.8}
)
ax.set_title("Spearman correlation matrix (scored rows)", fontsize=13)
plt.tight_layout()
plt.savefig(OUT_DIR / "06_correlation_heatmap.png", dpi=150)
plt.close()
log("Saved: 06_correlation_heatmap.png")

conclusion(
    "Strong multicollinearity among temperature features: tmp, surf_tmp, tmp_max, "
    "tmp_min, dp_tmp, wb_tmp are all highly correlated. Keeping all adds redundancy. "
    "Recommended: keep tmp, tmp_range, dp_tmp as representatives; drop wb_tmp "
    "(nearly identical to dp_tmp). Wind features carry independent signal."
)

# ══════════════════════════════════════════════════════════════════════════════
# 9. SCORE VS KEY FEATURES (box plots)
# ══════════════════════════════════════════════════════════════════════════════
log("\n" + "=" * 60)
log("9. SCORE VS KEY FEATURES")
log("=" * 60)

key_features = ["prec", "humidity", "tmp", "dp_tmp", "wind", "surf_pre"]

fig, axes = plt.subplots(2, 3, figsize=(15, 9))
axes = axes.flatten()

for i, col in enumerate(key_features):
    data_by_score = [scored[scored["score"] == s][col].dropna().values
                     for s in sorted(scored["score"].unique())]
    score_labels = sorted(scored["score"].unique())
    bp = axes[i].boxplot(data_by_score, labels=[str(int(s)) for s in score_labels],
                         patch_artist=True, medianprops={"color": "black", "linewidth": 1.5})
    box_colors = ["#639922", "#FAC775", "#EF9F27", "#D85A30", "#993C1D", "#4A1B0C"]
    for patch, color in zip(bp["boxes"], box_colors[:len(score_labels)]):
        patch.set_facecolor(color)
        patch.set_alpha(0.75)
    axes[i].set_title(f"{col} vs score")
    axes[i].set_xlabel("Score")
    axes[i].set_ylabel(col)

plt.suptitle("Key meteorological features by drought score", fontsize=13, y=1.01)
plt.tight_layout()
plt.savefig(OUT_DIR / "07_score_vs_features_boxplot.png", dpi=150, bbox_inches="tight")
plt.close()
log("Saved: 07_score_vs_features_boxplot.png")

conclusion(
    "Feature medians shift gradually with score level but distributions heavily overlap "
    "across all score levels — no single feature cleanly separates drought severity. "
    "prec and humidity trend slightly down at higher scores; tmp trends up. "
    "Model must rely on feature combinations rather than any single threshold."
)

# ══════════════════════════════════════════════════════════════════════════════
# 10. TEMPORAL PATTERNS
# ══════════════════════════════════════════════════════════════════════════════
log("\n" + "=" * 60)
log("10. TEMPORAL PATTERNS")
log("=" * 60)

monthly_score = scored.groupby("month")["score"].mean()
dow_score     = scored.groupby("day_of_week")["score"].mean()

log(f"\nAvg score by month:\n{monthly_score.to_string()}")
log(f"\nAvg score by day of week:\n{dow_score.to_string()}")

fig, axes = plt.subplots(1, 2, figsize=(13, 4))
month_labels = ["Jan", "Feb", "Mar", "Apr", "May", "Jun",
                "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]
axes[0].bar(monthly_score.index, monthly_score.values, color="#378ADD", edgecolor="white")
axes[0].set_xticks(range(1, 13))
axes[0].set_xticklabels(month_labels, rotation=30, ha="right")
axes[0].set_title("Mean drought score by month")
axes[0].set_ylabel("Mean score")

axes[1].bar(dow_score.index, dow_score.values, color="#1D9E75", edgecolor="white")
axes[1].set_xticks(range(7))
axes[1].set_title("Mean drought score by day of week")
axes[1].set_ylabel("Mean score")

plt.tight_layout()
plt.savefig(OUT_DIR / "08_temporal_patterns.png", dpi=150)
plt.close()
log("Saved: 08_temporal_patterns.png")

conclusion(
    "Seasonal signal exists but is weak: monthly mean score varies only 0.75-0.94 "
    "(range of 0.19). Jul-Sep trend slightly higher. Day of week is completely flat "
    "(0.80-0.86) — no predictive value, drop as feature. Month will be kept but "
    "expected to contribute minimally on its own."
)

# ══════════════════════════════════════════════════════════════════════════════
# 11. SCORE AUTOCORRELATION + TRANSITION MATRIX
# ══════════════════════════════════════════════════════════════════════════════
log("\n" + "=" * 60)
log("11. SCORE AUTOCORRELATION (sample regions)")
log("=" * 60)

sample_regions = region_counts_train[region_counts_train >= 5480].index[:5].tolist()
lag_range = range(1, 21)

fig, axes = plt.subplots(1, 2, figsize=(14, 4))

all_autocorrs = []
for rid in sample_regions:
    region_scores = (train[train["region_id"] == rid]
                     .dropna(subset=["score"])
                     .sort_values("date")["score"]
                     .values)
    autocorrs = [pd.Series(region_scores).autocorr(lag=l) for l in lag_range]
    all_autocorrs.append(autocorrs)
    axes[0].plot(list(lag_range), autocorrs, alpha=0.6, linewidth=1.2)

mean_ac = np.nanmean(all_autocorrs, axis=0)
axes[0].plot(list(lag_range), mean_ac, color="#E24B4A", linewidth=2.5, label="Mean")
axes[0].axhline(0, color="#888780", linewidth=0.8, linestyle="--")
axes[0].set_title("Score autocorrelation (weekly lags)")
axes[0].set_xlabel("Lag (weeks)")
axes[0].set_ylabel("Autocorrelation")
axes[0].legend()

log("Mean autocorrelation by lag (weeks 1-20):")
for lag, ac in zip(lag_range, mean_ac):
    log(f"  Lag {lag:2d}: {ac:.4f}")

score_vals  = sorted(scored["score"].unique())
transitions = pd.DataFrame(0, index=score_vals, columns=score_vals, dtype=float)
for rid in train["region_id"].unique()[:200]:
    s = (train[train["region_id"] == rid]
         .dropna(subset=["score"])
         .sort_values("date")["score"]
         .values)
    for a, b in zip(s[:-1], s[1:]):
        transitions.loc[a, b] += 1

transitions_norm = transitions.div(transitions.sum(axis=1), axis=0)
log(f"\nScore transition matrix (normalized, sample 200 regions):\n"
    f"{transitions_norm.round(3).to_string()}")

sns.heatmap(transitions_norm, annot=True, fmt=".2f", cmap="Blues",
            ax=axes[1], cbar_kws={"shrink": 0.8},
            linewidths=0.4, linecolor="white",
            annot_kws={"size": 10})
axes[1].set_title("Score transition probabilities\n(from row to column, sample 200 regions)")
axes[1].set_xlabel("Next score")
axes[1].set_ylabel("Current score")

plt.tight_layout()
plt.savefig(OUT_DIR / "09_autocorr_transitions.png", dpi=150)
plt.close()
log("Saved: 09_autocorr_transitions.png")

conclusion(
    "Most important finding in the entire EDA. "
    "Lag-1 autocorrelation = 0.936 — last week's score is an extremely strong predictor. "
    "Lag-5 (furthest prediction target) = 0.679, still very high. "
    "Transition matrix confirms drought is persistent: score stays the same ~90% of "
    "the time week-to-week. The last known score before the prediction window is the "
    "single most valuable feature. Lag score features at lags 1-4 should be the "
    "top priority in feature engineering."
)

# ══════════════════════════════════════════════════════════════════════════════
# 12. REGION VARIABILITY
# ══════════════════════════════════════════════════════════════════════════════
log("\n" + "=" * 60)
log("12. REGION VARIABILITY")
log("=" * 60)

region_score_stats = scored.groupby("region_id")["score"].agg(["mean", "std", "max"])
log(f"\nPer-region score stats:\n{region_score_stats.describe().to_string()}")

fig, axes = plt.subplots(1, 3, figsize=(15, 4))
for ax, col, color, label in zip(
    axes,
    ["mean", "std", "max"],
    ["#378ADD", "#7F77DD", "#D85A30"],
    ["Mean score", "Std dev of score", "Max score"]
):
    ax.hist(region_score_stats[col].dropna(), bins=40, color=color,
            edgecolor="white", linewidth=0.4)
    ax.set_title(f"Per-region {label}")
    ax.set_xlabel(label)
    ax.set_ylabel("Number of regions")

plt.suptitle("Region-level score variability across all regions", fontsize=13, y=1.01)
plt.tight_layout()
plt.savefig(OUT_DIR / "10_region_variability.png", dpi=150, bbox_inches="tight")
plt.close()
log("Saved: 10_region_variability.png")

conclusion(
    "Regions have very different drought baselines: per-region mean score ranges from "
    "0.08 (almost never drought) to 2.26 (frequently severe). A global model without "
    "region-level information will struggle. Recommended: include per-region historical "
    "mean score as a feature (region baseline). This encodes each region's long-term "
    "drought tendency and should significantly improve predictions."
)

# ══════════════════════════════════════════════════════════════════════════════
# 13. SAMPLE TIME SERIES
# ══════════════════════════════════════════════════════════════════════════════
log("\n" + "=" * 60)
log("13. SAMPLE TIME SERIES PLOTS")
log("=" * 60)

sample_rid = sample_regions[0]
r_data = train[train["region_id"] == sample_rid].sort_values("date").reset_index(drop=True)
r_data["day_idx"] = np.arange(len(r_data))
r_scored = r_data.dropna(subset=["score"])

fig, axes = plt.subplots(3, 1, figsize=(14, 10), sharex=True)
axes[0].step(r_scored["day_idx"], r_scored["score"], where="post",
             color="#D85A30", linewidth=1.5, label="score")
axes[0].set_ylabel("Drought score")
axes[0].set_title(f"Region {sample_rid}: drought score over time")
axes[0].set_ylim(-0.2, 5.5)
axes[0].legend()

axes[1].plot(r_data["day_idx"], r_data["prec"], color="#378ADD", linewidth=0.6, alpha=0.8)
axes[1].set_ylabel("Precipitation")
axes[1].set_title("Precipitation")

axes[2].plot(r_data["day_idx"], r_data["humidity"], color="#1D9E75",
             linewidth=0.6, alpha=0.8, label="humidity")
axes[2].plot(r_data["day_idx"], r_data["tmp"], color="#E24B4A",
             linewidth=0.6, alpha=0.7, label="tmp")
axes[2].set_ylabel("Value")
axes[2].set_title("Humidity & Temperature")
axes[2].legend()

plt.tight_layout()
plt.savefig(OUT_DIR / "11_sample_time_series.png", dpi=150)
plt.close()
log(f"Saved: 11_sample_time_series.png (region {sample_rid})")

conclusion(
    "Drought score is persistent and changes slowly, consistent with high autocorrelation. "
    "Precipitation spikes do not always immediately reduce score — there is a delayed "
    "response. Rolling aggregates over multiple weeks (e.g. 4-week cumulative rainfall) "
    "will be more predictive than single-day precipitation values."
)

# ══════════════════════════════════════════════════════════════════════════════
# 14. TRAIN vs TEST DATE OVERLAP
# ══════════════════════════════════════════════════════════════════════════════
log("\n" + "=" * 60)
log("14. TRAIN vs TEST DATE OVERLAP")
log("=" * 60)

test_rids  = set(test["region_id"].unique())
train_rids = set(train["region_id"].unique())
overlap   = test_rids & train_rids
only_test = test_rids - train_rids

log(f"Test regions also in train : {len(overlap)}")
log(f"Test regions not in train  : {len(only_test)}")
if only_test:
    log(f"  (sample): {list(only_test)[:10]}")

log("\nDate continuity check (sample 5 regions):")
for rid in list(overlap)[:5]:
    train_end  = train[train["region_id"] == rid]["date"].max()
    test_start = test[test["region_id"] == rid]["date"].min()
    log(f"  {rid}: train ends {train_end}, test starts {test_start}")

conclusion(
    "All 2,248 test regions exist in train — no cold-start problem. "
    "We can always retrieve each region's full training history. "
    "Region-level features (historical mean score, historical feature stats) "
    "can be computed from train and used for test prediction."
)

# ══════════════════════════════════════════════════════════════════════════════
# 15. WEEKLY AGGREGATION PREVIEW
# ══════════════════════════════════════════════════════════════════════════════
log("\n" + "=" * 60)
log("15. WEEKLY AGGREGATION PREVIEW")
log("=" * 60)

sample_r = train[train["region_id"] == sample_rid].sort_values("date").reset_index(drop=True).copy()
sample_r["week_id"] = sample_r.index // 7

weekly = sample_r.groupby("week_id").agg(
    score     = ("score", "first"),
    prec_sum  = ("prec", "sum"),
    prec_mean = ("prec", "mean"),
    humidity  = ("humidity", "mean"),
    tmp       = ("tmp", "mean"),
    tmp_max   = ("tmp_max", "max"),
    tmp_min   = ("tmp_min", "min"),
    dp_tmp    = ("dp_tmp", "mean"),
    wind      = ("wind", "mean"),
).dropna(subset=["score"])

log(f"\nWeekly aggregated table shape: {weekly.shape}")
log(f"Sample (first 5 rows):\n{weekly.head().to_string()}")

conclusion(
    "Weekly aggregation produces 782 rows per region — one row per scored week. "
    "This is the correct modeling unit. prec_sum (weekly total rainfall) is more "
    "meaningful than daily prec for drought prediction. Reduces dataset from 12.3M "
    "rows to ~1.76M rows while preserving all signal."
)

# ══════════════════════════════════════════════════════════════════════════════
# 16. DATE INFERENCE — CAN WE RECOVER REAL YEAR?
# ══════════════════════════════════════════════════════════════════════════════
log("\n" + "=" * 60)
log("16. DATE INFERENCE — CAN WE RECOVER REAL YEAR?")
log("=" * 60)

region_start = train.groupby("region_id").first()[["date", "year", "month", "day", "day_of_week"]]

log(f"\nStart date per region (sample 10):")
log(region_start.head(10).to_string())

log(f"\nStart month distribution:")
log(region_start["month"].value_counts().sort_index().to_string())

log(f"\nStart day distribution:")
log(region_start["day"].value_counts().sort_index().to_string())

log(f"\nStart day_of_week distribution:")
log(region_start["day_of_week"].value_counts().sort_index().to_string())

# Check 1: same month/day?
unique_start_md = region_start[["month", "day"]].drop_duplicates()
log(f"\nUnique (month, day) combinations at start: {len(unique_start_md)}")
log(f"Common start: month={unique_start_md['month'].values[0]}, "
    f"day={unique_start_md['day'].values[0]}")

# Check 2: unique years — the decisive check
n_unique_years  = region_start["year"].nunique()
n_total_regions = len(region_start)
log(f"\nTotal regions  : {n_total_regions}")
log(f"Unique years   : {n_unique_years}")
log(f"All years unique: {n_unique_years == n_total_regions}")
log(f"Year range     : {region_start['year'].min()} -> {region_start['year'].max()}")

year_diffs = region_start["year"].diff().dropna()
log(f"\nYear gap between consecutive regions:")
log(f"  Mean: {year_diffs.mean():.2f}, Std: {year_diffs.std():.2f}, "
    f"Min: {year_diffs.min():.0f}, Max: {year_diffs.max():.0f}")

fig, ax = plt.subplots(figsize=(11, 4))
ax.hist(region_start["year"].values, bins=50, color="#378ADD", edgecolor="white", linewidth=0.4)
ax.set_title("Distribution of start years across regions")
ax.set_xlabel("Start year (fictional)")
ax.set_ylabel("Number of regions")
plt.tight_layout()
plt.savefig(OUT_DIR / "12_region_start_years.png", dpi=150)
plt.close()
log("Saved: 12_region_start_years.png")

conclusion(
    "All regions share the same start date (Dec 31) but every region has a completely "
    "unique year (2,248 regions = 2,248 unique years, range 3004-58046). "
    "Year gaps between regions are random (std=13,796, min=-54,964, max=+43,988). "
    "The shared Dec 31 start is a dataset design choice, NOT evidence of a real year. "
    "VERDICT: Real year cannot be inferred. The 'year' column must be dropped entirely. "
    "Month and day are still usable for seasonality since they are consistent."
)

# ══════════════════════════════════════════════════════════════════════════════
# 17. SCORE SPARSITY PATTERN
# ══════════════════════════════════════════════════════════════════════════════
log("\n" + "=" * 60)
log("17. SCORE SPARSITY PATTERN")
log("=" * 60)

# 17a. Gap between consecutive scores
log("\n17a. Gap (days) between consecutive scores (sample 100 regions):")
gaps = []
sample_100 = train["region_id"].unique()[:100]
for rid in sample_100:
    r = train[train["region_id"] == rid].sort_values("jdn")
    r_sc = r.dropna(subset=["score"])
    if len(r_sc) > 1:
        gaps.extend(r_sc["jdn"].diff().dropna().values.tolist())

gaps = np.array(gaps)
gap_counts = pd.Series(gaps).value_counts().sort_index()
log(f"Unique gap values : {sorted(gap_counts.index.tolist())}")
log(f"Gap distribution  :\n{gap_counts.to_string()}")
log(f"Mean gap          : {gaps.mean():.4f}")
log(f"All gaps == 7     : {(gaps == 7).all()}")

fig, ax = plt.subplots(figsize=(9, 4))
ax.bar(gap_counts.index.astype(str), gap_counts.values, color="#378ADD", edgecolor="white")
ax.set_title("Gap (days) between consecutive scores — same region (sample 100)")
ax.set_xlabel("Gap (days)")
ax.set_ylabel("Count")
plt.tight_layout()
plt.savefig(OUT_DIR / "13_score_gaps.png", dpi=150)
plt.close()
log("Saved: 13_score_gaps.png")

# 17b. Which day of week carries the score?
log("\n17b. Day of week of scored rows:")
dow_score_dist = scored["day_of_week"].value_counts().sort_index()
log(dow_score_dist.to_string())
log(f"Unique day_of_week values with score: {sorted(scored['day_of_week'].unique())}")
if len(dow_score_dist) == 1:
    log(f"-> Score ALWAYS falls on the same day of week (day_of_week={dow_score_dist.index[0]})")
else:
    log(f"-> Score falls on {len(dow_score_dist)} different days of week")

fig, ax = plt.subplots(figsize=(9, 4))
ax.bar(dow_score_dist.index.astype(str), dow_score_dist.values, color="#1D9E75", edgecolor="white")
ax.set_title("Day of week distribution for scored rows")
ax.set_xlabel("Day of week (JDN % 7)")
ax.set_ylabel("Count")
plt.tight_layout()
plt.savefig(OUT_DIR / "14_score_day_of_week.png", dpi=150)
plt.close()
log("Saved: 14_score_day_of_week.png")

# 17c. Any weeks with missing score?
log("\n17c. Check for weeks with missing score (sample 100 regions):")
missing_score_weeks = []
for rid in sample_100:
    r = train[train["region_id"] == rid].sort_values("jdn").reset_index(drop=True)
    r["week_id"] = r.index // 7
    weeks_with_score = r.dropna(subset=["score"])["week_id"].nunique()
    total_weeks = r["week_id"].nunique()
    if weeks_with_score < total_weeks:
        missing_score_weeks.append({
            "region_id": rid,
            "total_weeks": total_weeks,
            "weeks_with_score": weeks_with_score,
            "missing": total_weeks - weeks_with_score
        })

if missing_score_weeks:
    df_miss = pd.DataFrame(missing_score_weeks)
    log(f"Regions with missing score weeks: {len(df_miss)}")
    log(df_miss.head(10).to_string())
else:
    log("-> All weeks have exactly 1 score. No missing weeks found.")

# 17d. Scores per week
log("\n17d. Number of scores per week (region R1):")
r1 = train[train["region_id"] == "R1"].sort_values("jdn").reset_index(drop=True)
r1["week_id"] = r1.index // 7
scores_per_week = r1.groupby("week_id")["score"].count()
log(f"Scores per week distribution:\n{scores_per_week.value_counts().to_string()}")

conclusion(
    "Score sparsity is structured and consistent — not random. "
    "Gap between scores is almost always exactly 7 days (99.1% of cases). "
    "Rare gap=6 or gap=8 (~0.9%) are calendar boundary effects at Dec 31/Jan 1 — "
    "not data errors. Score falls equally on all 7 days of week — day_of_week has "
    "no predictive signal and should be dropped. Each region has exactly 1 missing "
    "week (the first partial week on Dec 31). Dataset quality is high."
)

# ══════════════════════════════════════════════════════════════════════════════
# 18. TEST CONTINUITY + GAP WITH TRAIN
# ══════════════════════════════════════════════════════════════════════════════
log("\n" + "=" * 60)
log("18. TEST CONTINUITY + GAP WITH TRAIN")
log("=" * 60)

# 18a. Are 91 test days consecutive?
log("\n18a. Check if 91 test days are consecutive (sample 20 regions):")
non_consecutive = []
for rid in test["region_id"].unique()[:20]:
    t = test[test["region_id"] == rid].sort_values("jdn")
    diffs = t["jdn"].diff().dropna()
    if not (diffs == 1).all():
        non_consecutive.append({"region_id": rid, "unique_diffs": diffs.unique().tolist()})

if non_consecutive:
    log(f"Regions with non-consecutive days: {len(non_consecutive)}")
    for r in non_consecutive:
        log(f"  {r}")
else:
    log("-> All 91 test days are CONSECUTIVE (JDN diff = 1 every day)")

# 18b. Exact gap between train end and test start
log("\n18b. Gap (days) between train end and test start (sample 50 regions):")
common_regions = list(overlap)
gaps_tt = []
for rid in common_regions[:50]:
    train_last = train[train["region_id"] == rid]["jdn"].max()
    test_first = test[test["region_id"] == rid]["jdn"].min()
    gaps_tt.append(test_first - train_last)

gaps_tt  = np.array(gaps_tt)
gap_dist = pd.Series(gaps_tt).value_counts().sort_index()
log(f"Gap distribution:\n{gap_dist.to_string()}")
log(f"Mean gap : {gaps_tt.mean():.2f} days (~{gaps_tt.mean()/365:.1f} years)")
log(f"Min gap  : {gaps_tt.min()} days")
log(f"Max gap  : {gaps_tt.max()} days")
log(f"All gaps identical: {len(gap_dist) == 1}")

# 18c. Day of week at train end and test start
log("\n18c. Day_of_week at train end and test start:")
train_end_dows  = [train[train["region_id"] == rid].sort_values("jdn").iloc[-1]["day_of_week"]
                   for rid in common_regions[:50]]
test_start_dows = [test[test["region_id"] == rid].sort_values("jdn").iloc[0]["day_of_week"]
                   for rid in test["region_id"].unique()[:50]]
log(f"Train end   day_of_week:\n{pd.Series(train_end_dows).value_counts().sort_index().to_string()}")
log(f"Test  start day_of_week:\n{pd.Series(test_start_dows).value_counts().sort_index().to_string()}")

fig, axes = plt.subplots(1, 2, figsize=(13, 4))
axes[0].bar(gap_dist.index.astype(str), gap_dist.values, color="#378ADD", edgecolor="white")
axes[0].set_title("Gap (days): train end to test start")
axes[0].set_xlabel("Gap (days)")
axes[0].set_ylabel("Count of regions")
axes[0].tick_params(axis='x', rotation=90)

axes[1].barh(["Train\n(5480d)", "Gap\n(avg)", "Test\n(91d)", "Predict\n(35d)"],
             [5480, gaps_tt.mean(), 91, 35],
             color=["#378ADD", "#FAC775", "#1D9E75", "#D85A30"],
             edgecolor="white", height=0.5)
axes[1].set_title("Timeline: days per phase")
axes[1].set_xlabel("Number of days")
for i, v in enumerate([5480, gaps_tt.mean(), 91, 35]):
    axes[1].text(v + 20, i, f"{v:.0f}d", va="center", fontsize=10)

plt.tight_layout()
plt.savefig(OUT_DIR / "15_test_continuity_gap.png", dpi=150)
plt.close()
log("Saved: 15_test_continuity_gap.png")

conclusion(
    "91 test days are perfectly consecutive for all regions. "
    "However, gap between train end and test start varies widely across regions: "
    f"min={gaps_tt.min():.0f} days, max={gaps_tt.max():.0f} days, "
    f"mean={gaps_tt.mean():.0f} days (~{gaps_tt.mean()/365:.1f} year). "
    "This is a critical modeling challenge: lag score features from train end will be "
    "stale by different amounts per region. Regions with large gaps (>500 days) are "
    "harder to predict using lag features alone. The 91-day test window provides "
    "recent meteorological context but no score labels."
)

# ══════════════════════════════════════════════════════════════════════════════
# 19. TRAIN vs TEST FEATURE DISTRIBUTION
# ══════════════════════════════════════════════════════════════════════════════
log("\n" + "=" * 60)
log("19. TRAIN vs TEST FEATURE DISTRIBUTION COMPARISON")
log("=" * 60)

train_stats = train[METEO_COLS].describe().T[["mean", "std", "min", "max"]]
test_stats  = test[METEO_COLS].describe().T[["mean", "std", "min", "max"]]
train_stats.columns = ["train_mean", "train_std", "train_min", "train_max"]
test_stats.columns  = ["test_mean",  "test_std",  "test_min",  "test_max"]
comparison = pd.concat([train_stats, test_stats], axis=1)
comparison["mean_diff_pct"] = (
    (comparison["test_mean"] - comparison["train_mean"]).abs()
    / comparison["train_mean"].abs() * 100
).round(2)

log(f"\nTrain vs Test feature statistics:\n{comparison.to_string()}")

significant = comparison[comparison["mean_diff_pct"] > 5]
if len(significant) > 0:
    log(f"\nFeatures with mean_diff_pct > 5%:\n"
        f"{significant[['train_mean', 'test_mean', 'mean_diff_pct']].to_string()}")
else:
    log("\n-> All features have mean_diff < 5% — train and test distributions are similar")

top_features = ["tmp_min", "dp_tmp", "wb_tmp", "surf_tmp", "tmp", "tmp_max"]
fig, axes = plt.subplots(2, 3, figsize=(15, 9))
axes = axes.flatten()

for i, col in enumerate(top_features):
    ax = axes[i]
    train_sample = train[col].dropna().sample(min(50000, len(train)), random_state=42)
    test_sample  = test[col].dropna()
    ax.hist(train_sample, bins=50, alpha=0.6, color="#378ADD",
            density=True, label="Train", edgecolor="none")
    ax.hist(test_sample,  bins=50, alpha=0.6, color="#D85A30",
            density=True, label="Test",  edgecolor="none")
    ax.set_title(col)
    ax.set_xlabel("Value")
    ax.set_ylabel("Density")
    ax.legend(fontsize=9)
    diff_pct = comparison.loc[col, "mean_diff_pct"]
    ax.text(0.97, 0.93, f"mean diff: {diff_pct:.1f}%",
            transform=ax.transAxes, ha="right", va="top", fontsize=9,
            bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.7))

plt.suptitle("Feature distribution: Train vs Test (density)", fontsize=13, y=1.01)
plt.tight_layout()
plt.savefig(OUT_DIR / "16_train_test_distribution.png", dpi=150, bbox_inches="tight")
plt.close()
log("Saved: 16_train_test_distribution.png")

# KS test
log("\nKolmogorov-Smirnov test (train vs test):")
log(f"{'Feature':<12} {'KS stat':>10} {'p-value':>12} {'Same dist?':>12}")
log("-" * 50)
for col in METEO_COLS:
    s1 = train[col].dropna().sample(min(10000, len(train)), random_state=42)
    s2 = test[col].dropna().sample(min(len(test[col].dropna()), 10000), random_state=42)
    ks_stat, p_val = scipy_stats.ks_2samp(s1, s2)
    same = "YES" if p_val > 0.05 else "NO"
    log(f"{col:<12} {ks_stat:>10.4f} {p_val:>12.4f} {same:>12}")

conclusion(
    "Most critical finding for modeling strategy. "
    "13/14 features have significantly different distributions between train and test "
    "(KS test p~0). Temperature features differ most: tmp_min +70.9%, dp_tmp +61.5%, "
    "tmp +44.6%. Test data is consistently warmer and drier than train average. "
    "Most likely cause: test window falls in summer for most regions while train "
    "covers all 4 seasons over 15 years. This is a covariate shift problem. "
    "Mitigation: (1) train only on summer-like weeks, (2) weight training samples "
    "by similarity to test feature distribution, (3) include month as feature. "
    "Only surf_pre has similar distribution — it is season-independent."
)

# ══════════════════════════════════════════════════════════════════════════════
# SAVE SUMMARY
# ══════════════════════════════════════════════════════════════════════════════
with open(OUT_DIR / "eda_summary.txt", "w", encoding="utf-8") as f:
    f.write("\n".join(summary_lines))

log("\n" + "=" * 60)
log("EDA COMPLETE")
log(f"All outputs saved to: {OUT_DIR}/")
log("=" * 60)
print("\nFiles generated:")
for p in sorted(OUT_DIR.iterdir()):
    print(f"  {p.name}")