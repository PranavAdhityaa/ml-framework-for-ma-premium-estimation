"""
Exploratory Data Analysis
==========================
Generates all EDA charts for the final report and GitHub.
"""

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns
from scipy import stats
import warnings, os

warnings.filterwarnings("ignore")
os.makedirs("results/figures", exist_ok=True)
os.chdir("C:/Users/ASUS/Downloads/ma_premium_predictor_v2/ma_premium_predictor")

# ── Style ──────────────────────────────────────────────────────────────────────
PALETTE = {
    "blue":  "#1f4e79",
    "teal":  "#0d6e6e",
    "amber": "#e07b00",
    "red":   "#c0392b",
    "gray":  "#636e72",
    "light": "#dfe6e9",
}
plt.rcParams.update({
    "figure.facecolor": "white",
    "axes.facecolor":   "white",
    "axes.spines.top":  False,
    "axes.spines.right":False,
    "axes.grid":        True,
    "grid.alpha":       0.3,
    "grid.color":       "#cccccc",
    "font.family":      "sans-serif",
    "font.size":        11,
    "axes.titlesize":   13,
    "axes.titleweight": "bold",
})

df = pd.read_csv("data/processed/ma_deals_features.csv")
print(f"Loaded: {df.shape}")

# ══════════════════════════════════════════════════════════════════════════════
# FIGURE 1 — Premium Distribution Overview
# ══════════════════════════════════════════════════════════════════════════════
fig, axes = plt.subplots(1, 3, figsize=(16, 5))
fig.suptitle("Figure 1 — Acquisition Premium Distribution (1-Week Basis)",
             fontsize=15, fontweight="bold", y=1.02)

for ax, col, label in zip(axes,
    ["premium_1d", "premium_1w", "premium_4w"],
    ["1-Day Undisturbed", "1-Week Undisturbed", "4-Week Undisturbed"]):
    data = df[col].dropna()
    ax.hist(data, bins=40, color=PALETTE["blue"], alpha=0.8, edgecolor="white", linewidth=0.4)
    ax.axvline(data.mean(),   color=PALETTE["amber"], lw=2, linestyle="--", label=f"Mean {data.mean():.1f}%")
    ax.axvline(data.median(), color=PALETTE["red"],   lw=2, linestyle=":",  label=f"Median {data.median():.1f}%")
    ax.set_title(f"Premium ({label})")
    ax.set_xlabel("Acquisition Premium (%)")
    ax.set_ylabel("Number of Deals")
    ax.legend(fontsize=9)
    # Add normality note
    _, pval = stats.normaltest(data)
    ax.text(0.97, 0.95, f"Skew: {data.skew():.2f}\nKurt: {data.kurtosis():.2f}",
            transform=ax.transAxes, ha="right", va="top", fontsize=9,
            bbox=dict(facecolor="white", alpha=0.8, edgecolor="none"))

plt.tight_layout()
plt.savefig("results/figures/fig1_premium_distribution.png", dpi=150, bbox_inches="tight")
plt.close()
print("✓ Figure 1 saved")


# ══════════════════════════════════════════════════════════════════════════════
# FIGURE 2 — Premium by Deal Characteristics
# ══════════════════════════════════════════════════════════════════════════════
fig, axes = plt.subplots(2, 2, figsize=(14, 10))
fig.suptitle("Figure 2 — Premium by Deal Characteristics", fontsize=15, fontweight="bold")

# 2a: by deal type
ax = axes[0, 0]
dt_order = ["cash", "mixed", "stock"]
dt_means  = [df[df["deal_type"] == t]["premium_1w"].mean() for t in dt_order]
dt_sems   = [df[df["deal_type"] == t]["premium_1w"].sem()  for t in dt_order]
bars = ax.bar(dt_order, dt_means, yerr=dt_sems, capsize=5,
              color=[PALETTE["blue"], PALETTE["teal"], PALETTE["gray"]],
              edgecolor="white", width=0.5)
for bar, val in zip(bars, dt_means):
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
            f"{val:.1f}%", ha="center", va="bottom", fontweight="bold")
ax.set_title("a) By Deal Consideration Type")
ax.set_ylabel("Mean Premium (%) ± SE")
ax.set_xlabel("Consideration Type")

# 2b: by acquirer type
ax = axes[0, 1]
at_order = ["strategic", "financial"]
at_means  = [df[df["acquirer_type"] == t]["premium_1w"].mean() for t in at_order]
at_sems   = [df[df["acquirer_type"] == t]["premium_1w"].sem()  for t in at_order]
at_colors = [PALETTE["teal"], PALETTE["amber"]]
bars = ax.bar(["Strategic Buyer", "Financial Buyer (PE)"], at_means, yerr=at_sems,
              capsize=5, color=at_colors, edgecolor="white", width=0.4)
for bar, val in zip(bars, at_means):
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
            f"{val:.1f}%", ha="center", va="bottom", fontweight="bold")
ax.set_title("b) Strategic vs Financial Buyer")
ax.set_ylabel("Mean Premium (%) ± SE")

# 2c: hostile vs friendly
ax = axes[1, 0]
h_data = [df[df["hostile"] == 0]["premium_1w"], df[df["hostile"] == 1]["premium_1w"]]
bp = ax.boxplot(h_data,
                patch_artist=True, widths=0.4,
                boxprops=dict(facecolor=PALETTE["blue"], alpha=0.7),
                medianprops=dict(color=PALETTE["amber"], linewidth=2),
                flierprops=dict(marker="o", markersize=3, alpha=0.4))
ax.set_xticks([1, 2])
ax.set_xticklabels(["Friendly", "Hostile"])
means = [d.mean() for d in h_data]
ax.scatter([1, 2], means, marker="D", color=PALETTE["amber"], zorder=5, s=60, label="Mean")
ax.set_title("c) Hostile vs Friendly Deals")
ax.set_ylabel("Premium 1-Week (%)")
ax.legend()

# 2d: by number of bidders
ax = axes[1, 1]
bid_data  = {b: df[df["num_bidders"] == b]["premium_1w"] for b in [1, 2, 3]}
bid_means = {b: d.mean() for b, d in bid_data.items()}
bid_sems  = {b: d.sem()  for b, d in bid_data.items()}
xs = list(bid_data.keys())
ys = [bid_means[b] for b in xs]
ye = [bid_sems[b]  for b in xs]
bars = ax.bar([f"{b} Bidder{'s' if b>1 else ''}" for b in xs],
              ys, yerr=ye, capsize=5,
              color=[PALETTE["blue"], PALETTE["teal"], PALETTE["red"]],
              edgecolor="white", width=0.5)
for bar, val in zip(bars, ys):
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
            f"{val:.1f}%", ha="center", va="bottom", fontweight="bold")
ax.set_title("d) Competition Effect on Premium")
ax.set_ylabel("Mean Premium (%) ± SE")
ax.set_xlabel("Number of Bidders")

plt.tight_layout()
plt.savefig("results/figures/fig2_premium_by_deal_chars.png", dpi=150, bbox_inches="tight")
plt.close()
print("✓ Figure 2 saved")


# ══════════════════════════════════════════════════════════════════════════════
# FIGURE 3 — Premium Over Time
# ══════════════════════════════════════════════════════════════════════════════
fig, axes = plt.subplots(1, 2, figsize=(15, 5))
fig.suptitle("Figure 3 — Temporal Dynamics of Acquisition Premiums", fontsize=15, fontweight="bold")

# 3a: annual mean premium
ax = axes[0]
annual = df.groupby("year")["premium_1w"].agg(["mean", "sem", "count"]).reset_index()
ax.fill_between(annual["year"],
                annual["mean"] - annual["sem"],
                annual["mean"] + annual["sem"],
                alpha=0.2, color=PALETTE["blue"])
ax.plot(annual["year"], annual["mean"], color=PALETTE["blue"], lw=2.5, marker="o", markersize=5)
ax.axhline(annual["mean"].mean(), color=PALETTE["amber"], lw=1.5, linestyle="--",
           label=f"Overall mean {annual['mean'].mean():.1f}%")
# Shade crisis periods
for start, end, label in [(2001, 2002, "Dot-com"), (2008, 2009, "GFC"), (2020, 2020, "COVID")]:
    ax.axvspan(start - 0.5, end + 0.5, alpha=0.1, color=PALETTE["red"], label=label)
ax.set_title("a) Annual Mean Acquisition Premium")
ax.set_xlabel("Announcement Year")
ax.set_ylabel("Mean Premium (%)")
ax.legend(fontsize=9)
ax.set_xlim(1999, 2024)

# 3b: premium by sector
ax = axes[1]
sec_mean = df.groupby("sector")["premium_1w"].mean().sort_values(ascending=True)
colors   = [PALETTE["blue"] if v >= df["premium_1w"].mean() else PALETTE["gray"]
            for v in sec_mean.values]
bars = ax.barh(sec_mean.index, sec_mean.values, color=colors, edgecolor="white")
ax.axvline(df["premium_1w"].mean(), color=PALETTE["amber"], lw=1.5,
           linestyle="--", label="Overall mean")
for bar, val in zip(bars, sec_mean.values):
    ax.text(val + 0.3, bar.get_y() + bar.get_height()/2,
            f"{val:.1f}%", va="center", fontsize=9)
ax.set_title("b) Mean Premium by Sector")
ax.set_xlabel("Mean Premium (%)")
ax.legend()

plt.tight_layout()
plt.savefig("results/figures/fig3_premium_over_time.png", dpi=150, bbox_inches="tight")
plt.close()
print("✓ Figure 3 saved")


# ══════════════════════════════════════════════════════════════════════════════
# FIGURE 4 — Correlation with Key Financial Features
# ══════════════════════════════════════════════════════════════════════════════
fig, axes = plt.subplots(2, 3, figsize=(16, 10))
fig.suptitle("Figure 4 — Premium vs. Key Financial Features", fontsize=15, fontweight="bold")

scatter_feats = [
    ("ev_ebitda",       "EV/EBITDA Multiple",       "Finance: Expensive targets → less upside"),
    ("net_debt_ebitda", "Net Debt / EBITDA",         "Finance: Leverage → acquirer inherits debt"),
    ("revenue_growth",  "Revenue Growth (YoY)",      "Finance: Growth targets command higher bids"),
    ("ret_6m",          "6-Month Price Return (%)",  "Finance: Beaten-down stocks attract bids"),
    ("vix_at_ann",      "VIX at Announcement",       "Finance: Uncertainty compresses premiums"),
    ("deal_val_bn",     "Deal Value (USD Bn)",        "Finance: Large deals face more scrutiny"),
]

for ax, (feat, xlabel, note) in zip(axes.flat, scatter_feats):
    x = df[feat].clip(df[feat].quantile(0.02), df[feat].quantile(0.98))
    y = df["premium_1w"]

    # Compute correlation
    r, p = stats.pearsonr(x.dropna(), y[x.notna()])

    ax.scatter(x, y, alpha=0.25, s=15, color=PALETTE["blue"])
    # Regression line
    m, b = np.polyfit(x.dropna(), y[x.notna()], 1)
    xline = np.linspace(x.min(), x.max(), 100)
    ax.plot(xline, m * xline + b, color=PALETTE["red"], lw=2, linestyle="--")

    ax.set_xlabel(xlabel)
    ax.set_ylabel("Premium 1-Week (%)")
    ax.set_title(feat.replace("_", " ").title())
    ax.text(0.05, 0.93, f"r = {r:.3f}  (p={'<0.001' if p < 0.001 else f'{p:.3f}'})",
            transform=ax.transAxes, fontsize=10,
            bbox=dict(facecolor=PALETTE["light"], alpha=0.8, edgecolor="none"))
    ax.text(0.05, 0.07, note, transform=ax.transAxes, fontsize=8,
            color=PALETTE["gray"], style="italic")

plt.tight_layout()
plt.savefig("results/figures/fig4_scatter_vs_premium.png", dpi=150, bbox_inches="tight")
plt.close()
print("✓ Figure 4 saved")


# ══════════════════════════════════════════════════════════════════════════════
# FIGURE 5 — Correlation Heatmap
# ══════════════════════════════════════════════════════════════════════════════
key_feats = [
    "premium_1w", "deal_type_enc", "is_financial_buyer", "hostile",
    "num_bidders", "auction_intensity", "ev_ebitda", "net_debt_ebitda",
    "revenue_growth", "op_margin", "roe", "pb_ratio", "vix_at_ann",
    "treasury_10y_at_ann", "ret_6m", "ret_12m", "pct_from_52w_high",
    "deal_val_bn", "market_cycle", "quality_score",
]
key_feats = [f for f in key_feats if f in df.columns]
corr = df[key_feats].corr()

fig, ax = plt.subplots(figsize=(14, 11))
mask = np.triu(np.ones_like(corr, dtype=bool))
cmap = sns.diverging_palette(220, 20, as_cmap=True)
sns.heatmap(corr, mask=mask, cmap=cmap, center=0,
            annot=True, fmt=".2f", annot_kws={"size": 8},
            square=True, linewidths=0.5, ax=ax,
            cbar_kws={"shrink": 0.8})
ax.set_title("Figure 5 — Feature Correlation Matrix (Lower Triangle)\n"
             "Cells highlighted show |r| > 0.3",
             fontsize=13, fontweight="bold", pad=15)
plt.tight_layout()
plt.savefig("results/figures/fig5_correlation_heatmap.png", dpi=150, bbox_inches="tight")
plt.close()
print("✓ Figure 5 saved")


# ══════════════════════════════════════════════════════════════════════════════
# FIGURE 6 — Macro Environment Analysis
# ══════════════════════════════════════════════════════════════════════════════
fig, axes = plt.subplots(1, 3, figsize=(16, 5))
fig.suptitle("Figure 6 — Market Conditions and Acquisition Premium", fontsize=15, fontweight="bold")

# 6a: VIX bins
ax = axes[0]
df["vix_bin"] = pd.cut(df["vix_at_ann"], bins=[0, 14, 20, 28, 100],
                        labels=["Low (<14)", "Normal (14-20)", "Elevated (20-28)", "High (>28)"])
vix_mean = df.groupby("vix_bin", observed=True)["premium_1w"].mean()
vix_sem  = df.groupby("vix_bin", observed=True)["premium_1w"].sem()
colors = [PALETTE["teal"], PALETTE["blue"], PALETTE["amber"], PALETTE["red"]]
bars = ax.bar(vix_mean.index, vix_mean.values, yerr=vix_sem, capsize=5,
              color=colors, edgecolor="white")
for bar, val in zip(bars, vix_mean.values):
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.4,
            f"{val:.1f}%", ha="center", va="bottom", fontsize=9, fontweight="bold")
ax.set_title("a) VIX Regime vs Premium")
ax.set_ylabel("Mean Premium (%)")
ax.set_xlabel("VIX Level at Announcement")
ax.tick_params(axis="x", rotation=20)

# 6b: 10Y treasury vs premium scatter
ax = axes[1]
ax.scatter(df["treasury_10y_at_ann"], df["premium_1w"], alpha=0.25, s=15, color=PALETTE["blue"])
m, b = np.polyfit(df["treasury_10y_at_ann"], df["premium_1w"], 1)
xr = np.linspace(df["treasury_10y_at_ann"].min(), df["treasury_10y_at_ann"].max(), 100)
ax.plot(xr, m * xr + b, color=PALETTE["red"], lw=2)
r, _ = stats.pearsonr(df["treasury_10y_at_ann"], df["premium_1w"])
ax.set_title(f"b) Interest Rates vs Premium\n(r = {r:.3f})")
ax.set_xlabel("10-Year Treasury Yield (%)")
ax.set_ylabel("Premium (%)")

# 6c: Market cycle
ax = axes[2]
cycle_map = {-1: "Bear Market", 0: "Neutral", 1: "Bull Market"}
for k, v in cycle_map.items():
    subset = df[df["market_cycle"] == k]["premium_1w"]
    ax.hist(subset, bins=25, alpha=0.6,
            label=f"{v} (n={len(subset)}, μ={subset.mean():.1f}%)",
            color=[PALETTE["red"], PALETTE["gray"], PALETTE["teal"]][k + 1])
ax.set_title("c) Premium in Different Market Cycles")
ax.set_xlabel("Premium (%)")
ax.set_ylabel("Frequency")
ax.legend(fontsize=9)

plt.tight_layout()
plt.savefig("results/figures/fig6_macro_environment.png", dpi=150, bbox_inches="tight")
plt.close()
print("✓ Figure 6 saved")

print("\n✓ All EDA figures saved to results/figures/")
print(f"\nKey EDA insights:")
print(f"  Cash premium vs stock: {df[df.deal_type=='cash']['premium_1w'].mean():.1f}% vs "
      f"{df[df.deal_type=='stock']['premium_1w'].mean():.1f}%")
print(f"  Hostile vs friendly:   {df[df.hostile==1]['premium_1w'].mean():.1f}% vs "
      f"{df[df.hostile==0]['premium_1w'].mean():.1f}%")
print(f"  Strategic vs PE:       {df[df.acquirer_type=='strategic']['premium_1w'].mean():.1f}% vs "
      f"{df[df.acquirer_type=='financial']['premium_1w'].mean():.1f}%")
print(f"  Multi-bidder uplift:   "
      f"{df[df.num_bidders>1]['premium_1w'].mean() - df[df.num_bidders==1]['premium_1w'].mean():.1f}pp")
