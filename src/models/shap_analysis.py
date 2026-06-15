"""
SHAP Explainability Analysis
==============================
Generates:
  - Global feature importance (SHAP bar + beeswarm)
  - SHAP dependence plots for top features
  - Individual deal explanations (waterfall plots)
  - Finance insights derived from SHAP values
"""

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import shap
import xgboost as xgb
import lightgbm as lgb
import warnings, os, json
warnings.filterwarnings("ignore")

os.makedirs("results/figures", exist_ok=True)
os.makedirs("results/shap",    exist_ok=True)
os.chdir("C:/Users/ASUS/Downloads/ma_premium_predictor_v2/ma_premium_predictor")

PALETTE = {
    "blue": "#1f4e79", "teal": "#0d6e6e", "amber": "#e07b00",
    "red": "#c0392b", "gray": "#636e72", "light": "#dfe6e9",
    "green": "#27ae60", "purple": "#6c3483"
}
plt.rcParams.update({
    "figure.facecolor": "white", "axes.facecolor": "white",
    "axes.spines.top": False, "axes.spines.right": False,
    "axes.grid": True, "grid.alpha": 0.3, "font.size": 11,
})

# ── Load data and model ───────────────────────────────────────────────────────
df = pd.read_csv("data/processed/ma_deals_features.csv")
with open("results/models/feature_list.json") as f:
    FEATURES = json.load(f)

TARGET = "premium_1w"
train  = df[df["split"] == "train"]
val    = df[df["split"] == "val"]
test   = df[df["split"] == "test"]

X_train = train[FEATURES].values
X_val   = val[FEATURES].values
X_test  = test[FEATURES].values
y_test  = test[TARGET].values

# We run SHAP on XGBoost (tree explainer supports exact SHAP values)
# Re-train XGBoost on train+val combined for final SHAP analysis
X_trainval = np.vstack([X_train, X_val])
y_trainval = pd.concat([train[TARGET], val[TARGET]]).values

xgb_model = xgb.XGBRegressor(
    n_estimators=500, max_depth=4, learning_rate=0.05,
    subsample=0.8, colsample_bytree=0.7, min_child_weight=5,
    reg_alpha=0.1, reg_lambda=1.0,
    n_jobs=-1, random_state=42, verbosity=0
)
xgb_model.fit(X_trainval, y_trainval)

# Feature DataFrame for SHAP
X_test_df = pd.DataFrame(X_test, columns=FEATURES).apply(pd.to_numeric, errors="coerce").fillna(0)
X_all_df  = pd.DataFrame(df[FEATURES].values, columns=FEATURES).apply(pd.to_numeric, errors="coerce").fillna(0)

print("Computing SHAP values...")
explainer   = shap.TreeExplainer(xgb_model)
shap_expl   = explainer(X_test_df)        # Explanation object (for beeswarm/waterfall)
sv_matrix   = explainer.shap_values(X_test_df)  # ndarray (n_test, n_features)
print(f"SHAP matrix: {sv_matrix.shape}")


# ══════════════════════════════════════════════════════════════════════════════
# FIGURE 9 — Global Feature Importance (Mean |SHAP|)
# ══════════════════════════════════════════════════════════════════════════════
mean_abs_shap = np.abs(sv_matrix).mean(axis=0)
importance_df = pd.DataFrame({
    "feature":     FEATURES,
    "mean_abs_shap": mean_abs_shap
}).sort_values("mean_abs_shap", ascending=False)

importance_df.to_csv("results/shap/feature_importance.csv", index=False)

fig, ax = plt.subplots(figsize=(10, 9))
top_n   = 25
top_imp = importance_df.head(top_n)
colors  = [PALETTE["teal"] if i < 5 else PALETTE["blue"] if i < 15 else PALETTE["gray"]
           for i in range(len(top_imp))]
bars = ax.barh(top_imp["feature"][::-1], top_imp["mean_abs_shap"][::-1],
               color=colors[::-1], edgecolor="white")
for bar, val in zip(bars, top_imp["mean_abs_shap"][::-1]):
    ax.text(val + 0.02, bar.get_y() + bar.get_height()/2,
            f"{val:.3f}", va="center", fontsize=8.5)
ax.set_title("Figure 9 — Global Feature Importance (Mean |SHAP| Value)\n"
             "Measures average impact of each feature on premium prediction",
             fontsize=13, fontweight="bold")
ax.set_xlabel("Mean |SHAP Value| (percentage points of premium)")

legend_patches = [
    mpatches.Patch(color=PALETTE["teal"],  label="Top 5 features"),
    mpatches.Patch(color=PALETTE["blue"],  label="Features 6–15"),
    mpatches.Patch(color=PALETTE["gray"],  label="Features 16–25"),
]
ax.legend(handles=legend_patches, loc="lower right")
plt.tight_layout()
plt.savefig("results/figures/fig9_shap_global_importance.png", dpi=150, bbox_inches="tight")
plt.close()
print("✓ Figure 9 saved")


# ══════════════════════════════════════════════════════════════════════════════
# FIGURE 10 — SHAP Beeswarm (Summary Plot)
# ══════════════════════════════════════════════════════════════════════════════
fig, ax = plt.subplots(figsize=(11, 10))
shap.plots.beeswarm(shap_expl, max_display=20, show=False)
plt.title("Figure 10 — SHAP Beeswarm Plot\n"
          "Each dot = one deal. Color = feature value (red=high, blue=low).\n"
          "Position = SHAP impact on premium prediction (pp)",
          fontsize=12, fontweight="bold")
plt.tight_layout()
plt.savefig("results/figures/fig10_shap_beeswarm.png", dpi=150, bbox_inches="tight")
plt.close()
print("✓ Figure 10 saved")


# ══════════════════════════════════════════════════════════════════════════════
# FIGURE 11 — SHAP Dependence Plots (top 6 features)
# ══════════════════════════════════════════════════════════════════════════════
top_features = importance_df["feature"].head(6).tolist()
feature_labels = {
    "auction_intensity":    "Auction Intensity\n(hostile + #bidders − 1)",
    "hostile":              "Hostile Deal\n(0=Friendly, 1=Hostile)",
    "deal_type_enc":        "Deal Type\n(0=Stock, 1=Mixed, 2=Cash)",
    "ev_ebitda":            "Target EV/EBITDA\n(valuation multiple)",
    "is_financial_buyer":   "Financial Buyer\n(0=Strategic, 1=PE)",
    "net_debt_ebitda":      "Net Debt / EBITDA\n(leverage ratio)",
    "vix_at_ann":           "VIX at Announcement\n(market uncertainty)",
    "revenue_growth":       "Revenue Growth YoY\n(target growth rate)",
    "pct_from_52w_high":    "% from 52-Week High\n(stock beaten-down?)",
    "market_cycle":         "Market Cycle\n(−1=Bear, 0=Neutral, 1=Bull)",
    "quality_score":        "Financial Quality Score\n(ROE/ROA/margin composite)",
}

fig, axes = plt.subplots(2, 3, figsize=(16, 10))
fig.suptitle("Figure 11 — SHAP Dependence Plots for Top Features\n"
             "Shows how each feature value drives premium prediction",
             fontsize=14, fontweight="bold")

for ax, feat in zip(axes.flat, top_features):
    feat_idx = FEATURES.index(feat)
    feat_vals = X_test_df[feat].values
    shap_vals = sv_matrix[:, feat_idx]

    sc = ax.scatter(feat_vals, shap_vals, c=feat_vals, cmap="RdBu_r",
                    alpha=0.6, s=20, vmin=np.percentile(feat_vals, 5),
                    vmax=np.percentile(feat_vals, 95))
    ax.axhline(0, color="black", lw=0.8, linestyle="--")
    # Trend line
    try:
        z = np.polyfit(feat_vals, shap_vals, 1)
        p = np.poly1d(z)
        xs = np.linspace(feat_vals.min(), feat_vals.max(), 100)
        ax.plot(xs, p(xs), PALETTE["red"], lw=2, linestyle="--", alpha=0.8)
    except Exception:
        pass
    ax.set_xlabel(feature_labels.get(feat, feat.replace("_", " ").title()), fontsize=9)
    ax.set_ylabel("SHAP Value (pp impact on premium)", fontsize=8)
    ax.set_title(feat.replace("_", " ").title(), fontsize=10)
    plt.colorbar(sc, ax=ax, shrink=0.7, label="Feature Value")

plt.tight_layout()
plt.savefig("results/figures/fig11_shap_dependence.png", dpi=150, bbox_inches="tight")
plt.close()
print("✓ Figure 11 saved")


# ══════════════════════════════════════════════════════════════════════════════
# FIGURE 12 — Waterfall plots: 3 individual deal explanations
# ══════════════════════════════════════════════════════════════════════════════
y_pred_test = xgb_model.predict(X_test)

# Pick 3 interesting deals: high premium, low premium, near-average
high_idx = np.argmax(y_test)
low_idx  = np.argmin(y_test)
avg_idx  = np.argmin(np.abs(y_pred_test - np.median(y_pred_test)))

for idx, label, fname in [
    (high_idx, "High-Premium Deal", "fig12a"),
    (low_idx,  "Low-Premium Deal",  "fig12b"),
    (avg_idx,  "Average-Premium Deal", "fig12c"),
]:
    fig, ax = plt.subplots(figsize=(10, 8))
    shap.plots.waterfall(shap_expl[idx], max_display=15, show=False)
    actual = y_test[idx]
    pred   = y_pred_test[idx]
    plt.title(f"Figure 12 — Deal-Level SHAP Explanation: {label}\n"
              f"Actual Premium: {actual:.1f}%  |  Predicted: {pred:.1f}%",
              fontsize=12, fontweight="bold")
    plt.tight_layout()
    plt.savefig(f"results/figures/{fname}_waterfall_{label.replace(' ','_').lower()}.png",
                dpi=150, bbox_inches="tight")
    plt.close()
    print(f"✓ Figure 12 ({label}) saved — actual={actual:.1f}%, predicted={pred:.1f}%")


# ══════════════════════════════════════════════════════════════════════════════
# FIGURE 13 — Finance Insights: Quantified effects
# ══════════════════════════════════════════════════════════════════════════════
# Extract average SHAP impact of key binary/group variables
def mean_shap_effect(feat):
    """Average SHAP delta: HIGH feature value vs LOW feature value."""
    if feat not in FEATURES:
        return np.nan
    idx   = FEATURES.index(feat)
    vals  = np.array(X_test_df[feat].values, dtype=float)
    shaps = sv_matrix[:, idx]
    unique = np.unique(vals[~np.isnan(vals)])
    if len(unique) <= 2:
        # Binary feature: compare class 1 vs class 0
        m1 = shaps[vals == unique[-1]].mean() if (vals == unique[-1]).sum() > 0 else np.nan
        m0 = shaps[vals == unique[0]].mean()  if (vals == unique[0]).sum()  > 0 else np.nan
    else:
        # Continuous: above vs below median
        med = np.nanmedian(vals)
        hi  = vals > med
        lo  = vals <= med
        m1  = shaps[hi].mean() if hi.sum() > 0 else np.nan
        m0  = shaps[lo].mean() if lo.sum() > 0 else np.nan
    if np.isnan(m1) or np.isnan(m0):
        return np.nan
    return float(m1 - m0)

insights = {
    "Hostile bid":                 mean_shap_effect("hostile"),
    "Financial buyer discount":    mean_shap_effect("is_financial_buyer") * -1,  # invert: negative=discount
    "Additional bidder (auction)": mean_shap_effect("num_bidders"),
    "High EV/EBITDA (expensive)":  mean_shap_effect("ev_ebitda"),
    "Beaten-down stock (low ret)":  mean_shap_effect("pct_from_52w_high") * -1,
    "High VIX (uncertainty)":       mean_shap_effect("vix_at_ann") * -1,
    "High revenue growth":          mean_shap_effect("revenue_growth"),
    "Cross-border deal":            mean_shap_effect("cross_border"),
    "Cash consideration":           mean_shap_effect("deal_certainty_score"),
    "Distressed target":            mean_shap_effect("distress_flag") * -1,
}

insights_df = pd.DataFrame(list(insights.items()), columns=["factor", "premium_impact_pp"])
insights_df = insights_df.sort_values("premium_impact_pp", ascending=True)
insights_df.to_csv("results/shap/finance_insights.csv", index=False)

fig, ax = plt.subplots(figsize=(11, 7))
colors_i = [PALETTE["teal"] if v > 0 else PALETTE["red"] for v in insights_df["premium_impact_pp"]]
bars = ax.barh(insights_df["factor"], insights_df["premium_impact_pp"],
               color=colors_i, edgecolor="white", height=0.6)
ax.axvline(0, color="black", lw=0.8)
for bar, val in zip(bars, insights_df["premium_impact_pp"]):
    ha = "left" if val > 0 else "right"
    offset = 0.15 if val > 0 else -0.15
    ax.text(val + offset, bar.get_y() + bar.get_height()/2,
            f"{val:+.1f}pp", va="center", fontsize=10, fontweight="bold")
ax.set_title("Figure 13 — Quantified Finance Insights from SHAP\n"
             "Average premium impact (pp) of each deal/market characteristic",
             fontsize=13, fontweight="bold")
ax.set_xlabel("Impact on Acquisition Premium (percentage points)")
legend_patches = [
    mpatches.Patch(color=PALETTE["teal"], label="Premium-increasing"),
    mpatches.Patch(color=PALETTE["red"],  label="Premium-decreasing"),
]
ax.legend(handles=legend_patches)
plt.tight_layout()
plt.savefig("results/figures/fig13_finance_insights.png", dpi=150, bbox_inches="tight")
plt.close()
print("✓ Figure 13 saved")

print(f"\n{'='*60}")
print("FINANCE INSIGHTS FROM SHAP ANALYSIS")
print(f"{'='*60}")
for _, row in insights_df.sort_values("premium_impact_pp", ascending=False).iterrows():
    print(f"  {row['factor']:40s}: {row['premium_impact_pp']:+.1f}pp")
