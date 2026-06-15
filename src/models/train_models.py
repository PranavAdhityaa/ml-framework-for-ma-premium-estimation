"""
Modeling Pipeline
==================
Models trained:
  1. Linear Regression (baseline)
  2. Ridge Regression
  3. Lasso Regression (feature selection)
  4. ElasticNet
  5. Random Forest
  6. XGBoost (headline model)
  7. LightGBM

Evaluation:
  - Time-based train/val/test split (no look-ahead bias)
  - Metrics: MAE, RMSE, R², % within ±5pp
  - All results saved to results/model_comparison.csv
"""

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns
from sklearn.linear_model import LinearRegression, Ridge, Lasso, ElasticNet
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.pipeline import Pipeline
from sklearn.model_selection import cross_val_score, KFold
import xgboost as xgb
import lightgbm as lgb
import warnings, os, json, joblib

warnings.filterwarnings("ignore")
os.makedirs("results/figures", exist_ok=True)
os.makedirs("results/models",  exist_ok=True)
os.chdir("C:/Users/ASUS/Downloads/ma_premium_predictor_v2/ma_premium_predictor")

# ── Plotting style ─────────────────────────────────────────────────────────────
PALETTE = {
    "blue": "#1f4e79", "teal": "#0d6e6e", "amber": "#e07b00",
    "red": "#c0392b", "gray": "#636e72", "light": "#dfe6e9",
    "green": "#27ae60"
}
plt.rcParams.update({
    "figure.facecolor": "white", "axes.facecolor": "white",
    "axes.spines.top": False, "axes.spines.right": False,
    "axes.grid": True, "grid.alpha": 0.3, "font.size": 11,
    "axes.titlesize": 12, "axes.titleweight": "bold",
})


# ── Load data ─────────────────────────────────────────────────────────────────
df = pd.read_csv("data/processed/ma_deals_features.csv")

TARGET = "premium_1w"
EXCLUDE = {
    "deal_id", "target_id", "year", "month", "sector",
    "deal_type", "acquirer_type", "split",
    "premium_1d", "premium_1w", "premium_4w",
    "vix_bin",
}

bool_cols  = df.select_dtypes(include="bool").columns.tolist()
num_cols   = df.select_dtypes(include=["float64", "int64"]).columns.tolist()
FEATURES   = sorted(set(bool_cols + num_cols) - EXCLUDE)
print(f"Features: {len(FEATURES)}, Target: {TARGET}")

# ── Train / Val / Test split ──────────────────────────────────────────────────
train = df[df["split"] == "train"]
val   = df[df["split"] == "val"]
test  = df[df["split"] == "test"]

X_train, y_train = train[FEATURES].values, train[TARGET].values
X_val,   y_val   = val[FEATURES].values,   val[TARGET].values
X_test,  y_test  = test[FEATURES].values,  test[TARGET].values

print(f"Train: {len(train)} | Val: {len(val)} | Test: {len(test)}")


# ── Metrics ───────────────────────────────────────────────────────────────────
def evaluate(name, y_true, y_pred):
    mae  = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    r2   = r2_score(y_true, y_pred)
    within5 = np.mean(np.abs(y_pred - y_true) <= 5) * 100
    return {"model": name, "MAE": round(mae, 3), "RMSE": round(rmse, 3),
            "R2": round(r2, 4), "Within_5pp": round(within5, 2)}


# ── Scaler ───────────────────────────────────────────────────────────────────
scaler = StandardScaler()
X_train_s = scaler.fit_transform(X_train)
X_val_s   = scaler.transform(X_val)
X_test_s  = scaler.transform(X_test)
X_all_s   = scaler.transform(df[FEATURES].values)


# ══════════════════════════════════════════════════════════════════════════════
# Model Definitions
# ══════════════════════════════════════════════════════════════════════════════
models = {
    "Linear Regression": LinearRegression(),
    "Ridge (α=10)":      Ridge(alpha=10),
    "Lasso (α=0.5)":     Lasso(alpha=0.5, max_iter=5000),
    "ElasticNet":        ElasticNet(alpha=0.5, l1_ratio=0.5, max_iter=5000),
    "Random Forest":     RandomForestRegressor(
        n_estimators=300, max_depth=8, min_samples_leaf=10,
        max_features=0.5, n_jobs=-1, random_state=42
    ),
    "XGBoost": xgb.XGBRegressor(
        n_estimators=500, max_depth=4, learning_rate=0.05,
        subsample=0.8, colsample_bytree=0.7, min_child_weight=5,
        reg_alpha=0.1, reg_lambda=1.0,
        n_jobs=-1, random_state=42, verbosity=0
    ),
    "LightGBM": lgb.LGBMRegressor(
        n_estimators=500, max_depth=4, learning_rate=0.05,
        subsample=0.8, colsample_bytree=0.7, min_child_samples=15,
        reg_alpha=0.1, reg_lambda=1.0,
        n_jobs=-1, random_state=42, verbose=-1
    ),
}

# Use scaled data for linear models, raw for tree models
linear_names = {"Linear Regression", "Ridge (α=10)", "Lasso (α=0.5)", "ElasticNet"}

val_results  = []
test_results = []
trained      = {}
all_preds    = {}

for name, model in models.items():
    is_linear = name in linear_names
    Xtr = X_train_s if is_linear else X_train
    Xva = X_val_s   if is_linear else X_val
    Xte = X_test_s  if is_linear else X_test

    model.fit(Xtr, y_train)
    trained[name] = model

    pred_val  = model.predict(Xva)
    pred_test = model.predict(Xte)
    pred_all  = model.predict(X_all_s if is_linear else df[FEATURES].values)
    all_preds[name] = pred_all

    val_results.append(evaluate(f"{name}", y_val, pred_val))
    test_results.append(evaluate(f"{name}", y_test, pred_test))
    print(f"  {name:25s} | Val MAE={val_results[-1]['MAE']:.2f}  R²={val_results[-1]['R2']:.3f} "
          f"| Test MAE={test_results[-1]['MAE']:.2f}  R²={test_results[-1]['R2']:.3f}")

val_df  = pd.DataFrame(val_results).sort_values("MAE")
test_df = pd.DataFrame(test_results).sort_values("MAE")
val_df.to_csv("results/val_metrics.csv",  index=False)
test_df.to_csv("results/test_metrics.csv", index=False)

# Combined
combined = val_df.merge(test_df, on="model", suffixes=("_val", "_test"))
combined.to_csv("results/model_comparison.csv", index=False)
print(f"\n✓ Results saved to results/model_comparison.csv")


# ── Save best model ────────────────────────────────────────────────────────────
best_name = test_df.iloc[0]["model"]
joblib.dump(trained[best_name], "results/models/best_model.pkl")
joblib.dump(scaler, "results/models/scaler.pkl")
# Save feature list
with open("results/models/feature_list.json", "w") as f:
    json.dump(FEATURES, f)
print(f"✓ Best model ({best_name}) saved")


# ══════════════════════════════════════════════════════════════════════════════
# FIGURE 7 — Model Comparison Dashboard
# ══════════════════════════════════════════════════════════════════════════════
fig = plt.figure(figsize=(16, 12))
fig.suptitle("Figure 7 — Model Performance Comparison", fontsize=15, fontweight="bold")
gs = gridspec.GridSpec(2, 2, figure=fig, hspace=0.4, wspace=0.35)

model_order = [r["model"] for r in sorted(test_results, key=lambda x: x["MAE"])]
colors = [PALETTE["teal"] if m == best_name else PALETTE["blue"] for m in model_order]

# 7a: MAE comparison
ax = fig.add_subplot(gs[0, 0])
mae_val  = [next(r["MAE"] for r in val_results  if r["model"] == m) for m in model_order]
mae_test = [next(r["MAE"] for r in test_results if r["model"] == m) for m in model_order]
x = np.arange(len(model_order))
w = 0.35
ax.bar(x - w/2, mae_val,  w, label="Validation", color=PALETTE["blue"], alpha=0.8)
ax.bar(x + w/2, mae_test, w, label="Test",       color=PALETTE["teal"], alpha=0.8)
ax.set_xticks(x)
ax.set_xticklabels([m.split(" ")[0] for m in model_order], rotation=30, ha="right")
ax.set_title("a) Mean Absolute Error (pp)")
ax.set_ylabel("MAE (percentage points)")
ax.legend()

# 7b: R² comparison
ax = fig.add_subplot(gs[0, 1])
r2_val  = [next(r["R2"] for r in val_results  if r["model"] == m) for m in model_order]
r2_test = [next(r["R2"] for r in test_results if r["model"] == m) for m in model_order]
ax.bar(x - w/2, r2_val,  w, label="Validation", color=PALETTE["blue"], alpha=0.8)
ax.bar(x + w/2, r2_test, w, label="Test",       color=PALETTE["teal"], alpha=0.8)
ax.set_xticks(x)
ax.set_xticklabels([m.split(" ")[0] for m in model_order], rotation=30, ha="right")
ax.set_title("b) R² Score")
ax.set_ylabel("R²")
ax.axhline(0, color="black", lw=0.8)
ax.legend()

# 7c: Best model — predicted vs actual (test set)
ax = fig.add_subplot(gs[1, 0])
best_model = trained[best_name]
is_lin     = best_name in linear_names
X_te_use   = X_test_s if is_lin else X_test
y_pred_best= best_model.predict(X_te_use)
ax.scatter(y_test, y_pred_best, alpha=0.5, s=20, color=PALETTE["blue"])
lims = [min(y_test.min(), y_pred_best.min()) - 2,
        max(y_test.max(), y_pred_best.max()) + 2]
ax.plot(lims, lims, "r--", lw=1.5, label="Perfect prediction")
r2 = r2_score(y_test, y_pred_best)
mae = mean_absolute_error(y_test, y_pred_best)
ax.text(0.05, 0.93, f"R² = {r2:.3f}\nMAE = {mae:.2f}pp",
        transform=ax.transAxes, fontsize=10,
        bbox=dict(facecolor=PALETTE["light"], alpha=0.8, edgecolor="none"))
ax.set_title(f"c) {best_name} — Predicted vs Actual (Test)")
ax.set_xlabel("Actual Premium (%)")
ax.set_ylabel("Predicted Premium (%)")
ax.legend()

# 7d: Residuals distribution
ax = fig.add_subplot(gs[1, 1])
residuals = y_pred_best - y_test
ax.hist(residuals, bins=30, color=PALETTE["blue"], alpha=0.8, edgecolor="white")
ax.axvline(0, color=PALETTE["red"], lw=2, linestyle="--")
ax.axvline(residuals.mean(), color=PALETTE["amber"], lw=2, linestyle="-",
           label=f"Mean residual: {residuals.mean():.2f}pp")
ax.set_title(f"d) Residual Distribution (Test Set)")
ax.set_xlabel("Residual (Predicted − Actual, pp)")
ax.set_ylabel("Count")
ax.legend()
ax.text(0.97, 0.93,
        f"Std: {residuals.std():.2f}pp\n±5pp: {(np.abs(residuals)<=5).mean()*100:.1f}%",
        transform=ax.transAxes, ha="right", va="top", fontsize=9,
        bbox=dict(facecolor=PALETTE["light"], alpha=0.8, edgecolor="none"))

plt.savefig("results/figures/fig7_model_comparison.png", dpi=150, bbox_inches="tight")
plt.close()
print("✓ Figure 7 saved")


# ══════════════════════════════════════════════════════════════════════════════
# Lasso coefficients (interpretable feature selection)
# ══════════════════════════════════════════════════════════════════════════════
lasso = trained["Lasso (α=0.5)"]
lasso_coef = pd.DataFrame({
    "feature": FEATURES,
    "coefficient": lasso.coef_
}).query("coefficient != 0").sort_values("coefficient", key=abs, ascending=False)
lasso_coef.to_csv("results/lasso_selected_features.csv", index=False)
print(f"✓ Lasso selected {len(lasso_coef)} features")

fig, ax = plt.subplots(figsize=(10, 8))
top_lasso = lasso_coef.head(20)
colors_l  = [PALETTE["teal"] if c > 0 else PALETTE["red"] for c in top_lasso["coefficient"]]
bars = ax.barh(top_lasso["feature"], top_lasso["coefficient"], color=colors_l)
ax.axvline(0, color="black", lw=0.8)
ax.set_title("Figure 8 — Lasso Regression: Non-Zero Coefficients\n"
             "(Feature Selection — only factors with non-zero contribution shown)",
             fontsize=12, fontweight="bold")
ax.set_xlabel("Coefficient Value (standardized premium pp per unit)")
ax.invert_yaxis()
plt.tight_layout()
plt.savefig("results/figures/fig8_lasso_coefficients.png", dpi=150, bbox_inches="tight")
plt.close()
print("✓ Figure 8 (Lasso coefficients) saved")

print(f"\n{'='*60}")
print(f"Best model: {best_name}")
test_best = next(r for r in test_results if r["model"] == best_name)
print(f"  Test MAE:     {test_best['MAE']:.2f} percentage points")
print(f"  Test RMSE:    {test_best['RMSE']:.2f} percentage points")
print(f"  Test R²:      {test_best['R2']:.3f}")
print(f"  Within ±5pp:  {test_best['Within_5pp']:.1f}%")
print(f"{'='*60}")
