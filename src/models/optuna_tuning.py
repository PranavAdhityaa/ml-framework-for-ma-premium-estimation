"""
Hyperparameter Optimisation with Optuna
=========================================
Runs 100 trials of Bayesian HPO on XGBoost using time-series-aware
cross-validation. Compares tuned vs baseline XGBoost vs best Lasso.

Design choices:
  - TimeSeriesSplit (5 folds) instead of random KFold — no look-ahead bias
  - Optimises MAE (most interpretable metric for this problem)
  - Logs all trials for analysis
  - Saves tuned model separately from baseline
"""

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import optuna
import xgboost as xgb
from sklearn.model_selection import TimeSeriesSplit, cross_val_score
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.linear_model import Lasso
from sklearn.preprocessing import StandardScaler
import joblib, json, os, warnings
warnings.filterwarnings("ignore")
optuna.logging.set_verbosity(optuna.logging.WARNING)

os.chdir("C:/Users/ASUS/Downloads/ma_premium_predictor_v2/ma_premium_predictor")
os.makedirs("results/figures", exist_ok=True)
os.makedirs("results/models",  exist_ok=True)

PALETTE = {"blue":"#1f4e79","teal":"#0d6e6e","amber":"#e07b00",
           "red":"#c0392b","gray":"#636e72","light":"#dfe6e9","green":"#27ae60"}
plt.rcParams.update({"figure.facecolor":"white","axes.facecolor":"white",
    "axes.spines.top":False,"axes.spines.right":False,
    "axes.grid":True,"grid.alpha":0.3,"font.size":11})

# ── Load data ─────────────────────────────────────────────────────────────────
df = pd.read_csv("data/processed/ma_deals_features.csv")
with open("results/models/feature_list.json") as f:
    FEATURES = json.load(f)
TARGET = "premium_1w"

train = df[df["split"].isin(["train", "val"])].sort_values("year")  # train+val combined
test  = df[df["split"] == "test"]

X_train = train[FEATURES].apply(pd.to_numeric, errors="coerce").fillna(0).values.astype(float)
y_train = train[TARGET].values
X_test  = test[FEATURES].apply(pd.to_numeric, errors="coerce").fillna(0).values.astype(float)
y_test  = test[TARGET].values

print(f"Training set: {X_train.shape}, Test set: {X_test.shape}")

# ── Time-series cross-validation splitter ─────────────────────────────────────
# Critical: use TimeSeriesSplit, not KFold, to respect temporal ordering.
# This means each fold trains on past data and validates on future data.
tscv = TimeSeriesSplit(n_splits=5, gap=10)

# ── Baseline XGBoost (hand-tuned, from original training) ─────────────────────
xgb_baseline = xgb.XGBRegressor(
    n_estimators=500, max_depth=4, learning_rate=0.05,
    subsample=0.8, colsample_bytree=0.7, min_child_weight=5,
    reg_alpha=0.1, reg_lambda=1.0,
    n_jobs=-1, random_state=42, verbosity=0
)
xgb_baseline.fit(X_train, y_train)
pred_baseline = xgb_baseline.predict(X_test)
mae_baseline  = mean_absolute_error(y_test, pred_baseline)
r2_baseline   = r2_score(y_test, pred_baseline)
print(f"\nBaseline XGBoost  → Test MAE: {mae_baseline:.3f}pp  R²: {r2_baseline:.3f}")

# ── Lasso reference (best model from original training) ──────────────────────
scaler = StandardScaler()
X_train_s = scaler.fit_transform(X_train)
X_test_s  = scaler.transform(X_test)
lasso = Lasso(alpha=0.5, max_iter=5000)
lasso.fit(X_train_s, y_train)
pred_lasso = lasso.predict(X_test_s)
mae_lasso  = mean_absolute_error(y_test, pred_lasso)
r2_lasso   = r2_score(y_test, pred_lasso)
print(f"Lasso reference   → Test MAE: {mae_lasso:.3f}pp  R²: {r2_lasso:.3f}")


# ── Optuna objective function ─────────────────────────────────────────────────
trial_history = []

def objective(trial):
    params = {
        "n_estimators":      trial.suggest_int("n_estimators", 100, 1000),
        "max_depth":         trial.suggest_int("max_depth", 2, 8),
        "learning_rate":     trial.suggest_float("learning_rate", 0.005, 0.2, log=True),
        "subsample":         trial.suggest_float("subsample", 0.5, 1.0),
        "colsample_bytree":  trial.suggest_float("colsample_bytree", 0.4, 1.0),
        "min_child_weight":  trial.suggest_int("min_child_weight", 1, 20),
        "reg_alpha":         trial.suggest_float("reg_alpha", 1e-4, 10.0, log=True),
        "reg_lambda":        trial.suggest_float("reg_lambda", 1e-4, 10.0, log=True),
        "gamma":             trial.suggest_float("gamma", 0.0, 2.0),
        "n_jobs": -1, "random_state": 42, "verbosity": 0,
    }
    model = xgb.XGBRegressor(**params)

    # Time-series CV: average MAE across 5 folds
    maes = []
    for train_idx, val_idx in tscv.split(X_train):
        model.fit(X_train[train_idx], y_train[train_idx])
        pred = model.predict(X_train[val_idx])
        maes.append(mean_absolute_error(y_train[val_idx], pred))

    cv_mae = np.mean(maes)
    trial_history.append({"trial": trial.number, "cv_mae": cv_mae, **params})
    return cv_mae


# ── Run optimisation ─────────────────────────────────────────────────────────
print(f"\nRunning Optuna HPO (100 trials, TimeSeriesSplit n=5)...")
study = optuna.create_study(
    direction="minimize",
    sampler=optuna.samplers.TPESampler(seed=42),
    pruner=optuna.pruners.MedianPruner(n_warmup_steps=20),
)
study.optimize(objective, n_trials=100, show_progress_bar=False)

print(f"\n{'='*60}")
print(f"Optimisation complete")
print(f"  Best CV MAE:  {study.best_value:.3f}pp")
print(f"  Best params:")
for k, v in study.best_params.items():
    print(f"    {k:25s}: {v}")

# ── Fit tuned model on full train set ─────────────────────────────────────────
xgb_tuned = xgb.XGBRegressor(
    **study.best_params,
    n_jobs=-1, random_state=42, verbosity=0
)
xgb_tuned.fit(X_train, y_train)
pred_tuned  = xgb_tuned.predict(X_test)
mae_tuned   = mean_absolute_error(y_test, pred_tuned)
r2_tuned    = r2_score(y_test, pred_tuned)
rmse_tuned  = np.sqrt(mean_squared_error(y_test, pred_tuned))
w5_tuned    = np.mean(np.abs(pred_tuned - y_test) <= 5) * 100

print(f"\n{'='*60}")
print(f"FINAL COMPARISON (Test Set)")
print(f"{'='*60}")
print(f"  Baseline XGBoost   → MAE: {mae_baseline:.3f}pp  R²: {r2_baseline:.3f}")
print(f"  Tuned XGBoost      → MAE: {mae_tuned:.3f}pp  R²: {r2_tuned:.3f}  "
      f"(improvement: {(mae_baseline-mae_tuned)/mae_baseline*100:.1f}%)")
print(f"  Lasso (reference)  → MAE: {mae_lasso:.3f}pp  R²: {r2_lasso:.3f}")

# Save tuned model
joblib.dump(xgb_tuned, "results/models/xgb_tuned.pkl")
best_params_save = {**study.best_params, "cv_mae": study.best_value,
                    "test_mae": mae_tuned, "test_r2": r2_tuned}
with open("results/models/xgb_best_params.json", "w") as f:
    json.dump(best_params_save, f, indent=2)
print(f"\n✓ Tuned model saved → results/models/xgb_tuned.pkl")

# Save trial history
trial_df = pd.DataFrame(trial_history)
trial_df.to_csv("results/optuna_trials.csv", index=False)

# ── FIGURE 15 — HPO Analysis ─────────────────────────────────────────────────
fig, axes = plt.subplots(2, 2, figsize=(15, 11))
fig.suptitle("Figure 15 — Optuna Hyperparameter Optimisation (100 Trials, TimeSeriesSplit CV)",
             fontsize=14, fontweight="bold")

# 15a: optimisation history
ax = axes[0, 0]
trials_sorted = trial_df.sort_values("trial")
running_best  = trials_sorted["cv_mae"].cummin()
ax.plot(trials_sorted["trial"], trials_sorted["cv_mae"],
        alpha=0.3, color=PALETTE["gray"], lw=0.8, label="Trial MAE")
ax.plot(trials_sorted["trial"], running_best,
        color=PALETTE["teal"], lw=2.5, label="Best so far")
ax.axhline(mae_baseline, color=PALETTE["red"],    lw=1.5, linestyle="--", label=f"Baseline XGB {mae_baseline:.2f}pp")
ax.axhline(mae_lasso,    color=PALETTE["amber"],  lw=1.5, linestyle=":",  label=f"Lasso {mae_lasso:.2f}pp")
ax.set_title("a) Optimisation History (CV MAE per trial)")
ax.set_xlabel("Trial Number")
ax.set_ylabel("5-Fold CV MAE (pp)")
ax.legend(fontsize=9)
ax.text(0.98, 0.95, f"Best CV MAE: {study.best_value:.2f}pp\n(Trial {study.best_trial.number})",
        transform=ax.transAxes, ha="right", va="top", fontsize=9,
        bbox=dict(facecolor=PALETTE["light"], alpha=0.8, edgecolor="none"))

# 15b: parameter importance (via correlation with trial MAE)
ax = axes[0, 1]
param_cols = ["learning_rate", "max_depth", "subsample", "colsample_bytree",
              "n_estimators", "min_child_weight", "reg_alpha", "reg_lambda", "gamma"]
param_cols = [c for c in param_cols if c in trial_df.columns]
corrs = {p: abs(trial_df[p].corr(trial_df["cv_mae"])) for p in param_cols}
corrs_s = sorted(corrs.items(), key=lambda x: x[1], reverse=True)
names, vals = zip(*corrs_s)
bar_colors = [PALETTE["teal"] if v > 0.1 else PALETTE["gray"] for v in vals]
ax.barh(list(names)[::-1], list(vals)[::-1], color=bar_colors[::-1], edgecolor="white")
ax.set_title("b) Parameter Importance\n(|correlation| with CV MAE)")
ax.set_xlabel("|Pearson r| with MAE")
ax.axvline(0.1, color=PALETTE["amber"], lw=1.5, linestyle="--", label="r=0.10 threshold")
ax.legend(fontsize=9)

# 15c: model comparison bar chart
ax = axes[1, 0]
models_comp = ["Baseline\nXGBoost", "Tuned\nXGBoost", "Lasso\n(reference)"]
maes_comp   = [mae_baseline, mae_tuned, mae_lasso]
r2s_comp    = [r2_baseline, r2_tuned, r2_lasso]
bar_c = [PALETTE["gray"], PALETTE["teal"], PALETTE["amber"]]
bars = ax.bar(models_comp, maes_comp, color=bar_c, edgecolor="white", width=0.5)
for bar, val in zip(bars, maes_comp):
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.08,
            f"{val:.2f}pp", ha="center", va="bottom", fontweight="bold")
ax.set_title("c) Test MAE Comparison")
ax.set_ylabel("Test MAE (percentage points)")
improvement = (mae_baseline - mae_tuned) / mae_baseline * 100
ax.text(0.5, 0.95, f"Tuning improvement: {improvement:.1f}% vs baseline",
        transform=ax.transAxes, ha="center", va="top", fontsize=10,
        bbox=dict(facecolor=PALETTE["light"], alpha=0.8, edgecolor="none"))

# 15d: tuned model predicted vs actual
ax = axes[1, 1]
ax.scatter(y_test, pred_tuned, alpha=0.5, s=20, color=PALETTE["teal"])
lims = [min(y_test.min(), pred_tuned.min()) - 2, max(y_test.max(), pred_tuned.max()) + 2]
ax.plot(lims, lims, "r--", lw=1.5, label="Perfect prediction")
ax.set_title(f"d) Tuned XGBoost — Predicted vs Actual\nR² = {r2_tuned:.3f}  |  MAE = {mae_tuned:.2f}pp")
ax.set_xlabel("Actual Premium (%)")
ax.set_ylabel("Predicted Premium (%)")
ax.legend()

plt.tight_layout()
plt.savefig("results/figures/fig15_optuna_hpo.png", dpi=150, bbox_inches="tight")
plt.close()
print("✓ Figure 15 saved")

# ── Summary table ──────────────────────────────────────────────────────────────
summary = pd.DataFrame([
    {"model": "Baseline XGBoost",  "MAE": mae_baseline, "R2": r2_baseline,
     "HPO": "No", "CV": "None"},
    {"model": "Tuned XGBoost",     "MAE": mae_tuned,    "R2": r2_tuned,
     "HPO": "Optuna 100 trials", "CV": "TimeSeriesSplit(5)"},
    {"model": "Lasso (best)",      "MAE": mae_lasso,    "R2": r2_lasso,
     "HPO": "Manual", "CV": "Time-based"},
])
summary.to_csv("results/hpo_summary.csv", index=False)
print("✓ HPO summary saved")
print(f"\nConclusion: Tuned XGBoost achieves {improvement:.1f}% MAE reduction vs baseline")
print(f"Lasso still {'leads' if mae_lasso < mae_tuned else 'no longer leads'} — "
      f"gap is now {abs(mae_lasso - mae_tuned):.2f}pp")
