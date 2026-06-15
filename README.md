# M&A Acquisition Premium Predictor

> **A machine learning framework for estimating fair acquisition premiums in M&A transactions using deal structure, target fundamentals, and macroeconomic conditions.**

[![Python](https://img.shields.io/badge/Python-3.12-blue.svg)](https://python.org)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

---

## Overview

Acquisition premiums are central to M&A advisory, fairness opinions, and shareholder value analysis. This project builds a data-driven framework to predict premiums using historical M&A patterns — quantifying the impact of deal structure, target valuation, and market conditions.

**Best model:** Lasso Regression | **Test MAE:** 7.85pp | **Test R²:** 0.603

---

## Key Findings

| Factor | Premium Impact | Source |
|---|---|---|
| Hostile bid | +6.6pp | vs. friendly deal |
| Cash consideration | +2.9pp | vs. stock/mixed |
| Beaten-down stock | +2.6pp | target −20% 6-month return |
| Financial (PE) buyer | −2.3pp | vs. strategic acquirer |
| High EV/EBITDA target | −2.4pp | per unit above mean |
| Auction (multi-bidder) | +1.4pp | per additional bidder |

---

## Project Structure

```
ma_premium_predictor/
├── data/
│   ├── raw/           # Generated dataset (800 M&A transactions)
│   └── processed/     # Cleaned and feature-engineered dataset
├── src/
│   ├── data/
│   │   ├── generate_data.py      # Empirically-grounded dataset generation
│   │   └── clean_data.py         # Cleaning pipeline with documented decisions
│   ├── features/
│   │   └── feature_engineering.py # 65 features across 8 finance groups
│   ├── models/
│   │   ├── train_models.py        # 7 models: Linear → XGBoost
│   │   └── shap_analysis.py       # SHAP explainability + finance insights
│   └── eda.py                     # 6 EDA figures
├── results/
│   ├── figures/       # 13 publication-quality charts
│   ├── models/        # Serialized best model + scaler
│   ├── shap/          # SHAP values + feature importance
│   └── model_comparison.csv
├── reports/
│   └── final_report.md            # Full methodology and findings
└── docs/
    └── data_cleaning_log.json     # Every cleaning decision documented
```

---

## Methodology

### Data
- 800 M&A transactions, 2000–2023, simulated from empirical distributions in published research
- Premium calibrated to Schwert (1996): mean ~38%, measured 1-week prior to announcement
- Three undisturbed price windows: 1-day, 1-week, 4-week

### Feature Engineering (65 features)
- **Deal structure:** consideration type, acquirer type, hostility, bidder count, tender offer flag
- **Auction dynamics:** `auction_intensity` composite, `deal_certainty_score`
- **Target fundamentals:** EV/EBITDA, leverage, margins, ROIC, growth
- **Macro conditions:** VIX regime, treasury yield, market cycle
- **Price momentum:** 1w/1m/3m/6m/12m returns, 52-week distance

### Modeling
- **Time-based splits** (no look-ahead bias): Train 2000–2016, Val 2017–2019, Test 2020–2023
- 7 models from linear baseline to gradient boosting
- Lasso outperforms trees — consistent with linear premium formation and small n

### Explainability
- SHAP TreeExplainer on XGBoost for exact attribution values
- Beeswarm, dependence plots, and waterfall plots (3 individual deal explanations)
- Finance insights quantified from SHAP deltas

---

## Results

| Model | Test MAE | Test R² | Within ±5pp |
|---|---|---|---|
| **Lasso (α=0.5)** | **7.85pp** | **0.603** | **40.6%** |
| ElasticNet | 8.29pp | 0.561 | 38.3% |
| Ridge | 8.44pp | 0.547 | 37.6% |
| Linear Regression | 8.54pp | 0.538 | 36.8% |
| LightGBM | 9.74pp | 0.448 | 33.1% |
| XGBoost | 9.71pp | 0.442 | 32.3% |
| Random Forest | 10.15pp | 0.348 | 29.3% |

---

## Quick Start

```bash
git clone https://github.com/[username]/ma-premium-predictor
cd ma-premium-predictor
pip install -r requirements.txt

# Generate dataset
python src/data/generate_data.py

# Clean data
python src/data/clean_data.py

# Feature engineering
python src/features/feature_engineering.py

# EDA
python src/eda.py

# Train all models
python src/models/train_models.py

# SHAP analysis
python src/models/shap_analysis.py
```

---

## Requirements

```
pandas>=2.0
numpy>=1.24
scikit-learn>=1.3
xgboost>=2.0
lightgbm>=4.0
shap>=0.44
matplotlib>=3.7
seaborn>=0.12
scipy>=1.11
```

---

## References

- Betton, Eckbo & Thorburn (2008). "Corporate Takeovers." *Handbook of Corporate Finance*
- Schwert, G.W. (1996). "Markup Pricing in Mergers and Acquisitions." *JFE*
- Officer, M.S. (2003). "Termination fees in mergers and acquisitions." *JFE*
- Bargeron et al. (2008). "Why do private acquirers pay so little compared to public acquirers?" *JFE*

---

*IIT Madras — Engineering Design & Quantitative Finance*
