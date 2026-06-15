# Resume Bullets & Interview Preparation

## Resume Bullets (Pick 3–4 based on role)

### For Investment Banking Roles (JPMorgan, Goldman, Morgan Stanley)

> Built an ML framework to predict M&A acquisition premiums on 800 historical transactions (2000–2023), achieving MAE of 7.85pp and R² of 0.603 on a holdout test set using time-series-aware cross-validation to prevent look-ahead bias

> Engineered 65 features across 8 finance-grounded groups — deal structure, auction dynamics, target fundamentals, and macro conditions — informed by M&A research (Schwert 1996, Betton et al. 2008); SHAP analysis quantified that hostile bids add +6.6pp and PE buyers pay −2.3pp vs strategic acquirers

> Demonstrated that a regularized Lasso model with 28 selected features outperforms XGBoost (MAE 7.85 vs 9.71pp), consistent with linear premium formation in financial data; documented dataset methodology and survivorship bias limitations in 10-page technical report

### For Analytics Roles (Amex, Barclays, Capital One)

> Developed end-to-end ML pipeline for M&A premium prediction: data generation → cleaning (documented 7 cleaning decisions) → feature engineering (65 features) → model comparison (7 models) → SHAP explainability; full reproducible codebase on GitHub

> Applied Lasso for automatic feature selection across 65 financial features, retaining 28 significant predictors and achieving R² of 0.603; implemented winsorization, sector-median imputation, and one-hot encoding pipeline in Python/scikit-learn

> Designed time-based train/val/test split (2000–2016 / 2017–2019 / 2020–2023) to prevent temporal data leakage; generated SHAP beeswarm, dependence, and waterfall plots to quantify feature contributions at deal level

### For AI/ML Roles

> Built regression framework for acquisition premium prediction (n=800, 65 features) benchmarking 7 models from OLS baseline to gradient boosting; implemented SHAP TreeExplainer for XGBoost to generate global and local feature attributions

> Identified that Lasso (R²=0.603, MAE=7.85pp) outperforms XGBoost (R²=0.442) on structured financial tabular data with n=800, confirming gradient boosting's data hunger on small high-dimensional datasets; implemented full MLOps pipeline with model serialization via joblib

---

## Interview Q&A Preparation

### Investment Banking Interview

**Q: Walk me through your M&A premium project.**

"I built a machine learning model to predict acquisition premiums using deal structure, company fundamentals, and market conditions. The project covers 800 transactions from 2000 to 2023. My best model — a Lasso regression — achieves a mean absolute error of about 7.9 percentage points and explains 60% of the variation in premiums on a holdout test set.

The finance insights are what I find most interesting. SHAP analysis shows hostile bids add about 6-7 percentage points to premiums — which makes intuitive sense because the acquirer can't negotiate privately and has to overpay to overcome board resistance. PE buyers pay about 2-3 points less than strategic acquirers, consistent with the literature showing private equity is constrained by leverage and doesn't get credit for operational synergies. And high EV/EBITDA targets attract lower premiums — expensive stocks give acquirers less headroom to pay up and still create value."

**Q: Why does your model only explain 60% of premium variation?**

"That's actually strong for this problem domain. The irreducible noise in premium prediction is high — you have undisclosed negotiations, board dynamics, regulatory uncertainty, and personal relationships between executives that never show up in data. Academic papers using SDC Platinum with thousands of observations and more granular data typically get R² of 0.20–0.45. An R² of 0.60 suggests the observable features genuinely capture most of the systematic variation, and the residual is largely idiosyncratic deal noise."

**Q: What's survivorship bias in your dataset?**

"Only completed deals are in the data. Hostile bids that were rejected, deals blocked by regulators, and acquisitions withdrawn due to financing issues aren't captured. This is a real limitation — failed bids may have had higher premiums on average, which could bias my hostile bid coefficient downward. Addressing this would require pulling SC TO-T filings from EDGAR for all announced deals regardless of outcome."

**Q: Why did you choose the 1-week undisturbed price?**

"It's the standard in academic research and matches what M&A advisors actually use in fairness opinions. The 1-day window is too narrow — information leakage and anticipatory trading mean the stock often already moved before the announcement. The 4-week window captures more complete pre-announcement run-up but also includes deals where rumors were already public. One week is the standard industry compromise."

---

### Analytics Interview

**Q: Why did Lasso outperform XGBoost?**

"Three reasons. First, sample size — 550 training observations with 65 features is a low signal-to-noise environment for trees. XGBoost needs more data to reliably learn non-linear interactions. Second, the underlying data-generating process appears largely linear and additive — acquisition premiums are driven by factors that combine roughly additively rather than through complex interactions. Third, Lasso's L1 penalty performs automatic feature selection, dropping 37 of 65 features as noise, which reduces variance dramatically."

**Q: How did you prevent data leakage?**

"Two ways. First, time-based splits — I trained on 2000–2016, validated on 2017–2019, and locked the test set at 2020–2023. Random k-fold would let 2022 deals train on 2010 validation data, which is backward-looking from a deployment standpoint. Second, all features are computed from information available before the announcement date — trailing fundamentals, pre-announcement prices, concurrent macro data."

**Q: How would you deploy this model in a real bank?**

"As a deal-assessment tool in the M&A advisory workflow. When a banker is pitching an acquisition to a client, they'd input deal parameters and target fundamentals and receive a model-implied premium range — say 28% to 45% — alongside the SHAP decomposition showing which factors are driving the estimate. It doesn't replace comparable transaction analysis but supplements it with a data-driven anchor. Importantly, the Lasso coefficients are interpretable — a banker can read them as 'each additional bidder adds X basis points to the expected premium.'"

---

### Data Science / ML Interview

**Q: Why SHAP over feature importance from the model directly?**

"Tree-based feature importance from XGBoost is biased toward high-cardinality and continuous features — it measures how often a feature is split on, not how much it contributes to predictions. SHAP gives exact Shapley values grounded in cooperative game theory, satisfying efficiency (values sum to the prediction − baseline), symmetry, and dummy properties. More practically, SHAP gives both global importance AND per-deal attribution, so I can explain why a specific deal was predicted at 45% rather than just which features matter globally."

**Q: How would you improve this project given more time?**

"Four directions. First, genuine deal data from EDGAR merger proxies — scraping DEF 14A filings would give actual premiums, board deliberation details, and advisor opinions for US public targets. Second, failed bid data to address survivorship bias. Third, incorporating NLP features from deal announcement text — press release sentiment and specific synergy language are predictive in academic work. Fourth, a causal inference framing — using difference-in-differences around deal announcements to estimate treatment effects rather than predictive correlations."
