"""
M&A Premium Predictor — REST API
===================================
FastAPI endpoint that accepts deal parameters and returns:
  - Predicted acquisition premium (%)
  - Prediction confidence interval (±1 MAE)
  - SHAP-based explanation of top drivers
  - Finance interpretation of the prediction

Run:  uvicorn src.api.app:app --reload --port 8000
Test: python src/api/test_api.py
Docs: http://localhost:8000/docs  (auto-generated Swagger UI)
"""

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from typing import Optional, List
import numpy as np
import pandas as pd
import joblib, json, shap, os, sys
import xgboost as xgb

# ── Setup ─────────────────────────────────────────────────────────────────────
BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
sys.path.insert(0, BASE_DIR)

MODEL_PATH   = os.path.join(BASE_DIR, "results/models/best_model.pkl")
SCALER_PATH  = os.path.join(BASE_DIR, "results/models/scaler.pkl")
FEAT_PATH    = os.path.join(BASE_DIR, "results/models/feature_list.json")
XGB_PATH     = os.path.join(BASE_DIR, "results/models/xgb_tuned.pkl")

lasso_model  = joblib.load(MODEL_PATH)
scaler       = joblib.load(SCALER_PATH)
xgb_model    = joblib.load(XGB_PATH)
with open(FEAT_PATH) as f:
    FEATURES = json.load(f)

# Pre-load training medians for default imputation
train_df      = pd.read_csv(os.path.join(BASE_DIR, "data/processed/ma_deals_features.csv"))
TRAIN_MEDIANS = train_df[FEATURES].apply(pd.to_numeric, errors="coerce").median()
MODEL_MAE     = 7.85  # from test set evaluation

# Pre-compute SHAP explainer
xgb_explainer = shap.TreeExplainer(xgb_model)

app = FastAPI(
    title="M&A Acquisition Premium Predictor",
    description=(
        "Predicts acquisition premiums for M&A transactions using ML. "
        "Built as part of IIT Madras QF research project. "
        "Model: Lasso Regression (MAE=7.85pp, R²=0.603 on 2020-2023 holdout). "
        "SHAP explanations via XGBoost TreeExplainer."
    ),
    version="1.0.0",
)


# ── Request / Response schemas ────────────────────────────────────────────────
class DealRequest(BaseModel):
    # Deal structure (required)
    deal_type: str = Field(..., description="'cash', 'mixed', or 'stock'",
                           json_schema_extra={"example": "cash"})
    acquirer_type: str = Field(..., description="'strategic' or 'financial'",
                               json_schema_extra={"example": "strategic"})

    # Deal structure (optional with defaults)
    hostile: int = Field(0, ge=0, le=1, description="1=hostile bid, 0=friendly")
    num_bidders: int = Field(1, ge=1, le=5, description="Number of competing bidders")
    cross_border: int = Field(0, ge=0, le=1, description="1=cross-border deal")
    tender_offer: int = Field(0, ge=0, le=1, description="1=tender offer structure")
    deal_val_bn: float = Field(5.0, gt=0, description="Deal value in USD billions")

    # Target fundamentals (optional — defaults to dataset medians)
    ev_ebitda: Optional[float] = Field(None, description="Target EV/EBITDA multiple")
    revenue_growth: Optional[float] = Field(None, description="Revenue growth YoY (e.g. 0.15 = 15%)")
    op_margin: Optional[float] = Field(None, description="Operating margin (e.g. 0.20 = 20%)")
    net_debt_ebitda: Optional[float] = Field(None, description="Net Debt / EBITDA leverage ratio")
    pb_ratio: Optional[float] = Field(None, description="Price-to-Book ratio")
    market_cap_bn: Optional[float] = Field(None, description="Target market cap in USD billions")
    ret_6m: Optional[float] = Field(None, description="6-month price return % (e.g. -20 = -20%)")
    pct_from_52w_high: Optional[float] = Field(None, description="% from 52-week high (negative)")

    # Macro conditions (optional)
    vix_at_ann: Optional[float] = Field(None, description="VIX level at announcement (e.g. 18)")
    treasury_10y_at_ann: Optional[float] = Field(None, description="10Y Treasury yield % at announcement")
    market_cycle: Optional[int] = Field(None, description="-1=bear, 0=neutral, 1=bull")

    # Sector
    sector: Optional[str] = Field("Technology", description="GICS sector of target company")

    class Config:
        json_schema_extra = {
            "example": {
                "deal_type": "cash",
                "acquirer_type": "strategic",
                "hostile": 0,
                "num_bidders": 1,
                "cross_border": 0,
                "tender_offer": 0,
                "deal_val_bn": 13.7,
                "ev_ebitda": 22.0,
                "revenue_growth": 0.02,
                "op_margin": 0.04,
                "net_debt_ebitda": -0.5,
                "pb_ratio": 2.1,
                "market_cap_bn": 9.8,
                "ret_6m": -14.0,
                "pct_from_52w_high": -25.0,
                "vix_at_ann": 10.8,
                "treasury_10y_at_ann": 2.15,
                "market_cycle": 1,
                "sector": "Consumer Staples"
            }
        }


class SHAPFactor(BaseModel):
    feature: str
    shap_value: float
    feature_value: float
    direction: str        # "increases" or "decreases"
    finance_note: str


class PredictionResponse(BaseModel):
    predicted_premium_pct: float
    confidence_interval_low: float
    confidence_interval_high: float
    model_mae_pp: float
    premium_range_label: str
    top_drivers: List[SHAPFactor]
    finance_summary: str
    comparable_deals: List[str]
    disclaimer: str


# ── Feature helpers ───────────────────────────────────────────────────────────
FINANCE_NOTES = {
    "auction_intensity":      "Competitive bidding pressure — more bidders/hostility drives premiums up",
    "deal_type_enc":          "Cash deals signal acquirer conviction and command higher premiums",
    "is_financial_buyer":     "PE/financial buyers pay less than strategic buyers (no synergy premium)",
    "hostile":                "Hostile bids bypass private negotiation — acquirer must overpay to win",
    "ev_ebitda":              "High EV/EBITDA target is already richly valued — less headroom for premium",
    "net_debt_ebitda":        "High leverage constrains deal structure and acquirer's willingness to pay",
    "revenue_growth":         "Fast-growing targets attract higher bids — acquirers pay for future earnings",
    "market_cycle":           "Bull markets support higher premiums; bear markets compress them",
    "vix_at_ann":             "High VIX (uncertainty) leads acquirers to bid more conservatively",
    "pct_from_52w_high":      "Beaten-down stocks attract higher premiums — acquirers perceive undervaluation",
    "momentum_reversal":      "Recent price decline signals potential undervaluation in acquirer's view",
    "tender_offer":           "Tender offers are faster and more certain, supporting higher premiums",
    "quality_score":          "High-quality businesses (strong margins/returns) command premium prices",
}

SECTORS = ["Technology","Healthcare","Financials","Energy","Industrials",
           "Consumer Discretionary","Materials","Communication Services",
           "Consumer Staples","Real Estate","Utilities"]

COMPARABLE_DEALS = {
    "cash+strategic+low_hostile":  ["Amazon/Whole Foods (27%)", "Oracle/Cerner (37%)", "Microsoft/Nuance (26%)"],
    "cash+strategic+hostile":      ["Twitter/Elon Musk (38%)", "Allergan/AbbVie (44%)"],
    "cash+strategic+high":         ["IBM/Red Hat (63%)", "Pfizer/Seagen (50%)", "BMS/Celgene (54%)"],
    "mixed+strategic":             ["AT&T/Time Warner (36%)", "CVS/Aetna (40%)", "Broadcom/VMware (44%)"],
    "financial_buyer":             ["Nuance/Microsoft (26%)", "Oracle/Cerner (37%)"],
    "default":                     ["LinkedIn/Microsoft (50%)", "Bayer/Monsanto (44%)", "AbbVie/Allergan (44%)"],
}


def req_to_feature_vector(req: DealRequest) -> np.ndarray:
    """Convert API request to model feature vector."""
    deal_type_map = {"cash": 2, "mixed": 1, "stock": 0}
    deal_enc = deal_type_map.get(req.deal_type, 2)
    is_fin   = 1 if req.acquirer_type == "financial" else 0

    row = {f: float(TRAIN_MEDIANS[f]) for f in FEATURES}

    def fill(key, val):
        if key in row and val is not None:
            row[key] = float(val)

    # Direct feature mappings
    fill("deal_type_enc",       deal_enc)
    fill("is_financial_buyer",  is_fin)
    fill("hostile",             req.hostile)
    fill("num_bidders",         req.num_bidders)
    fill("cross_border",        req.cross_border)
    fill("tender_offer",        req.tender_offer)
    fill("deal_val_bn",         req.deal_val_bn)
    fill("log_deal_val",        np.log1p(req.deal_val_bn))
    fill("ev_ebitda",           req.ev_ebitda)
    fill("revenue_growth",      req.revenue_growth)
    fill("op_margin",           req.op_margin)
    fill("net_debt_ebitda",     req.net_debt_ebitda)
    fill("pb_ratio",            req.pb_ratio)
    fill("market_cap_bn",       req.market_cap_bn)
    fill("ret_6m",              req.ret_6m)
    fill("pct_from_52w_high",   req.pct_from_52w_high)
    fill("vix_at_ann",          req.vix_at_ann)
    fill("treasury_10y_at_ann", req.treasury_10y_at_ann)
    fill("market_cycle",        req.market_cycle)

    # Sector dummies
    for sc in [c for c in FEATURES if c.startswith("sec_")]:
        row[sc] = 1.0 if sc == f"sec_{req.sector}" else 0.0

    # Engineered features
    nd_ebitda = row["net_debt_ebitda"]
    ev_eb     = row["ev_ebitda"]
    ret6m     = row.get("ret_6m", -2.0)
    pe_r      = row.get("pe_ratio", 25.0)
    rev_g     = row.get("revenue_growth", 0.08)

    row["auction_intensity"]       = req.hostile + req.num_bidders - 1
    row["deal_certainty_score"]    = int(deal_enc == 2) * 2 + req.tender_offer
    row["pe_leverage_interaction"] = is_fin * max(nd_ebitda, 0)
    row["valuation_discount"]      = 1 / max(ev_eb, 1)
    row["log_pb"]                  = np.log1p(max(row["pb_ratio"], 0.1))
    row["peg_proxy"]               = min(pe_r / max(rev_g * 100, 0.01), 50)
    row["quality_score"]           = (np.clip(row.get("roe",0.1),-1,1)*0.4 +
                                       np.clip(row.get("roa",0.04),-0.5,0.5)*0.4 +
                                       np.clip(row.get("op_margin",0.12),-0.5,0.5)*0.2)
    row["distress_flag"]           = int(nd_ebitda > 5 or
                                          row.get("current_ratio",1.8) < 0.8 or
                                          row.get("op_margin",0.12) < -0.1)
    row["momentum_reversal"]       = np.clip(-ret6m, -60, 60)
    row["momentum_divergence"]     = row.get("ret_1m",-1.0) - row.get("ret_12m",-5.0)
    row["funding_cost_proxy"]      = row["treasury_10y_at_ann"]
    row["macro_risk"]              = row["vix_at_ann"] / 20
    row["ma_heat"]                 = row.get("sector_ma_activity", 0.7) * (row.get("market_cycle",0) + 2)
    row["relative_deal_size"]      = min(np.log1p(req.deal_val_bn) /
                                          np.log1p(max(row["market_cap_bn"], 0.1)), 5)
    row["analyst_bullish"]         = int(row.get("analyst_rating",2.5) < 2.5 and
                                          row.get("analyst_upside",15) > 15)
    row["analyst_upside_clipped"]  = np.clip(row.get("analyst_upside", 15), -30, 80)
    row["is_crisis_year"]          = 0
    row["is_wave_year"]            = 0

    return np.array([float(row.get(f, 0.0)) for f in FEATURES])


def get_premium_label(prem: float) -> str:
    if prem < 20:   return "Low (below typical range)"
    elif prem < 30: return "Below average (15th–35th percentile)"
    elif prem < 40: return "Average (35th–65th percentile)"
    elif prem < 50: return "Above average (65th–85th percentile)"
    elif prem < 65: return "High (85th–95th percentile)"
    else:           return "Very high (top 5% — contested/hostile territory)"


def get_comparables(req: DealRequest, pred: float) -> List[str]:
    if req.acquirer_type == "financial":
        return COMPARABLE_DEALS["financial_buyer"]
    if req.hostile == 1:
        return COMPARABLE_DEALS["cash+strategic+hostile"]
    if req.deal_type == "mixed":
        return COMPARABLE_DEALS["mixed+strategic"]
    if pred > 50:
        return COMPARABLE_DEALS["cash+strategic+high"]
    if req.deal_type == "cash":
        return COMPARABLE_DEALS["cash+strategic+low_hostile"]
    return COMPARABLE_DEALS["default"]


def build_finance_summary(req: DealRequest, pred: float, drivers: List[SHAPFactor]) -> str:
    top = drivers[0].feature.replace("_", " ")
    direction = "above" if pred > 37.9 else "below"
    deal_desc = f"{'hostile' if req.hostile else 'friendly'} {req.deal_type} {req.acquirer_type}"
    return (
        f"Model predicts a {pred:.1f}% premium for this {deal_desc} deal — "
        f"{direction} the dataset average of 37.9%. "
        f"The dominant driver is {top} (SHAP: {drivers[0].shap_value:+.1f}pp). "
        f"{'Cash consideration and strategic buyer status support a higher premium.' if req.deal_type=='cash' and req.acquirer_type=='strategic' else ''}"
        f"{'The hostile bid nature significantly inflates the required premium.' if req.hostile else ''}"
        f"{'PE/financial buyer structure suppresses the premium vs comparable strategic deals.' if req.acquirer_type=='financial' else ''}"
    ).strip()


# ── API Endpoints ─────────────────────────────────────────────────────────────
@app.get("/", tags=["Health"])
def root():
    return {
        "service": "M&A Acquisition Premium Predictor",
        "version": "1.0.0",
        "model":   "Lasso Regression (primary) + XGBoost (SHAP explanations)",
        "metrics": {"test_mae_pp": MODEL_MAE, "test_r2": 0.603},
        "endpoints": ["/predict", "/health", "/docs"]
    }


@app.get("/health", tags=["Health"])
def health():
    return {"status": "ok", "model_loaded": True, "features": len(FEATURES)}


@app.post("/predict", response_model=PredictionResponse, tags=["Prediction"])
def predict(req: DealRequest):
    try:
        X_raw = req_to_feature_vector(req)
        if np.any(np.isnan(X_raw)):
            X_raw = np.nan_to_num(X_raw, nan=0.0)

        # Primary prediction: Lasso
        X_scaled   = scaler.transform(X_raw.reshape(1, -1))
        prediction = float(lasso_model.predict(X_scaled)[0])
        prediction = round(np.clip(prediction, 5, 120), 2)
        ci_low     = round(max(prediction - MODEL_MAE, 0), 1)
        ci_high    = round(prediction + MODEL_MAE, 1)

        # SHAP: XGBoost for explainability
        X_df       = pd.DataFrame([X_raw], columns=FEATURES)
        shap_vals  = xgb_explainer.shap_values(X_df)[0]

        # Top 5 SHAP drivers
        top_idx    = np.argsort(np.abs(shap_vals))[::-1][:5]
        top_drivers = []
        for i in top_idx:
            feat = FEATURES[i]
            sv   = float(shap_vals[i])
            fv   = float(X_raw[i])
            top_drivers.append(SHAPFactor(
                feature      = feat,
                shap_value   = round(sv, 3),
                feature_value= round(fv, 3),
                direction    = "increases" if sv > 0 else "decreases",
                finance_note = FINANCE_NOTES.get(feat,
                    f"{feat.replace('_',' ').title()} impacts premium formation")
            ))

        finance_summary = build_finance_summary(req, prediction, top_drivers)
        comparables     = get_comparables(req, prediction)

        return PredictionResponse(
            predicted_premium_pct   = prediction,
            confidence_interval_low = ci_low,
            confidence_interval_high= ci_high,
            model_mae_pp            = MODEL_MAE,
            premium_range_label     = get_premium_label(prediction),
            top_drivers             = top_drivers,
            finance_summary         = finance_summary,
            comparable_deals        = comparables,
            disclaimer              = (
                "Model trained on 800 M&A transactions (2000-2023). "
                "Predictions should be used alongside comparable transaction analysis, "
                "not as a standalone valuation tool. MAE ≈ 7.9pp on holdout data."
            )
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/feature-importance", tags=["Model Info"])
def feature_importance():
    """Returns the top 20 most important features from SHAP global analysis."""
    try:
        fi = pd.read_csv(os.path.join(BASE_DIR, "results/shap/feature_importance.csv"))
        top = fi.head(20).to_dict(orient="records")
        return {"top_features": top, "total_features": len(fi)}
    except Exception:
        return {"error": "Feature importance file not found. Run shap_analysis.py first."}


@app.get("/model-info", tags=["Model Info"])
def model_info():
    return {
        "primary_model":    "Lasso Regression (alpha=0.5)",
        "shap_model":       "XGBoost (Optuna-tuned, 100 trials)",
        "training_period":  "2000-2016",
        "validation":       "2017-2019",
        "test_period":      "2020-2023",
        "n_features":       len(FEATURES),
        "lasso_selected":   28,
        "test_mae_pp":      7.85,
        "test_rmse_pp":     10.05,
        "test_r2":          0.603,
        "within_5pp_pct":   40.6,
        "target_variable":  "premium_1w (1-week undisturbed price basis)",
        "real_validation":  {"n_deals": 15, "mae_pp": 14.3, "within_10pp_pct": 46.7},
        "cv_strategy":      "TimeSeriesSplit(n_splits=5) — no look-ahead bias",
    }
