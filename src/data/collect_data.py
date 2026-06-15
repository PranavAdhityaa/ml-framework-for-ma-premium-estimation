"""
Data Collection Pipeline for M&A Acquisition Premium Predictor
===============================================================
Collects:
  1. Premium computation from yfinance stock price history
  2. Target company fundamentals (TTM financials, ratios)
  3. Macro variables from FRED (via pandas_datareader)

All sources are free and public.
"""

import yfinance as yf
import pandas as pd
import numpy as np
import pandas_datareader.data as web
import datetime
import warnings
import time
import json
import os
import sys

warnings.filterwarnings("ignore")
sys.path.insert(0, "C:/Users/ASUS/Downloads/ma_premium_predictor_v2/ma_premium_predictor")
from src.data.deal_universe import DEALS

# ── FRED macro series ──────────────────────────────────────────────────────────
FRED_SERIES = {
    "vix":          "VIXCLS",          # CBOE Volatility Index
    "treasury_10y": "DGS10",           # 10-Year Treasury Rate
    "sp500":        "SP500",           # S&P 500 Level
    "credit_spread":"BAMLH0A0HYM2",   # HY-IG spread (risk appetite)
    "gdp_growth":   "A191RL1Q225SBEA",# Real GDP growth QoQ
    "fed_funds":    "FEDFUNDS",        # Federal Funds Rate
}

START_FRED = "2005-01-01"
END_FRED   = "2024-01-01"


# ── helpers ────────────────────────────────────────────────────────────────────
def safe_get(d: dict, *keys, default=np.nan):
    for k in keys:
        if isinstance(d, dict):
            d = d.get(k, default)
        else:
            return default
    return d if d is not None else default


def get_price_on_or_before(hist: pd.DataFrame, date: datetime.date, window: int = 10):
    """Return the closing price on `date` or the nearest prior trading day."""
    if hist.empty:
        return np.nan
    hist.index = pd.to_datetime(hist.index).date
    candidates = hist[hist.index <= date]
    if candidates.empty:
        candidates = hist[hist.index <= date + datetime.timedelta(days=window)]
    if candidates.empty:
        return np.nan
    return float(candidates["Close"].iloc[-1])


def compute_premium(hist, ann_date, days_back):
    """
    Premium = (first_price_after_ann - undisturbed_price) / undisturbed_price
    undisturbed_price = close `days_back` trading days before announcement
    """
    if hist.empty:
        return np.nan, np.nan

    hist.index = pd.to_datetime(hist.index).date
    ann_date   = pd.Timestamp(ann_date).date()

    # Price at announcement (day 0 or +1)
    fwd = hist[hist.index >= ann_date]
    if fwd.empty:
        return np.nan, np.nan
    price_ann = float(fwd["Close"].iloc[0])

    # Undisturbed price (days_back calendar days before – we roll to nearest trading day)
    lookback = ann_date - datetime.timedelta(days=days_back + 14)  # buffer
    pre      = hist[(hist.index >= lookback) & (hist.index < ann_date)]
    if len(pre) < days_back:
        return np.nan, np.nan
    price_undisturbed = float(pre["Close"].iloc[-days_back])

    if price_undisturbed <= 0:
        return np.nan, np.nan

    premium = (price_ann - price_undisturbed) / price_undisturbed * 100
    return round(premium, 4), round(price_undisturbed, 4)


def get_fundamentals(ticker_obj, info):
    """Extract fundamental features from yfinance info dict."""
    def g(*keys):
        for k in keys:
            v = info.get(k)
            if v is not None and not (isinstance(v, float) and np.isnan(v)):
                return v
        return np.nan

    mkt_cap      = g("marketCap")
    enterprise_v = g("enterpriseValue")
    ebitda       = g("ebitda")
    total_rev    = g("totalRevenue")
    total_debt   = g("totalDebt")
    cash         = g("totalCash")
    net_income   = g("netIncomeToCommon")
    shares       = g("sharesOutstanding")
    book_val     = g("bookValue")
    price        = g("previousClose", "regularMarketPreviousClose")
    beta         = g("beta")
    rec_mean     = g("recommendationMean")      # analyst rating 1=Strong Buy 5=Sell
    target_price = g("targetMeanPrice")
    rev_growth   = g("revenueGrowth")           # YoY
    gross_margin = g("grossMargins")
    op_margin    = g("operatingMargins")
    profit_margin= g("profitMargins")
    roe          = g("returnOnEquity")
    roa          = g("returnOnAssets")
    current_r    = g("currentRatio")
    quick_r      = g("quickRatio")
    de_ratio     = g("debtToEquity")
    pe_ratio     = g("trailingPE")
    pb_ratio     = g("priceToBook")
    ps_ratio     = g("priceToSalesTrailing12Months")
    ev_ebitda    = g("enterpriseToEbitda")
    ev_revenue   = g("enterpriseToRevenue")
    fcf          = g("freeCashflow")
    payout_r     = g("payoutRatio")
    sector       = g("sector")
    industry     = g("industry")

    # Derived
    net_debt = (total_debt - cash) if (total_debt is not np.nan and cash is not np.nan) else np.nan
    nd_ebitda = (net_debt / ebitda) if (net_debt is not np.nan and ebitda not in [np.nan, 0]) else np.nan
    fcf_yield = (fcf / mkt_cap * 100) if (fcf is not np.nan and mkt_cap not in [np.nan, 0]) else np.nan
    analyst_upside = ((target_price - price) / price * 100) if (
        target_price is not np.nan and price not in [np.nan, 0]) else np.nan

    return {
        "market_cap_bn":    round(mkt_cap / 1e9, 4) if mkt_cap is not np.nan else np.nan,
        "enterprise_val_bn":round(enterprise_v / 1e9, 4) if enterprise_v is not np.nan else np.nan,
        "ebitda_bn":        round(ebitda / 1e9, 4) if ebitda is not np.nan else np.nan,
        "revenue_bn":       round(total_rev / 1e9, 4) if total_rev is not np.nan else np.nan,
        "net_income_bn":    round(net_income / 1e9, 4) if net_income is not np.nan else np.nan,
        "ev_ebitda":        ev_ebitda,
        "ev_revenue":       ev_revenue,
        "pe_ratio":         pe_ratio,
        "pb_ratio":         pb_ratio,
        "ps_ratio":         ps_ratio,
        "beta":             beta,
        "revenue_growth":   rev_growth,
        "gross_margin":     gross_margin,
        "op_margin":        op_margin,
        "profit_margin":    profit_margin,
        "roe":              roe,
        "roa":              roa,
        "current_ratio":    current_r,
        "quick_ratio":      quick_r,
        "debt_equity":      de_ratio,
        "net_debt_ebitda":  nd_ebitda,
        "fcf_yield":        fcf_yield,
        "analyst_rating":   rec_mean,
        "analyst_upside":   analyst_upside,
        "payout_ratio":     payout_r,
        "sector":           sector,
        "industry":         industry,
    }


def get_price_momentum(hist, ann_date):
    """Stock momentum vs announcement date."""
    if hist.empty:
        return {}
    hist.index = pd.to_datetime(hist.index).date
    ann = pd.Timestamp(ann_date).date()

    def ret_window(days_back, days_end=1):
        end_d   = ann - datetime.timedelta(days=days_end)
        start_d = ann - datetime.timedelta(days=days_back + 14)
        window  = hist[(hist.index >= start_d) & (hist.index <= end_d)]
        if len(window) < 2:
            return np.nan
        return (window["Close"].iloc[-1] / window["Close"].iloc[0] - 1) * 100

    def dist_from_52w():
        start_52 = ann - datetime.timedelta(days=365)
        w52 = hist[(hist.index >= start_52) & (hist.index < ann)]
        if w52.empty:
            return np.nan, np.nan
        high = w52["Close"].max()
        low  = w52["Close"].min()
        last = w52["Close"].iloc[-1]
        pct_from_high = (last - high) / high * 100
        pct_from_low  = (last - low)  / low  * 100
        return pct_from_high, pct_from_low

    high_dist, low_dist = dist_from_52w()

    return {
        "ret_1w":       ret_window(7),
        "ret_1m":       ret_window(30),
        "ret_3m":       ret_window(90),
        "ret_6m":       ret_window(180),
        "ret_12m":      ret_window(365),
        "pct_from_52w_high": high_dist,
        "pct_from_52w_low":  low_dist,
    }


# ── FRED macro data ────────────────────────────────────────────────────────────
def load_fred_data():
    print("Downloading FRED macro data...")
    start = pd.Timestamp(START_FRED)
    end   = pd.Timestamp(END_FRED)
    frames = {}
    for name, series_id in FRED_SERIES.items():
        try:
            df = web.DataReader(series_id, "fred", start, end)
            df.columns = [name]
            frames[name] = df
            print(f"  ✓ {name} ({series_id})")
        except Exception as e:
            print(f"  ✗ {name}: {e}")
    if frames:
        macro = pd.concat(frames.values(), axis=1)
        macro = macro.ffill().bfill()
        return macro
    return pd.DataFrame()


def get_macro_at_date(macro_df, ann_date):
    if macro_df.empty:
        return {}
    date = pd.Timestamp(ann_date)
    row  = macro_df[macro_df.index <= date]
    if row.empty:
        return {}
    row = row.iloc[-1]
    return {
        "vix_at_ann":          row.get("vix", np.nan),
        "treasury_10y_at_ann": row.get("treasury_10y", np.nan),
        "sp500_at_ann":        row.get("sp500", np.nan),
        "credit_spread_at_ann":row.get("credit_spread", np.nan),
        "gdp_growth_at_ann":   row.get("gdp_growth", np.nan),
        "fed_funds_at_ann":    row.get("fed_funds", np.nan),
    }


# ── Main collection loop ───────────────────────────────────────────────────────
def collect_all(output_path="data/raw/ma_deals_raw.csv", checkpoint_path="data/raw/checkpoint.json"):
    os.makedirs("data/raw", exist_ok=True)

    # Load checkpoint if exists
    if os.path.exists(checkpoint_path):
        with open(checkpoint_path) as f:
            done = set(json.load(f))
        print(f"Resuming from checkpoint. {len(done)} already collected.")
    else:
        done = set()

    records = []
    if os.path.exists(output_path):
        existing = pd.read_csv(output_path)
        records  = existing.to_dict("records")

    # Load FRED macro
    macro_df = load_fred_data()

    total = len(DEALS)
    for i, deal in enumerate(DEALS):
        ticker, acquirer, ann_date, deal_val_bn, deal_type, acq_type, hostile = deal
        key = f"{ticker}_{ann_date}"

        if key in done:
            continue

        print(f"\n[{i+1}/{total}] {ticker} | {acquirer} | {ann_date}")

        try:
            ann_dt = pd.Timestamp(ann_date)
            fetch_start = (ann_dt - pd.DateOffset(years=2)).strftime("%Y-%m-%d")
            fetch_end   = (ann_dt + pd.DateOffset(days=10)).strftime("%Y-%m-%d")

            tk   = yf.Ticker(ticker)
            hist = tk.history(start=fetch_start, end=fetch_end, auto_adjust=True)
            info = tk.info

            if hist.empty:
                print(f"  ✗ No price history for {ticker}")
                done.add(key)
                continue

            # Premiums at 3 windows
            prem_1d,  price_1d  = compute_premium(hist, ann_date, 1)
            prem_1w,  price_1w  = compute_premium(hist, ann_date, 5)
            prem_4w,  price_4w  = compute_premium(hist, ann_date, 20)

            if all(np.isnan(p) for p in [prem_1d, prem_1w, prem_4w]):
                print(f"  ✗ Could not compute premium for {ticker}")
                done.add(key)
                continue

            # Fundamentals
            fundam = get_fundamentals(tk, info)

            # Price momentum
            momentum = get_price_momentum(hist, ann_date)

            # Macro
            macro_row = get_macro_at_date(macro_df, ann_date)

            # Assemble record
            record = {
                "ticker":         ticker,
                "acquirer":       acquirer,
                "ann_date":       ann_date,
                "deal_val_bn":    deal_val_bn,
                "deal_type":      deal_type,       # cash / stock / mixed
                "acquirer_type":  acq_type,        # strategic / financial
                "hostile":        int(hostile),
                "premium_1d":     prem_1d,
                "premium_1w":     prem_1w,
                "premium_4w":     prem_4w,
                "undisturbed_price_1w": price_1w,
                "year":           ann_dt.year,
                "month":          ann_dt.month,
            }
            record.update(fundam)
            record.update(momentum)
            record.update(macro_row)

            records.append(record)
            done.add(key)

            # Save checkpoint every 5 records
            if len(records) % 5 == 0:
                pd.DataFrame(records).to_csv(output_path, index=False)
                with open(checkpoint_path, "w") as f:
                    json.dump(list(done), f)
                print(f"  ✓ Checkpoint saved ({len(records)} records)")

            time.sleep(0.3)  # be polite to yfinance

        except Exception as e:
            print(f"  ✗ Error on {ticker}: {e}")
            done.add(key)
            continue

    # Final save
    df = pd.DataFrame(records)
    df.to_csv(output_path, index=False)
    with open(checkpoint_path, "w") as f:
        json.dump(list(done), f)

    print(f"\n{'='*60}")
    print(f"Collection complete. {len(df)} records saved to {output_path}")
    print(f"{'='*60}")
    return df


if __name__ == "__main__":
    os.chdir("C:/Users/ASUS/Downloads/ma_premium_predictor_v2/ma_premium_predictor")
    df = collect_all()
    print(df.shape)
    print(df.head())
