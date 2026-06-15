"""
Synthetic M&A Dataset Generator
================================
Generates a realistic M&A transaction dataset grounded in empirical distributions
from published academic research:

Key references used to calibrate distributions:
- Betton, Eckbo & Thorburn (2008): "Corporate Takeovers" - premium distributions
- Schwert (1996): "Markup Pricing in Mergers & Acquisitions" - 30-40% avg premium
- Officer (2003): "Termination fees in mergers & acquisitions" - deal characteristics
- Bargeron et al. (2008): PE vs strategic buyer premiums
- Moeller, Schlingemann & Stulz (2004): Deal size and acquirer returns

Empirical facts embedded in the simulation:
  - Mean acquisition premium ≈ 30-37% (1-week basis)
  - Cash deals: ~5pp higher premium than stock deals
  - Hostile bids: ~15pp higher than friendly
  - Financial buyers: ~8pp lower than strategic buyers
  - Higher VIX → lower premiums
  - Beaten-down stocks (low 52w performance) → higher premiums
  - Multiple bidders → significant premium inflation
  - EV/EBITDA: higher priced targets → lower incremental premium
"""

import numpy as np
import pandas as pd
from scipy import stats
import os

np.random.seed(42)

N = 800  # number of simulated deals

# ── 1. Deal Categorical Features ───────────────────────────────────────────────
def generate_deal_features(n):
    raw_p = np.array([2,2,3,3,4,5,5,6,3,5,6,5,6,5,7,7,8,7,6,5,6,5,4,4], dtype=float)
    years = np.random.choice(range(2000, 2024), n, p=raw_p / raw_p.sum())

    # Slight skew: more deals in bull markets (2004-07, 2014-19, 2021)
    months       = np.random.choice(range(1, 13), n)

    deal_type    = np.random.choice(
        ["cash", "stock", "mixed"], n,
        p=[0.52, 0.22, 0.26]        # Cash deals dominate post-2008
    )
    acq_type     = np.random.choice(
        ["strategic", "financial"], n,
        p=[0.72, 0.28]
    )
    hostile      = np.random.choice([0, 1], n, p=[0.91, 0.09])
    num_bidders  = np.where(
        hostile == 1,
        np.random.choice([1, 2, 3], n, p=[0.4, 0.35, 0.25]),
        np.random.choice([1, 2, 3], n, p=[0.70, 0.22, 0.08])
    )
    cross_border = np.random.choice([0, 1], n, p=[0.75, 0.25])
    tender_offer = np.where(
        (deal_type == "cash") | (hostile == 1),
        np.random.choice([0, 1], n, p=[0.45, 0.55]),
        np.zeros(n, dtype=int)
    )

    sectors = [
        "Technology", "Healthcare", "Financials", "Energy",
        "Industrials", "Consumer Discretionary", "Materials",
        "Communication Services", "Consumer Staples", "Real Estate", "Utilities"
    ]
    sector = np.random.choice(sectors, n, p=[0.18, 0.16, 0.12, 0.09,
                                               0.10, 0.09, 0.06, 0.08,
                                               0.06, 0.04, 0.02])

    deal_val_bn = np.exp(np.random.normal(1.8, 1.3, n))  # log-normal deal sizes
    deal_val_bn = np.clip(deal_val_bn, 0.05, 250)

    return pd.DataFrame({
        "year": years,
        "month": months,
        "deal_type": deal_type,
        "acquirer_type": acq_type,
        "hostile": hostile,
        "num_bidders": num_bidders,
        "cross_border": cross_border,
        "tender_offer": tender_offer,
        "sector": sector,
        "deal_val_bn": np.round(deal_val_bn, 3),
    })


# ── 2. Target Fundamentals ─────────────────────────────────────────────────────
def generate_fundamentals(n, sector):
    # Market cap (log-normal, most targets mid-to-large cap)
    market_cap_bn = np.exp(np.random.normal(1.5, 1.4, n))
    market_cap_bn = np.clip(market_cap_bn, 0.05, 500)

    # EV/EBITDA varies by sector
    sector_ev_map = {
        "Technology": (18, 8), "Healthcare": (16, 7), "Financials": (11, 5),
        "Energy": (7, 4), "Industrials": (12, 5), "Consumer Discretionary": (13, 6),
        "Materials": (10, 4), "Communication Services": (14, 6),
        "Consumer Staples": (14, 5), "Real Estate": (20, 8), "Utilities": (12, 4)
    }
    ev_ebitda = np.array([
        max(1, np.random.normal(*sector_ev_map.get(s, (12, 5))))
        for s in sector
    ])

    # Revenue growth
    rev_growth = np.random.normal(0.08, 0.15, n)  # ~8% mean, high variance

    # Profitability
    op_margin    = np.clip(np.random.normal(0.12, 0.12, n), -0.3, 0.5)
    profit_margin= op_margin * np.random.uniform(0.5, 0.9, n)
    gross_margin = np.clip(op_margin + np.random.uniform(0.1, 0.3, n), 0, 0.9)

    # Leverage
    debt_equity  = np.abs(np.random.lognormal(0.5, 1.0, n))
    debt_equity  = np.clip(debt_equity, 0, 15)
    net_debt_ebitda = np.random.normal(1.8, 1.5, n)
    net_debt_ebitda = np.clip(net_debt_ebitda, -2, 10)

    # Returns
    roe = np.clip(np.random.normal(0.12, 0.15, n), -0.5, 0.6)
    roa = np.clip(np.random.normal(0.05, 0.07, n), -0.2, 0.3)

    # Valuation multiples
    pe_ratio = np.where(
        profit_margin > 0,
        np.clip(np.random.lognormal(2.8, 0.6, n), 5, 100),
        np.nan
    )
    pb_ratio = np.abs(np.random.lognormal(0.8, 0.7, n))
    pb_ratio = np.clip(pb_ratio, 0.2, 20)

    fcf_yield = np.random.normal(0.04, 0.04, n)  # free cash flow yield

    # Liquidity
    current_ratio = np.abs(np.random.normal(1.8, 0.8, n))
    current_ratio = np.clip(current_ratio, 0.3, 8)

    # Analyst coverage
    analyst_rating = np.random.uniform(1.5, 4.0, n)  # 1=Strong Buy, 5=Sell
    analyst_upside = np.random.normal(15, 20, n)      # % upside to target price

    return pd.DataFrame({
        "market_cap_bn":    np.round(market_cap_bn, 3),
        "ev_ebitda":        np.round(ev_ebitda, 2),
        "revenue_growth":   np.round(rev_growth, 4),
        "op_margin":        np.round(op_margin, 4),
        "profit_margin":    np.round(profit_margin, 4),
        "gross_margin":     np.round(gross_margin, 4),
        "debt_equity":      np.round(debt_equity, 3),
        "net_debt_ebitda":  np.round(net_debt_ebitda, 3),
        "roe":              np.round(roe, 4),
        "roa":              np.round(roa, 4),
        "pe_ratio":         np.round(pe_ratio, 2),
        "pb_ratio":         np.round(pb_ratio, 3),
        "fcf_yield":        np.round(fcf_yield, 4),
        "current_ratio":    np.round(current_ratio, 3),
        "analyst_rating":   np.round(analyst_rating, 2),
        "analyst_upside":   np.round(analyst_upside, 2),
    })


# ── 3. Market Conditions ───────────────────────────────────────────────────────
def generate_macro(n, years):
    # Approximate real values by year (based on historical data)
    vix_by_year = {
        2000: 23, 2001: 25, 2002: 28, 2003: 22, 2004: 15,
        2005: 13, 2006: 12, 2007: 17, 2008: 40, 2009: 31,
        2010: 22, 2011: 24, 2012: 18, 2013: 14, 2014: 14,
        2015: 16, 2016: 15, 2017: 11, 2018: 16, 2019: 14,
        2020: 29, 2021: 18, 2022: 25, 2023: 17
    }
    tr10y_by_year = {
        2000: 6.0, 2001: 5.0, 2002: 4.6, 2003: 4.0, 2004: 4.3,
        2005: 4.3, 2006: 4.8, 2007: 4.6, 2008: 3.7, 2009: 3.3,
        2010: 3.2, 2011: 2.8, 2012: 1.8, 2013: 2.4, 2014: 2.5,
        2015: 2.1, 2016: 1.8, 2017: 2.3, 2018: 2.9, 2019: 2.1,
        2020: 0.9, 2021: 1.4, 2022: 3.1, 2023: 4.2
    }
    sp500_by_year = {
        2000: 1450, 2001: 1200, 2002: 900, 2003: 1000, 2004: 1130,
        2005: 1250, 2006: 1380, 2007: 1500, 2008: 1000, 2009: 950,
        2010: 1200, 2011: 1250, 2012: 1400, 2013: 1700, 2014: 1950,
        2015: 2050, 2016: 2100, 2017: 2450, 2018: 2700, 2019: 2900,
        2020: 3300, 2021: 4300, 2022: 3900, 2023: 4200
    }

    vix   = np.array([vix_by_year.get(y, 18) + np.random.normal(0, 3)  for y in years])
    tr10y = np.array([tr10y_by_year.get(y, 3) + np.random.normal(0, 0.3) for y in years])
    sp500 = np.array([sp500_by_year.get(y, 2000) * np.random.uniform(0.9, 1.1) for y in years])

    # Market cycle: bull (2004-07, 2012-19, 2020-21), bear (2001-02, 2008-09, 2022)
    bear_years = {2001, 2002, 2008, 2009, 2022}
    bull_years = {2004, 2005, 2006, 2007, 2013, 2014, 2017, 2019, 2020, 2021}
    market_cycle = np.array([
        -1 if y in bear_years else (1 if y in bull_years else 0)
        for y in years
    ])

    # Sector M&A activity (deal heat in sector — proxy)
    sector_activity = np.random.uniform(0.2, 1.0, n)  # relative deal heat index

    return pd.DataFrame({
        "vix_at_ann":           np.round(np.clip(vix, 8, 80), 2),
        "treasury_10y_at_ann":  np.round(np.clip(tr10y, 0.1, 9), 3),
        "sp500_at_ann":         np.round(sp500, 0),
        "market_cycle":         market_cycle,
        "sector_ma_activity":   np.round(sector_activity, 3),
    })


# ── 4. Price Momentum ──────────────────────────────────────────────────────────
def generate_momentum(n):
    # Beaten-down stocks get higher premiums — model this correlation
    ret_12m = np.random.normal(-0.02, 0.35, n)  # avg slight underperformance before M&A
    ret_6m  = ret_12m * 0.5 + np.random.normal(0, 0.2, n)
    ret_3m  = ret_6m  * 0.4 + np.random.normal(0, 0.12, n)
    ret_1m  = ret_3m  * 0.3 + np.random.normal(0, 0.08, n)
    ret_1w  = ret_1m  * 0.2 + np.random.normal(0, 0.04, n)

    pct_from_52w_high = np.clip(-abs(np.random.normal(0.22, 0.18, n)), -0.80, 0.05)
    pct_from_52w_low  = np.abs(np.random.normal(0.30, 0.25, n))
    pct_from_52w_low  = np.clip(pct_from_52w_low, 0.01, 3.0)

    return pd.DataFrame({
        "ret_1w":            np.round(ret_1w * 100, 3),
        "ret_1m":            np.round(ret_1m * 100, 3),
        "ret_3m":            np.round(ret_3m * 100, 3),
        "ret_6m":            np.round(ret_6m * 100, 3),
        "ret_12m":           np.round(ret_12m * 100, 3),
        "pct_from_52w_high": np.round(pct_from_52w_high * 100, 3),
        "pct_from_52w_low":  np.round(pct_from_52w_low  * 100, 3),
    })


# ── 5. Premium Generation (the TARGET variable) ────────────────────────────────
def generate_premiums(df):
    """
    Generate acquisition premiums using a structural equation with noise.
    Coefficients calibrated from academic literature.
    
    Base premium: ~30% (Schwert 1996, Betton et al. 2008)
    
    Key effects:
      cash deal:        +5pp  (Officer 2003)
      stock deal:       -4pp  (Andrade et al. 2001)
      hostile:          +14pp (Schwert 2000)
      financial buyer:  -8pp  (Bargeron et al. 2008)
      each extra bidder:+6pp  (Boone & Mulherin 2007)
      cross-border:     +3pp  (Eckbo et al. 2018)
      tender offer:     +5pp  (Comment & Schwert 1995)
      EV/EBITDA (high): -0.5pp per unit (expensive target = less room)
      net leverage:     -1.5pp per unit of ND/EBITDA
      rev growth:       +8pp per unit (growing targets → higher bids)
      VIX (high):       -0.3pp per unit (uncertainty discount)
      market bull:      +3pp in bull, -4pp in bear
      beaten down (-20%)+6pp (acquirer sees value)
    """
    n = len(df)

    # Deal type effect
    deal_effect = np.where(df["deal_type"] == "cash", 5,
                  np.where(df["deal_type"] == "stock", -4, 1.5))

    # Acquirer type
    acq_effect  = np.where(df["acquirer_type"] == "financial", -8, 0)

    # Hostile
    hostile_eff = df["hostile"] * 14

    # Bidder competition
    bidder_eff  = (df["num_bidders"] - 1) * 6

    # Cross-border / tender
    cb_eff      = df["cross_border"] * 3
    to_eff      = df["tender_offer"] * 5

    # Fundamentals
    ev_eff      = np.clip(-0.5 * (df["ev_ebitda"] - 12), -8, 4)  # expensive → less premium
    lev_eff     = np.clip(-1.5 * df["net_debt_ebitda"], -12, 4)
    growth_eff  = np.clip(8 * df["revenue_growth"], -5, 15)
    fcf_eff     = np.clip(20 * df["fcf_yield"], -4, 10)

    # Market conditions
    vix_eff     = -0.3 * (df["vix_at_ann"] - 15)
    cycle_eff   = df["market_cycle"] * 3.5

    # Beaten-down stock
    beaten_eff  = np.clip(-0.15 * df["pct_from_52w_high"], 0, 15)

    # Size discount (large deals harder to close)
    size_eff    = np.clip(-1.5 * np.log1p(df["deal_val_bn"]) + 2, -6, 2)

    # Sector premiums (tech and healthcare command higher)
    sector_premium = {
        "Technology": 6, "Healthcare": 4, "Financials": -1,
        "Energy": -2, "Industrials": 0, "Consumer Discretionary": 1,
        "Materials": -1, "Communication Services": 3,
        "Consumer Staples": 2, "Real Estate": -3, "Utilities": -4
    }
    sec_eff = df["sector"].map(sector_premium).fillna(0)

    # Base + all effects + noise
    noise = np.random.normal(0, 10, n)  # irreducible noise (~10pp std)

    premium_1w = (
        30               # intercept (base premium)
        + deal_effect
        + acq_effect
        + hostile_eff
        + bidder_eff
        + cb_eff
        + to_eff
        + ev_eff
        + lev_eff
        + growth_eff
        + fcf_eff
        + vix_eff
        + cycle_eff
        + beaten_eff
        + size_eff
        + sec_eff
        + noise
    )

    # Winsorize: real premiums rarely below -10% or above 120%
    premium_1w = np.clip(premium_1w, -10, 120)

    # 1-day and 4-week premiums derived from 1-week with additional noise
    premium_1d = premium_1w - np.random.normal(2, 3, n)  # slightly lower (less run-up)
    premium_4w = premium_1w + np.random.normal(1, 4, n)  # slightly higher (more run-up)
    premium_1d = np.clip(premium_1d, -15, 125)
    premium_4w = np.clip(premium_4w, -8, 130)

    return (
        np.round(premium_1d, 2),
        np.round(premium_1w, 2),
        np.round(premium_4w, 2),
    )


# ── Main ───────────────────────────────────────────────────────────────────────
def generate_dataset(n=800, output_path="data/raw/ma_deals_raw.csv"):
    os.makedirs("data/raw", exist_ok=True)

    print(f"Generating {n} M&A deal records...")

    # Generate component dataframes
    deals  = generate_deal_features(n)
    fundm  = generate_fundamentals(n, deals["sector"].values)
    macro  = generate_macro(n, deals["year"].values)
    momt   = generate_momentum(n)

    # Combine
    df = pd.concat([deals, fundm, macro, momt], axis=1)

    # Add fake but realistic deal identifiers
    sectors = deals["sector"].values
    acq_suffixes = ["Inc", "Corp", "Ltd", "Group", "Holdings", "Partners", "Capital"]
    np.random.seed(42)
    df["deal_id"]  = [f"MA{2000+i:04d}" for i in range(n)]
    df["target_id"] = [f"TGT-{i:04d}" for i in range(n)]

    # Generate premiums (target variables)
    p1d, p1w, p4w = generate_premiums(df)
    df["premium_1d"] = p1d
    df["premium_1w"] = p1w
    df["premium_4w"] = p4w

    # Reorder columns nicely
    id_cols    = ["deal_id", "target_id", "year", "month", "sector"]
    deal_cols  = ["deal_type", "acquirer_type", "hostile", "num_bidders",
                  "cross_border", "tender_offer", "deal_val_bn"]
    fund_cols  = list(fundm.columns)
    macro_cols = list(macro.columns)
    momt_cols  = list(momt.columns)
    tgt_cols   = ["premium_1d", "premium_1w", "premium_4w"]

    df = df[id_cols + deal_cols + fund_cols + macro_cols + momt_cols + tgt_cols]

    df.to_csv(output_path, index=False)
    print(f"✓ Dataset saved: {output_path}")
    print(f"  Shape: {df.shape}")
    print(f"\n  Premium statistics (1-week basis):")
    print(f"  Mean:   {df['premium_1w'].mean():.1f}%")
    print(f"  Median: {df['premium_1w'].median():.1f}%")
    print(f"  Std:    {df['premium_1w'].std():.1f}%")
    print(f"  Min:    {df['premium_1w'].min():.1f}%")
    print(f"  Max:    {df['premium_1w'].max():.1f}%")
    print(f"\n  Null counts:\n{df.isnull().sum()[df.isnull().sum() > 0]}")
    return df


if __name__ == "__main__":
    import os
    os.chdir("C:/Users/ASUS/Downloads/ma_premium_predictor_v2/ma_premium_predictor")
    df = generate_dataset(n=800)
    print("\nSample records:")
    print(df[["year", "sector", "deal_type", "hostile", "ev_ebitda",
              "premium_1w"]].head(10).to_string())
