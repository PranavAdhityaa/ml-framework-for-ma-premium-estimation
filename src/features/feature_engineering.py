"""
Feature Engineering Pipeline
==============================
Creates derived financial ratios, interaction terms, and transformed features.
Every feature has a documented finance rationale.
"""

import pandas as pd
import numpy as np
import os


def engineer_features(
    input_path="data/processed/ma_deals_clean.csv",
    output_path="data/processed/ma_deals_features.csv"
):
    os.makedirs("data/processed", exist_ok=True)
    df = pd.read_csv(input_path)
    print(f"Input: {df.shape}")

    # ── Group 1: Deal Complexity / Risk ──────────────────────────────────────
    # Rationale: A deal that is both hostile AND has multiple bidders represents
    # a genuine auction with maximum competitive pressure → highest premiums.
    df["auction_intensity"] = df["hostile"] + df["num_bidders"] - 1
    # Rationale: Cash + tender offer = fastest, most credible deal structure
    df["deal_certainty_score"] = (
        (df["deal_type_enc"] == 2).astype(int) * 2 +
        df["tender_offer"].astype(int)
    )
    # Log deal size: captures diminishing premium effect for very large deals
    df["log_deal_val"] = np.log1p(df["deal_val_bn"])
    # Interaction: financial buyer × leverage (PE loves leveraged targets)
    df["pe_leverage_interaction"] = df["is_financial_buyer"] * df["net_debt_ebitda"].clip(lower=0)

    # ── Group 2: Valuation Signals ────────────────────────────────────────────
    # Rationale: A target trading at a large discount to intrinsic value
    # (low P/B, low EV/EBITDA relative to sector) offers the acquirer more
    # "headroom" to pay a higher premium and still create value.
    df["valuation_discount"] = 1 / (df["ev_ebitda"].clip(lower=1))  # higher = cheaper target
    df["log_pb"] = np.log1p(df["pb_ratio"].clip(lower=0.1))
    # Price-to-earnings normalised by growth (PEG-style)
    df["peg_proxy"] = df["pe_ratio"] / (df["revenue_growth"].clip(lower=0.01) * 100)
    df["peg_proxy"] = df["peg_proxy"].clip(upper=50)

    # ── Group 3: Financial Quality / Health ──────────────────────────────────
    # Rationale: Acquirers pay premiums for quality. High ROIC firms generate
    # more synergies. High leverage deters premium because the acquirer
    # inherits debt service obligations.
    df["quality_score"] = (
        df["roe"].clip(-1, 1) * 0.4 +
        df["roa"].clip(-0.5, 0.5) * 0.4 +
        df["op_margin"].clip(-0.5, 0.5) * 0.2
    )
    df["distress_flag"] = (
        (df["net_debt_ebitda"] > 5) |
        (df["current_ratio"]   < 0.8) |
        (df["op_margin"]       < -0.1)
    ).astype(int)

    # ── Group 4: Growth / Momentum ───────────────────────────────────────────
    # Rationale: Momentum reversal — beaten-down stocks get higher bids
    # as acquirers perceive temporary undervaluation. But long-term
    # underperformers may signal structural issues → nuanced effect.
    df["momentum_reversal"] = -df["ret_6m"]  # negative momentum → higher premium
    df["momentum_reversal"] = df["momentum_reversal"].clip(-60, 60)
    # Short-term vs long-term momentum divergence (negative = recent selloff)
    df["momentum_divergence"] = df["ret_1m"] - df["ret_12m"]

    # ── Group 5: Macro Environment ───────────────────────────────────────────
    # Rationale: High VIX + high credit spreads = acquirers demand higher
    # return hurdle. Also, high rates make debt-financed deals more expensive,
    # which pressures financial buyers to bid lower.
    # credit_spread not available (blocked API) — use treasury as proxy for funding cost
    df["funding_cost_proxy"] = df["treasury_10y_at_ann"]
    df["macro_risk"] = df["vix_at_ann"] / 20  # normalised VIX (1.0 = "normal")
    # Deal environment: is it a hot M&A year? (sector activity × market cycle)
    if "sector_ma_activity" in df.columns and "market_cycle" in df.columns:
        df["ma_heat"] = df["sector_ma_activity"] * (df["market_cycle"] + 2)
    else:
        df["ma_heat"] = 1.0

    # ── Group 6: Relative Size ────────────────────────────────────────────────
    # Rationale: A very small deal relative to acquirer has low integration
    # risk; a transformational deal (deal_val > acquirer market cap) is riskier
    # and may be priced lower to leave buffer.
    df["relative_deal_size"] = np.log1p(df["deal_val_bn"]) / np.log1p(df["market_cap_bn"])
    df["relative_deal_size"] = df["relative_deal_size"].clip(0, 5)

    # ── Group 7: Analyst Sentiment ───────────────────────────────────────────
    # Rationale: Analyst consensus buy rating with large upside means the
    # market already thinks the stock is undervalued — acquirer must pay up.
    df["analyst_bullish"] = (
        (df["analyst_rating"] < 2.5) &
        (df["analyst_upside"] > 15)
    ).astype(int)
    df["analyst_upside_clipped"] = df["analyst_upside"].clip(-30, 80)

    # ── Group 8: Year dummies for temporal effects ────────────────────────────
    # Rationale: Controls for M&A wave periods not captured by macro variables
    crisis_years = [2001, 2002, 2008, 2009, 2020]
    wave_years   = [2005, 2006, 2007, 2014, 2015, 2016, 2017, 2018, 2021]
    df["is_crisis_year"] = df["year"].isin(crisis_years).astype(int)
    df["is_wave_year"]   = df["year"].isin(wave_years).astype(int)

    print(f"Output: {df.shape}")
    df.to_csv(output_path, index=False)
    print(f"✓ Features saved → {output_path}")

    # Print feature summary
    new_features = [
        "auction_intensity", "deal_certainty_score", "log_deal_val",
        "pe_leverage_interaction", "valuation_discount", "log_pb", "peg_proxy",
        "quality_score", "distress_flag", "momentum_reversal",
        "momentum_divergence", "funding_cost_proxy", "macro_risk", "ma_heat",
        "relative_deal_size", "analyst_bullish", "analyst_upside_clipped",
        "is_crisis_year", "is_wave_year"
    ]
    print(f"\n{len(new_features)} engineered features added:")
    for f in new_features:
        print(f"  · {f}: {df[f].describe()['mean']:.3f} mean, "
              f"{df[f].describe()['std']:.3f} std")

    return df


def get_feature_columns(df):
    """Return the final list of model-ready numeric feature columns."""
    # Exclude identifiers, raw categoricals, target variables, split column
    exclude = {
        "deal_id", "target_id", "year", "month", "sector",
        "deal_type", "acquirer_type", "split",
        "premium_1d", "premium_1w", "premium_4w",
        "rev_ebitda",  # was marked skip
    }
    bool_cols  = df.select_dtypes(include="bool").columns.tolist()
    float_cols = df.select_dtypes(include=["float64", "int64"]).columns.tolist()
    all_num    = list(set(bool_cols + float_cols) - exclude)
    # Also exclude the 'pe_ratio_missing' indicator from modelling set?
    # No — keep it: it is informative (loss-making targets)
    all_num = sorted(all_num)
    return all_num


if __name__ == "__main__":
    import os
    os.chdir("C:/Users/ASUS/Downloads/ma_premium_predictor_v2/ma_premium_predictor")
    df = engineer_features()
    feat_cols = get_feature_columns(df)
    print(f"\nTotal model features: {len(feat_cols)}")
    print(feat_cols[:20])
