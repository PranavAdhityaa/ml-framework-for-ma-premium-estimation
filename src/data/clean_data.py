"""
Data Cleaning Pipeline
======================
Cleans the raw M&A dataset with documented, finance-grounded decisions.
Each cleaning step has a rationale comment.
"""

import pandas as pd
import numpy as np
import os
import json

def clean_dataset(
    input_path="data/raw/ma_deals_raw.csv",
    output_path="data/processed/ma_deals_clean.csv",
    log_path="docs/data_cleaning_log.json"
):
    os.makedirs("data/processed", exist_ok=True)
    os.makedirs("docs", exist_ok=True)

    df = pd.read_csv(input_path)
    log = {"original_shape": list(df.shape), "steps": []}
    print(f"Raw dataset: {df.shape[0]} rows × {df.shape[1]} cols")

    # ── Step 1: Drop exact duplicates ────────────────────────────────────────
    before = len(df)
    df = df.drop_duplicates(subset=["deal_id"])
    dropped = before - len(df)
    log["steps"].append({"step": "drop_duplicates", "dropped": dropped})
    print(f"Step 1 — Duplicates removed: {dropped}")

    # ── Step 2: Filter implausible premiums ─────────────────────────────────
    # Finance rationale: premiums below -10% suggest data error (acquirer
    # paying below market price without a distress flag). Premiums above 120%
    # are extremely rare outside bankruptcy/squeeze-out scenarios and likely
    # measurement error in a research dataset.
    before = len(df)
    df = df[(df["premium_1w"] >= -10) & (df["premium_1w"] <= 120)]
    dropped = before - len(df)
    log["steps"].append({"step": "filter_premium_bounds [-10, 120]", "dropped": dropped})
    print(f"Step 2 — Out-of-bounds premiums dropped: {dropped}")

    # ── Step 3: Drop rows with missing target variable ───────────────────────
    before = len(df)
    df = df.dropna(subset=["premium_1w"])
    dropped = before - len(df)
    log["steps"].append({"step": "drop_missing_premium", "dropped": dropped})
    print(f"Step 3 — Missing premium dropped: {dropped}")

    # ── Step 4: Handle missing PE ratio ─────────────────────────────────────
    # Finance rationale: PE is undefined for loss-making firms.
    # Strategy: impute with sector median. This preserves cross-sectional
    # variation better than global median imputation.
    sector_pe = df.groupby("sector")["pe_ratio"].median()
    df["pe_ratio_missing"] = df["pe_ratio"].isna().astype(int)  # indicator feature
    df["pe_ratio"] = df.groupby("sector")["pe_ratio"].transform(
        lambda x: x.fillna(x.median())
    )
    # Fallback for sectors with all-NA PE
    global_pe_median = df["pe_ratio"].median()
    df["pe_ratio"] = df["pe_ratio"].fillna(global_pe_median)
    log["steps"].append({"step": "impute_pe_ratio", "method": "sector_median"})
    print(f"Step 4 — PE ratio imputed with sector medians")

    # ── Step 5: Winsorize continuous features ────────────────────────────────
    # Finance rationale: Extreme leverage ratios (e.g., ND/EBITDA > 20x)
    # and negative EV/EBITDA values are typically data artefacts or corner
    # cases (distressed firms) that would distort model training.
    winsorize_cols = {
        "ev_ebitda":        (1, 50),
        "debt_equity":      (0, 15),
        "net_debt_ebitda":  (-3, 15),
        "pe_ratio":         (5, 80),
        "pb_ratio":         (0.1, 15),
        "rev_ebitda":       (None, None),  # skip
        "revenue_growth":   (-0.5, 1.5),
        "op_margin":        (-0.4, 0.6),
        "fcf_yield":        (-0.15, 0.25),
        "ret_12m":          (-80, 150),
        "ret_6m":           (-60, 100),
        "ret_3m":           (-40, 60),
        "deal_val_bn":      (0.01, 200),
    }
    for col, (lo, hi) in winsorize_cols.items():
        if col not in df.columns or lo is None:
            continue
        df[col] = df[col].clip(lower=lo, upper=hi)
    log["steps"].append({"step": "winsorize", "columns": list(winsorize_cols.keys())})
    print(f"Step 5 — Winsorized {len(winsorize_cols)} continuous features")

    # ── Step 6: Encode categorical features ─────────────────────────────────
    # deal_type: cash=2, mixed=1, stock=0 (ordinal by typical premium level)
    deal_map = {"cash": 2, "mixed": 1, "stock": 0}
    df["deal_type_enc"] = df["deal_type"].map(deal_map)

    # acquirer_type: binary
    df["is_financial_buyer"] = (df["acquirer_type"] == "financial").astype(int)

    # Sector dummies (drop 'Utilities' as baseline — lowest premiums)
    sector_dummies = pd.get_dummies(df["sector"], prefix="sec", drop_first=False)
    if "sec_Utilities" in sector_dummies.columns:
        sector_dummies = sector_dummies.drop(columns=["sec_Utilities"])
    df = pd.concat([df, sector_dummies], axis=1)

    log["steps"].append({"step": "encode_categoricals",
                         "deal_type": "ordinal", "sector": "one-hot (minus Utilities)"})
    print(f"Step 6 — Categoricals encoded ({len(sector_dummies.columns)} sector dummies)")

    # ── Step 7: Year-based train/val/test split flags ─────────────────────────
    # Finance rationale: Random splits would leak future deal information into
    # training, violating the look-ahead constraint. Time-based splits mimic
    # real deployment conditions.
    df["split"] = pd.cut(
        df["year"],
        bins=[1999, 2016, 2019, 2099],
        labels=["train", "val", "test"]
    )
    split_counts = df["split"].value_counts()
    log["steps"].append({"step": "time_based_split",
                         "train": "2000-2016", "val": "2017-2019", "test": "2020-2023"})
    print(f"Step 7 — Time splits: {dict(split_counts)}")

    # ── Step 8: Final check ───────────────────────────────────────────────────
    assert df["premium_1w"].isna().sum() == 0, "Target has NaN!"
    remaining_nulls = df.isnull().sum()
    remaining_nulls = remaining_nulls[remaining_nulls > 0]
    if len(remaining_nulls):
        print(f"  Remaining nulls:\n{remaining_nulls}")
        # Fill any remaining with column median
        for col in remaining_nulls.index:
            if df[col].dtype in [float, int, "float64", "int64"]:
                df[col] = df[col].fillna(df[col].median())

    log["final_shape"] = list(df.shape)
    log["null_counts_final"] = df.isnull().sum().to_dict()

    df.to_csv(output_path, index=False)
    with open(log_path, "w") as f:
        json.dump(log, f, indent=2)

    print(f"\n✓ Clean dataset: {df.shape[0]} rows × {df.shape[1]} cols → {output_path}")
    print(f"✓ Cleaning log  → {log_path}")
    return df


if __name__ == "__main__":
    import os
    os.chdir("C:/Users/ASUS/Downloads/ma_premium_predictor_v2/ma_premium_predictor")
    df = clean_dataset()
    print("\nFinal dtypes summary:")
    print(df.dtypes.value_counts())
    print("\nSplit distribution:")
    print(df["split"].value_counts())
