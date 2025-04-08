import pandas as pd
import numpy as np
import os

# Config
input_dir = "analysis_output"
output_dir = "3_5_outputs"
output_file = os.path.join(output_dir, "decile_returns_all_ga_metrics.csv")
os.makedirs(output_dir, exist_ok=True)

ga_metrics = [
    "goodwill_to_sales_lagged",
    "goodwill_to_equity_lagged",
    "goodwill_to_market_cap_lagged"
]

all_dfs = []

for metric in ga_metrics:
    file = f"decile_assignments_{metric}_all.csv"
    path = os.path.join(input_dir, file)

    if not os.path.exists(path):
        print(f"❌ File not found: {file}")
        continue

    df = pd.read_csv(path, parse_dates=["crsp_date"])
    df = df[df["decile"].between(1, 10) & df["ret"].notna()].copy()
    df["ME"] = df["prc"].abs() * df["csho"]

    # Equal-weighted return
    ew = df.groupby(["crsp_date", "decile"]).agg(
        ret_ew=("ret", "mean"),
        num_firms=("permno", "nunique")
    ).reset_index()

    # Value-weighted return
    vw = df.groupby(["crsp_date", "decile"]).apply(
        lambda x: np.average(x["ret"], weights=x["ME"]) if x["ME"].sum() > 0 else np.nan
    ).reset_index(name="ret_vw")

    # Combine
    merged = pd.merge(ew, vw, on=["crsp_date", "decile"])
    merged["ga_metric"] = metric
    merged.rename(columns={"crsp_date": "date"}, inplace=True)

    all_dfs.append(merged)

# Combine all metrics
if all_dfs:
    result_df = pd.concat(all_dfs)
    result_df.sort_values(["ga_metric", "date", "decile"], inplace=True)
    result_df.to_csv(output_file, index=False)
    print(f"✅ Saved combined decile returns to: {output_file}")
else:
    print("⚠️ No decile files found or matched. Nothing was saved.")
