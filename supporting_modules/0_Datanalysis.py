# General analysis of processed, raw, and output data

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
import glob

# Set output directory
output_dir = "analysis_output"
os.makedirs(output_dir, exist_ok=True)

########################################
### 1. Data Analysis of Compustat File #
########################################

# Load Compustat data
compustat = pd.read_csv('data/compustat_20250317.csv', parse_dates=['date'])

# Ensure columns are numeric
compustat['gdwl'] = pd.to_numeric(compustat['gdwl'], errors='coerce')
compustat['gvkey'] = pd.to_numeric(compustat['gvkey'], errors='coerce')

# Basic statistics
total_firms = compustat['gvkey'].nunique()
firms_with_goodwill = compustat.loc[compustat['gdwl'] > 0, 'gvkey'].nunique()
firms_with_gdwl_na = compustat.loc[compustat['gdwl'].isna(), 'gvkey'].nunique()

# Output summary
print(f"🏢 Total unique firms in Compustat: {total_firms}")
print(f"💰 Firms with at least one non-zero goodwill: {firms_with_goodwill}")
print(f"❓ Firms with missing goodwill data (NaN): {firms_with_gdwl_na}")
print(f"📊 Percentage of firms with goodwill: {firms_with_goodwill / total_firms:.2%}")
print(f"📊 Percentage of firms with missing goodwill: {firms_with_gdwl_na / total_firms:.2%}")

######################################
### 2. Analysis of Processed Data ####
######################################

# Path to processed data
file_path = 'data/processed_data.csv'
chunk_size = 500_000

# Initialize counters and storage
total_rows, unique_gvkeys, unique_permnos, ga_values, returns, years = 0, set(), set(), [], [], set()

# Process in chunks
for i, chunk in enumerate(pd.read_csv(file_path, chunksize=chunk_size)):
    print(f"Processing chunk {i + 1}")
    total_rows += len(chunk)
    unique_gvkeys.update(chunk['gvkey'].unique())
    unique_permnos.update(chunk['permno'].unique())
    ga_values.extend(chunk['ga'].dropna())
    returns.extend(chunk['ret'].dropna())
    years.update(pd.to_datetime(chunk['crsp_date']).dt.year.unique())

# Summary
print("\n===== SUMMARY =====")
print(f"Total rows: {total_rows}")
print(f"Unique gvkeys (firms): {len(unique_gvkeys)}")
print(f"Unique permnos (stocks): {len(unique_permnos)}")
print(f"Years covered: {sorted(years)}")
print("\nGA Summary:\n", pd.Series(ga_values).describe(percentiles=[0.01, 0.05, 0.25, 0.5, 0.75, 0.95, 0.99]))
print("\nReturn Summary:\n", pd.Series(returns).describe(percentiles=[0.01, 0.05, 0.25, 0.5, 0.75, 0.95, 0.99]))

############################################
### 3. Load and Analyze Raw Extracted Data #
############################################

# Helper to get latest file version
def load_latest_file(pattern):
    files = sorted(glob.glob(pattern))
    return files[-1] if files else None

# Load raw files
compustat_file = load_latest_file('data/compustat_*.csv')
crsp_ret_file = load_latest_file('data/crsp_ret_*.csv')
crsp_delist_file = load_latest_file('data/crsp_delist_*.csv')
crsp_compustat_file = load_latest_file('data/crsp_compustat_*.csv')

print("\n🗂️ Loaded Raw Files:")
print(f"Compustat: {compustat_file}\nCRSP Returns: {crsp_ret_file}\nCRSP Delist: {crsp_delist_file}\nCRSP Compustat Link: {crsp_compustat_file}")

# Quick summary function
def quick_summary(df, name):
    print(f"\n📊 Summary of {name}:")
    print(f"Rows: {df.shape[0]}, Columns: {df.shape[1]}")
    print(df.dtypes)
    if 'date' in df.columns:
        print("Date Range:", df['date'].min(), "to", df['date'].max())
    elif 'dlstdt' in df.columns:
        print("Date Range (Delisting):", df['dlstdt'].min(), "to", df['dlstdt'].max())
    print("Missing values summary:\n", df.isna().sum())


compustat_raw = pd.read_csv(compustat_file, parse_dates=['date'])
crsp_ret_raw = pd.read_csv(crsp_ret_file, parse_dates=['date'])
crsp_delist_raw = pd.read_csv(crsp_delist_file, parse_dates=['dlstdt'])
crsp_compustat_raw = pd.read_csv(crsp_compustat_file, parse_dates=['linkdt', 'linkenddt'])

quick_summary(compustat_raw, "Compustat")
quick_summary(crsp_ret_raw, "CRSP Returns")
quick_summary(crsp_delist_raw, "CRSP Delist")
quick_summary(crsp_compustat_raw, "CRSP-Compustat Link")

############################################
### 4. Processed Data Overview #############
############################################

df = pd.read_csv(file_path, parse_dates=['crsp_date'])
print("\n🎯 Processed Data Overview:")
print(df['ga'].describe())
print("Missing GA:", df['ga'].isna().sum())
print("Zero GA rows:", (df['ga'] == 0).sum())
print("Unique companies:", df['gvkey'].nunique())

############################################
### 5. GA and Firm Insights ################
############################################

# GA Zero vs Non-zero
ga_zero_nonzero = df.groupby(df['crsp_date'].dt.year).agg(
    total_obs=('ga', 'count'),
    ga_zero=('ga', lambda x: (x == 0).sum()),
    ga_nonzero=('ga', lambda x: (x != 0).sum())
).reset_index()
ga_zero_nonzero.to_excel(f"{output_dir}/GA_zero_vs_nonzero_per_year.xlsx", index=False)

# GA Percentiles
percentiles = df.groupby(df['crsp_date'].dt.year)['ga'].quantile([0, 0.25, 0.5, 0.75, 1]).unstack()
percentiles.columns = ['P0', 'P25', 'P50', 'P75', 'P100']
percentiles.to_excel(f"{output_dir}/GA_percentiles_per_year.xlsx", index=True)

# Firms per month
firms_per_month = df.groupby(df['crsp_date'].dt.to_period('M'))['gvkey'].nunique().reset_index()
firms_per_month.columns = ['Month', 'Unique Firms']
firms_per_month.to_excel(f"{output_dir}/firms_per_month.xlsx", index=False)

############################################
### 6. Plots and Distributions #############
############################################

# Mean and median GA over time
ga_trend = df.groupby(df['crsp_date'].dt.year).agg(mean_ga=('ga', 'mean'), median_ga=('ga', 'median')).reset_index()
sns.lineplot(data=ga_trend, x='crsp_date', y='mean_ga', label='Mean GA')
sns.lineplot(data=ga_trend, x='crsp_date', y='median_ga', label='Median GA')
plt.legend(); plt.title('Mean and Median GA over time')
plt.savefig(f"{output_dir}/GA_mean_median_trend.png"); plt.close()

# Histograms
sns.histplot(df['ga'], bins=50, kde=True).figure.savefig(f"{output_dir}/ga_distribution.png"); plt.close()

############################################
### 7. GA Factor Summary ###################
############################################

##################################
### GA Factor Return Plots ###
##################################

# Load GA factor files — NOTE: These are monthly files, not annual!
ga_eq = pd.read_csv('output/factors/ga_factor_returns_monthly_equal.csv', parse_dates=['date'])
ga_val = pd.read_csv('output/factors/ga_factor_returns_monthly_value.csv', parse_dates=['date'])

# Plot GA factor over time
plt.figure(figsize=(10, 6))
plt.plot(ga_eq['date'], ga_eq['ga_factor'], label='Equal-weighted')
plt.plot(ga_val['date'], ga_val['ga_factor'], label='Value-weighted')
plt.axhline(0, color='gray', linestyle='--')
plt.title('GA Factor (High - Low) Monthly Returns over Time')
plt.xlabel('Date')
plt.ylabel('Monthly Return')
plt.legend()
plt.tight_layout()
plt.savefig(f"{output_dir}/GA_factor_comparison.png")
plt.close()

# Summary stats GA factor
ga_summary = pd.DataFrame({
    'Equal Mean GA': [ga_eq['ga_factor'].mean()],
    'Equal Std GA': [ga_eq['ga_factor'].std()],
    'Value Mean GA': [ga_val['ga_factor'].mean()],
    'Value Std GA': [ga_val['ga_factor'].std()]
})
ga_summary.to_excel(f"{output_dir}/GA_factor_summary.xlsx", index=False)
print("✅ GA factor plots and summary saved.")
