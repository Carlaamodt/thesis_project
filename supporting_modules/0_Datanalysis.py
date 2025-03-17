# General analysis of processed, raw, and output data

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
import glob

import pandas as pd

import pandas as pd

# Load Compustat data (adjust path if needed)
compustat = pd.read_csv('data/compustat_20250317.csv', parse_dates=['date'])

# Ensure 'gdwl' and 'gvkey' are numeric (if not already)
compustat['gdwl'] = pd.to_numeric(compustat['gdwl'], errors='coerce')
compustat['gvkey'] = pd.to_numeric(compustat['gvkey'], errors='coerce')

# Total unique firms in Compustat
total_firms = compustat['gvkey'].nunique()

# Firms that have at least one non-zero goodwill
firms_with_goodwill = compustat.loc[compustat['gdwl'] > 0, 'gvkey'].nunique()

# Firms that have goodwill reported as NaN (missing)
firms_with_gdwl_na = compustat.loc[compustat['gdwl'].isna(), 'gvkey'].nunique()

# Print results
print(f"🏢 Total unique firms in Compustat: {total_firms}")
print(f"💰 Firms with at least one non-zero goodwill: {firms_with_goodwill}")
print(f"❓ Firms with missing goodwill data (NaN): {firms_with_gdwl_na}")

# Optional: Share percentage
print(f"📊 Percentage of firms with goodwill: {firms_with_goodwill / total_firms:.2%}")
print(f"📊 Percentage of firms with missing goodwill: {firms_with_gdwl_na / total_firms:.2%}")



# Path to processed data
file_path = 'data/processed_data.csv'
chunk_size = 500_000

# Initialize counters and storage
total_rows = 0
unique_gvkeys = set()
unique_permnos = set()
ga_values = []
returns = []
years = set()

# Process in chunks
for i, chunk in enumerate(pd.read_csv(file_path, chunksize=chunk_size)):
    print(f"Processing chunk {i + 1}")

    total_rows += len(chunk)

    unique_gvkeys.update(chunk['gvkey'].unique())
    unique_permnos.update(chunk['permno'].unique())

    ga_values.extend(chunk['ga'].dropna().tolist())
    returns.extend(chunk['ret'].dropna().tolist())

    years.update(pd.to_datetime(chunk['crsp_date']).dt.year.unique())

# Results summary
print("\n===== SUMMARY =====")
print(f"Total rows: {total_rows}")
print(f"Unique gvkeys (firms): {len(unique_gvkeys)}")
print(f"Unique permnos (stocks): {len(unique_permnos)}")
print(f"Years covered: {sorted(years)}")

# Summary statistics for GA
ga_series = pd.Series(ga_values)
print("\nGA Summary:")
print(ga_series.describe(percentiles=[0.01, 0.05, 0.25, 0.5, 0.75, 0.95, 0.99]))

# Summary statistics for Returns
ret_series = pd.Series(returns)
print("\nReturn Summary:")
print(ret_series.describe(percentiles=[0.01, 0.05, 0.25, 0.5, 0.75, 0.95, 0.99]))

######Real analysis#########
# Set display options
pd.set_option('display.float_format', lambda x: '%.3f' % x)

# Create output directory
output_dir = "analysis_output"
os.makedirs(output_dir, exist_ok=True)

#############################
### Load Raw Extracted Data ###
#############################

# Find latest files based on timestamp
def load_latest_file(pattern):
    files = sorted(glob.glob(pattern))
    return files[-1] if files else None

compustat_file = load_latest_file('data/compustat_*.csv')
crsp_ret_file = load_latest_file('data/crsp_ret_*.csv')
crsp_delist_file = load_latest_file('data/crsp_delist_*.csv')
crsp_compustat_file = load_latest_file('data/crsp_compustat_*.csv')

print("🗂️ Loaded Raw Files:")
print(f"Compustat: {compustat_file}")
print(f"CRSP Returns: {crsp_ret_file}")
print(f"CRSP Delist: {crsp_delist_file}")
print(f"CRSP Compustat Link: {crsp_compustat_file}")

# Load raw data
compustat_raw = pd.read_csv(compustat_file, parse_dates=['date'])
crsp_ret_raw = pd.read_csv(crsp_ret_file, parse_dates=['date'])
crsp_delist_raw = pd.read_csv(crsp_delist_file, parse_dates=['dlstdt'])
crsp_compustat_raw = pd.read_csv(crsp_compustat_file, parse_dates=['linkdt', 'linkenddt'])

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

quick_summary(compustat_raw, "Compustat")
quick_summary(crsp_ret_raw, "CRSP Returns")
quick_summary(crsp_delist_raw, "CRSP Delist")
quick_summary(crsp_compustat_raw, "CRSP-Compustat Link")

########### Load Processed Data ######################
processed_file = load_latest_file('data/processed_data*.csv')
df = pd.read_csv(processed_file, parse_dates=['crsp_date'])
crsp_delist = crsp_delist_raw

############################################
### Basic Processed Data Overview ###
############################################

print("\n🎯 Processed Data Overview:")
print(df['ga'].describe())  # Corrected to GA
print("Missing GA:", df['ga'].isna().sum())
print("Zero GA rows:", (df['ga'] == 0).sum())
print("Unique companies:", df['gvkey'].nunique())

##################################
### GA Zero vs. Non-Zero per Year ###
##################################

ga_zero_nonzero = df.groupby(df['crsp_date'].dt.year).agg(
    total_obs=('ga', 'count'),
    ga_zero=('ga', lambda x: (x == 0).sum()),
    ga_nonzero=('ga', lambda x: (x != 0).sum())
).reset_index()
ga_zero_nonzero.to_excel(f"{output_dir}/GA_zero_vs_nonzero_per_year.xlsx", index=False)

##################################
### GA Percentiles per Year ###
##################################

percentiles = df.groupby(df['crsp_date'].dt.year)['ga'].quantile([0, 0.25, 0.5, 0.75, 1]).unstack()
percentiles.columns = ['P0', 'P25', 'P50', 'P75', 'P100']
percentiles.to_excel(f"{output_dir}/GA_percentiles_per_year.xlsx", index=True)

##################################
### Number of Firms Per Month ###
##################################

firms_per_month = df.groupby(df['crsp_date'].dt.to_period('M'))['gvkey'].nunique().reset_index()
firms_per_month.columns = ['Month', 'Unique Firms']
firms_per_month.to_excel(f"{output_dir}/firms_per_month.xlsx", index=False)

##################################
### Median and Mean GA Trend ###
##################################

ga_trend = df.groupby(df['crsp_date'].dt.year).agg(
    mean_ga=('ga', 'mean'),
    median_ga=('ga', 'median')
).reset_index()
sns.lineplot(data=ga_trend, x='crsp_date', y='mean_ga', label='Mean GA')
sns.lineplot(data=ga_trend, x='crsp_date', y='median_ga', label='Median GA')
plt.legend()
plt.title('Mean and Median GA over time')
plt.savefig(f"{output_dir}/GA_mean_median_trend.png")
plt.close()

##################################
### Histograms ###
##################################

# Goodwill Distribution
sns.histplot(df['gdwl'], bins=50, kde=True).figure.savefig(f"{output_dir}/goodwill_distribution.png")
plt.close()

# GA Distribution (consider log-scale if very skewed)
sns.histplot(df['ga'], bins=50, kde=True).figure.savefig(f"{output_dir}/ga_distribution.png")
plt.close()

##################################
### GA Factor Return Plots ###
##################################

# Load GA factor files — CHECK that 'year' or 'date' exists!
ga_eq = pd.read_csv('data/ga_factor_returns_annual_equal.csv')
ga_val = pd.read_csv('data/ga_factor_returns_annual_value.csv')

# Plot GA factor (check if 'year' or 'date')
if 'year' in ga_eq.columns:
    time_col = 'year'
else:
    time_col = 'date'

plt.figure(figsize=(10, 6))
plt.plot(ga_eq[time_col], ga_eq['GA_factor'], label='Equal-weighted')
plt.plot(ga_val[time_col], ga_val['GA_factor'], label='Value-weighted')
plt.title('GA Factor (High - Low) Returns over Time')
plt.legend()
plt.savefig(f"{output_dir}/GA_factor_comparison.png")
plt.close()

# Summary stats GA factor
ga_summary = pd.DataFrame({
    'Equal Mean GA': [ga_eq['GA_factor'].mean()],
    'Equal Std GA': [ga_eq['GA_factor'].std()],
    'Value Mean GA': [ga_val['GA_factor'].mean()],
    'Value Std GA': [ga_val['GA_factor'].std()]
})
ga_summary.to_excel(f"{output_dir}/GA_factor_summary.xlsx", index=False)

##################################
### Delisting Statistics ###
##################################

merged_delist = pd.merge(df[['permno']].drop_duplicates(), crsp_delist, on='permno', how='left')
delisting_counts = merged_delist['dlstdt'].dt.year.value_counts().sort_index().reset_index()
delisting_counts.columns = ['year', 'num_delistings']
delisting_counts.to_excel(f"{output_dir}/delisting_counts.xlsx", index=False)

##################################
### Final Message ###
##################################
print("🎉 Complete data analysis finished! All outputs in 'analysis_output'.")