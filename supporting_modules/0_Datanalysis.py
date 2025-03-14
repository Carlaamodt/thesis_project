# General analysis of processed, raw, and output data

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
import glob

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
df = pd.read_csv(processed_file, parse_dates=['date'])
crsp_delist = crsp_delist_raw

############################################
### Basic Processed Data Overview ###
############################################

print("\n🎯 Processed Data Overview:")
print(df['goodwill_intensity'].describe())
print("Missing goodwill_intensity:", df['goodwill_intensity'].isna().sum())
print("Zero goodwill_intensity rows:", (df['goodwill_intensity'] == 0).sum())
print("Unique companies:", df['gvkey'].nunique())

##################################
### Goodwill and Firm Data Over Time ###
##################################

df['year'] = df['date'].dt.year
goodwill_summary = df.groupby('year').agg(
    total_gdwl=('gdwl', 'sum'),
    avg_gdwl=('gdwl', 'mean'),
    total_ga=('goodwill_intensity', 'sum'),
    avg_ga=('goodwill_intensity', 'mean'),
    total_firms=('gvkey', 'nunique'),
    firms_with_gdwl=('gdwl', lambda x: (x > 0).sum())
).reset_index()
goodwill_summary.to_excel(f"{output_dir}/goodwill_summary.xlsx", index=False)

##################################
### Goodwill Change 2003-2023 ###
##################################

goodwill_2003, goodwill_2023 = df[df['year'] == 2003]['gdwl'].sum(), df[df['year'] == 2023]['gdwl'].sum()
change_percent = (goodwill_2023 - goodwill_2003) / goodwill_2003 * 100
with open(f"{output_dir}/goodwill_change.txt", 'w') as f:
    f.write(f"2003 Goodwill: {goodwill_2003:.2f}\n2023 Goodwill: {goodwill_2023:.2f}\nChange: {change_percent:.2f}%\n")

##################################
### Delisting Statistics ###
##################################

merged_delist = pd.merge(df[['permno']].drop_duplicates(), crsp_delist, on='permno', how='left')
delisting_counts = merged_delist['dlstdt'].dt.year.value_counts().sort_index().reset_index()
delisting_counts.columns = ['year', 'num_delistings']
delisting_counts.to_excel(f"{output_dir}/delisting_counts.xlsx", index=False)

##################################
### Histograms ###
##################################

# Goodwill Distribution
sns.histplot(df['gdwl'], bins=50, kde=True).figure.savefig(f"{output_dir}/goodwill_distribution.png")
plt.close()

# Goodwill Intensity Distribution
sns.histplot(df['goodwill_intensity'], bins=50, kde=True).figure.savefig(f"{output_dir}/ga_distribution.png")
plt.close()

##################################
### Trend Plots ###
##################################

# Goodwill trend
sns.lineplot(data=goodwill_summary, x='year', y='total_gdwl').figure.savefig(f"{output_dir}/goodwill_trend.png")
plt.close()

# GA trend
sns.lineplot(data=goodwill_summary, x='year', y='avg_ga').figure.savefig(f"{output_dir}/ga_trend.png")
plt.close()

##################################
### Analysis on GA Factor Returns ###
##################################

# Load factor returns
ga_eq = pd.read_csv('data/ga_factor_returns_annual_equal.csv')
ga_val = pd.read_csv('data/ga_factor_returns_annual_value.csv')

# Plot GA Factor (Equal vs. Value)
plt.figure(figsize=(10, 6))
plt.plot(ga_eq['year'], ga_eq['GA_factor'], label='Equal-weighted')
plt.plot(ga_val['year'], ga_val['GA_factor'], label='Value-weighted')
plt.title('GA Factor (High - Low) Returns over Time')
plt.legend()
plt.savefig(f"{output_dir}/GA_factor_comparison.png")
plt.close()

# Summary stats GA Factor
ga_summary = pd.DataFrame({
    'Equal Mean GA': [ga_eq['GA_factor'].mean()],
    'Equal Std GA': [ga_eq['GA_factor'].std()],
    'Value Mean GA': [ga_val['GA_factor'].mean()],
    'Value Std GA': [ga_val['GA_factor'].std()]
})
ga_summary.to_excel(f"{output_dir}/GA_factor_summary.xlsx", index=False)

##################################
### Analysis on Regression Results ###
##################################

# Load regression output if available
reg_output = load_latest_file('output/ga_factor_regression_results.xlsx')
if reg_output:
    print("\n📈 Found Regression Results — Consider Reviewing in Excel for detailed insights.")

##################################
### Final Message ###
##################################
print("🎉 Complete data analysis finished! All outputs in 'analysis_output'.")