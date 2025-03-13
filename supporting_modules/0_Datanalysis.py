# General analysis of processed and raw data

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

# Load them
compustat_raw = pd.read_csv(compustat_file, parse_dates=['date'])
crsp_ret_raw = pd.read_csv(crsp_ret_file, parse_dates=['date'])
crsp_delist_raw = pd.read_csv(crsp_delist_file, parse_dates=['dlstdt'])
crsp_compustat_raw = pd.read_csv(crsp_compustat_file, parse_dates=['linkdt', 'linkenddt'])

# Quick Summary
def quick_summary(df, name):
    print(f"\n📊 Summary of {name}:")
    print(f"Rows: {df.shape[0]}, Columns: {df.shape[1]}")
    print(df.dtypes)
    
    # Dynamic date handling
    if 'date' in df.columns:
        print("Date Range:", df['date'].min(), "to", df['date'].max())
    elif 'dlstdt' in df.columns:
        print("Date Range (Delisting):", df['dlstdt'].min(), "to", df['dlstdt'].max())
    elif 'linkdt' in df.columns and 'linkenddt' in df.columns:
        print("Link Date Range:", df['linkdt'].min(), "to", df['linkenddt'].max())
    else:
        print("Date Range: N/A")
    
    print("Missing values summary:\n", df.isna().sum())

quick_summary(compustat_raw, "Compustat")
quick_summary(crsp_ret_raw, "CRSP Returns")
quick_summary(crsp_delist_raw, "CRSP Delist")
quick_summary(crsp_compustat_raw, "CRSP-Compustat Link")

########### Load processed data dynamically ######################
processed_file = load_latest_file('data/processed_data*.csv')
df = pd.read_csv(processed_file, parse_dates=['date'])
crsp_delist = crsp_delist_raw  # Use already loaded raw delist file

############################################
### Basic Overview and Missing/Zero Check ###
############################################

print(df['goodwill_intensity'].describe())
print("Missing values in goodwill_intensity:", df['goodwill_intensity'].isna().sum())
zero_count = df[df['goodwill_intensity'] == 0].shape[0]
print("Number of rows with goodwill_intensity = 0:", zero_count)
unique_companies = df['gvkey'].nunique()
print("Number of unique companies:", unique_companies)

##################################
### Yearly Goodwill and Firm Data ###
##################################

df['year'] = df['date'].dt.year

# Goodwill Aggregation
goodwill_summary = df.groupby('year').agg(
    total_gdwl=('gdwl', 'sum'),
    avg_gdwl=('gdwl', 'mean'),
    total_ga=('goodwill_intensity', 'sum'),
    avg_ga=('goodwill_intensity', 'mean'),
    total_firms=('gvkey', 'nunique'),
    firms_with_gdwl=('gdwl', lambda x: (x > 0).sum())
).reset_index()

goodwill_summary.to_excel(f"{output_dir}/goodwill_summary.xlsx", index=False)
print("✅ Goodwill and firm summary saved.")

##################################
### Sum & Avg Goodwill for Periods ###
##################################

period_summary = pd.DataFrame({
    'Period': ['2003-2010', '2017-2023'],
    'Total_Goodwill': [
        df[(df['year'] >= 2003) & (df['year'] <= 2010)]['gdwl'].sum(),
        df[(df['year'] >= 2017) & (df['year'] <= 2023)]['gdwl'].sum()
    ],
    'Average_Goodwill': [
        df[(df['year'] >= 2003) & (df['year'] <= 2010)]['gdwl'].mean(),
        df[(df['year'] >= 2017) & (df['year'] <= 2023)]['gdwl'].mean()
    ]
})
period_summary.to_excel(f"{output_dir}/goodwill_period_summary.xlsx", index=False)
print("✅ Period goodwill analysis saved.")

##################################
### Goodwill change from 2003 to 2023 ###
##################################

goodwill_2003 = df[df['year'] == 2003]['gdwl'].sum()
goodwill_2023 = df[df['year'] == 2023]['gdwl'].sum()
change_percent = (goodwill_2023 - goodwill_2003) / goodwill_2003 * 100

with open(f"{output_dir}/goodwill_change.txt", 'w') as f:
    f.write(f"Total goodwill in 2003: {goodwill_2003:.2f}\n")
    f.write(f"Total goodwill in 2023: {goodwill_2023:.2f}\n")
    f.write(f"Percentage change: {change_percent:.2f}%\n")

print("✅ Goodwill change from 2003 to 2023 saved.")

##################################
### Delisting Statistics ###
##################################

# Linking delisting with processed dataset
merged_delist = pd.merge(df[['permno']].drop_duplicates(), crsp_delist, on='permno', how='left')
delisting_counts = merged_delist['dlstdt'].dt.year.value_counts().sort_index().reset_index()
delisting_counts.columns = ['year', 'num_delistings']
delisting_counts.to_excel(f"{output_dir}/delisting_counts.xlsx", index=False)
print("✅ Delisting statistics saved.")

##################################
### Histograms / Distribution ###
##################################

# Goodwill Distribution
plt.figure(figsize=(8, 5))
sns.histplot(df['gdwl'], bins=50, kde=True)
plt.title('Distribution of Goodwill')
plt.xlabel('Goodwill')
plt.savefig(f"{output_dir}/goodwill_distribution.png")
plt.close()

# Goodwill Intensity Distribution
plt.figure(figsize=(8, 5))
sns.histplot(df['goodwill_intensity'], bins=50, kde=True)
plt.title('Distribution of Goodwill Intensity (GA)')
plt.xlabel('Goodwill Intensity')
plt.savefig(f"{output_dir}/ga_distribution.png")
plt.close()

print("✅ Histograms saved.")

##################################
### Trend plots over time ###
##################################

# Goodwill trend
plt.figure(figsize=(10, 6))
sns.lineplot(data=goodwill_summary, x='year', y='total_gdwl')
plt.title('Total Goodwill over Time')
plt.savefig(f"{output_dir}/goodwill_trend.png")
plt.close()

# GA trend
plt.figure(figsize=(10, 6))
sns.lineplot(data=goodwill_summary, x='year', y='avg_ga')
plt.title('Average Goodwill Intensity (GA) over Time')
plt.savefig(f"{output_dir}/ga_trend.png")
plt.close()

print("✅ Trend plots saved.")

##################################
### Final Message ###
##################################
print("🎉 General analysis completed! Ready to review the outputs in 'analysis_output' folder.")