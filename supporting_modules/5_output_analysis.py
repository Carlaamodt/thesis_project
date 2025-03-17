import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import os

# Set output directory
output_dir = "analysis_output"
os.makedirs(output_dir, exist_ok=True)

# Load decile data
print("📥 Loading decile assignment data...")
decile_data = pd.read_csv("analysis_output/decile_assignments.csv", parse_dates=['crsp_date'])

# Filter for valid deciles
decile_data = decile_data[decile_data['decile'].between(1, 10)]

#############################
### 1. Firms per Decile and Goodwill Breakdown
#############################

print("🔢 Counting firms per decile and goodwill status...")
decile_goodwill_counts = decile_data.groupby(['FF_year', 'decile', 'has_goodwill_firm']).agg(
    num_firms=('permno', 'nunique')
).reset_index()

decile_goodwill_pivot = decile_goodwill_counts.pivot_table(
    index=['FF_year', 'decile'],
    columns='has_goodwill_firm',
    values='num_firms',
    fill_value=0
).reset_index()

# Rename columns based on actual pivot output
col_map = {0: 'firms_without_goodwill', 1: 'firms_with_goodwill'}
decile_goodwill_pivot.rename(columns=col_map, inplace=True)

# Add total firms column
decile_goodwill_pivot['total_firms'] = (
    decile_goodwill_pivot.get('firms_with_goodwill', 0) +
    decile_goodwill_pivot.get('firms_without_goodwill', 0)
)

decile_goodwill_pivot.to_excel(f'{output_dir}/firms_with_goodwill_per_decile_per_year.xlsx', index=False)

#############################
### 2. GA Stats Per Decile
#############################

print("📊 Calculating GA stats per decile...")
ga_decile_stats = decile_data.groupby(['FF_year', 'decile']).agg(
    avg_ga=('ga_lagged', 'mean'),
    median_ga=('ga_lagged', 'median')
).reset_index()
ga_decile_stats.to_excel(f'{output_dir}/ga_stats_per_decile_per_year.xlsx', index=False)

#############################
### 3. Market Cap (ME) Per Decile
#############################

print("💰 Calculating Market Cap per decile...")
if 'ME' in decile_data.columns:
    me_decile_stats = decile_data.groupby(['FF_year', 'decile']).agg(
        avg_me=('ME', 'mean'),
        median_me=('ME', 'median')
    ).reset_index()
    me_decile_stats.to_excel(f'{output_dir}/market_cap_per_decile_per_year.xlsx', index=False)
else:
    print("⚠️ Column 'ME' not found in decile_data—skipping market cap stats.")
    me_decile_stats = None

#############################
### 4. Combined Characteristics Table
#############################

print("📑 Merging all decile statistics...")
combined_stats = pd.merge(decile_goodwill_pivot, ga_decile_stats, on=['FF_year', 'decile'], how='left')
if me_decile_stats is not None:
    combined_stats = pd.merge(combined_stats, me_decile_stats, on=['FF_year', 'decile'], how='left')
combined_stats.to_excel(f'{output_dir}/combined_portfolio_characteristics.xlsx', index=False)

#############################
### 5. Correlation Matrix
#############################

print("🔗 Creating correlation matrix...")

# Load FF factors
ff_factors_path = 'data/FamaFrench_factors_with_momentum.csv'
ff_factors = pd.read_csv(ff_factors_path)
print("✅ Loaded Fama-French factors. Columns:", ff_factors.columns.tolist())

# Parse FF date (assuming YYYYMM format, e.g., 196307)
ff_factors['date'] = pd.to_datetime(ff_factors['date'], format='%Y%m', errors='coerce')
# Normalize to end of month to match GA factor
ff_factors['date'] = ff_factors['date'] + pd.offsets.MonthEnd(0)

# Load GA factor returns (monthly)
ga_eq = pd.read_csv('output/factors/ga_factor_returns_monthly_equal.csv', parse_dates=['date'])
ga_val = pd.read_csv('output/factors/ga_factor_returns_monthly_value.csv', parse_dates=['date'])

# Merge monthly GA and FF factors
merged = pd.merge(ga_eq[['date', 'ga_factor']], ff_factors, on='date', how='inner')

# Correlation matrix
ff_cols = ['mkt_rf', 'smb', 'hml', 'rmw', 'cma', 'mom']
available_ff_cols = [col for col in ff_cols if col in merged.columns]
factors_for_corr = merged[['ga_factor'] + available_ff_cols]
corr_matrix = factors_for_corr.corr()
corr_matrix.to_excel(os.path.join(output_dir, "factor_correlation_matrix.xlsx"))

# Heatmap
plt.figure(figsize=(8, 6))
sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', center=0, fmt='.2f')
plt.title("Correlation Matrix of GA and FF Factors")
plt.savefig(os.path.join(output_dir, "factor_correlation_heatmap.png"))
plt.close()

#############################
### 6. Cumulative Return Plots with Market
#############################

print("📈 Plotting GA factor and cumulative returns...")

# Cumulative returns (monthly)
ga_eq['cum_return'] = (1 + ga_eq['ga_factor']).cumprod()
ga_val['cum_return'] = (1 + ga_val['ga_factor']).cumprod()
if not merged.empty and 'mkt_rf' in merged.columns:
    merged['mkt_cum_return'] = (1 + merged['mkt_rf']).cumprod()
else:
    print("⚠️ No matching dates for market returns—skipping market line.")
    merged['mkt_cum_return'] = np.nan

# Plot
plt.figure(figsize=(10, 6))
plt.plot(ga_eq['date'], ga_eq['cum_return'], label='Equal-weighted GA (Cumulative)')
plt.plot(ga_val['date'], ga_val['cum_return'], label='Value-weighted GA (Cumulative)')
if merged['mkt_cum_return'].notna().any():
    plt.plot(merged['date'], merged['mkt_cum_return'], label='Market (Cumulative)', linestyle='--', color='green')
plt.xlabel('Date')
plt.ylabel('Cumulative Return')
plt.title('Cumulative GA Factor vs. Market Returns')
plt.legend()
plt.grid(True, linestyle='--', alpha=0.7)
plt.savefig(f'{output_dir}/GA_vs_market_cumulative_returns.png')
plt.close()

#############################
### Final message
#############################
print("\n🎉 All File 5 outputs generated successfully! Ready for thesis use in 'analysis_output'.")