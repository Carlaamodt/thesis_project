import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import os

# Set output directory
output_dir = "analysis_output"
os.makedirs(output_dir, exist_ok=True)

# Load decile assignment data
print("📥 Loading decile assignment data...")
decile_data = pd.read_csv("analysis_output/decile_assignments.csv", parse_dates=['crsp_date'])

# Filter for valid deciles (1-10)
decile_data = decile_data[decile_data['decile'].between(1, 10)]

#############################
### 1. Firms per Decile and Goodwill Breakdown
#############################

print("🔢 Counting firms per decile and goodwill status...")
decile_goodwill_counts = decile_data.groupby(['FF_year', 'decile', 'has_goodwill_firm']).agg(
    num_firms=('permno', 'nunique')
).reset_index()

# Pivot to get counts of firms with and without goodwill
decile_goodwill_pivot = decile_goodwill_counts.pivot_table(
    index=['FF_year', 'decile'],
    columns='has_goodwill_firm',
    values='num_firms',
    fill_value=0
).reset_index()

# Important: Remove name of columns to avoid "has_goodwill_firm" sticking around
decile_goodwill_pivot.columns.name = None

# Rename columns clearly
if 0 in decile_goodwill_pivot.columns and 1 in decile_goodwill_pivot.columns:
    decile_goodwill_pivot.rename(columns={0: 'firms_without_goodwill', 1: 'firms_with_goodwill'}, inplace=True)
else:
    decile_goodwill_pivot['firms_without_goodwill'] = 0
    decile_goodwill_pivot['firms_with_goodwill'] = 0

# Total firms per decile
decile_goodwill_pivot['total_firms'] = decile_goodwill_pivot['firms_with_goodwill'] + decile_goodwill_pivot['firms_without_goodwill']

# Save output
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
me_decile_stats = decile_data.groupby(['FF_year', 'decile']).agg(
    avg_me=('ME', 'mean'),
    median_me=('ME', 'median')
).reset_index()
me_decile_stats.to_excel(f'{output_dir}/market_cap_per_decile_per_year.xlsx', index=False)

#############################
### 4. Combined Characteristics Table
#############################

print("📑 Merging all decile statistics...")
combined_stats = pd.merge(decile_goodwill_pivot, ga_decile_stats, on=['FF_year', 'decile'], how='left')
combined_stats = pd.merge(combined_stats, me_decile_stats, on=['FF_year', 'decile'], how='left')
combined_stats.to_excel(f'{output_dir}/combined_portfolio_characteristics.xlsx', index=False)

#############################
### 5. Correlation Matrix
#############################

print("🔗 Creating correlation matrix...")
ga_eq = pd.read_csv('data/ga_factor_returns_annual_equal.csv')
ff_factors = pd.read_csv('data/FamaFrench_factors_with_momentum.csv', parse_dates=['quarter'])
ff_factors['year'] = ff_factors['quarter'].dt.year
ff_factors_annual = ff_factors.groupby('year').mean().reset_index()

# Merge GA and FF factors
merged = pd.merge(ga_eq, ff_factors_annual, on='year', how='inner')
factors_for_corr = merged[['GA_factor', 'mkt', 'smb', 'hml', 'rmw', 'cma', 'mom']]
corr_matrix = factors_for_corr.corr()
corr_matrix.to_excel(os.path.join(output_dir, "factor_correlation_matrix.xlsx"))

# Plot heatmap
plt.figure(figsize=(8, 6))
sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', center=0)
plt.title("Correlation Matrix of GA and FF Factors")
plt.savefig(os.path.join(output_dir, "factor_correlation_heatmap.png"))
plt.close()

#############################
### 6. Cumulative Return Plots with Market
#############################

print("📈 Plotting GA factor and cumulative returns...")
ga_val = pd.read_csv('data/ga_factor_returns_annual_value.csv')
ga_eq = pd.merge(ga_eq, ff_factors_annual[['year', 'mkt']], on='year', how='left')
ga_val = pd.merge(ga_val, ff_factors_annual[['year', 'mkt']], on='year', how='left')

ga_eq['cum_return'] = (1 + ga_eq['GA_factor']).cumprod()
ga_val['cum_return'] = (1 + ga_val['GA_factor']).cumprod()
ga_eq['mkt_cum_return'] = (1 + ga_eq['mkt']).cumprod()

# Plot
plt.figure(figsize=(10, 6))
plt.plot(ga_eq['year'], ga_eq['cum_return'], label='Equal-weighted GA (Cumulative)')
plt.plot(ga_val['year'], ga_val['cum_return'], label='Value-weighted GA (Cumulative)')
plt.plot(ga_eq['year'], ga_eq['mkt_cum_return'], label='Market (Cumulative)', linestyle='--')
plt.legend()
plt.title('Cumulative GA Factor vs. Market Returns')
plt.savefig(f'{output_dir}/GA_vs_market_cumulative_returns.png')
plt.close()

#############################
### Final message
#############################
print("\n🎉 All File 5 outputs generated successfully! Ready for thesis use in 'analysis_output'.")