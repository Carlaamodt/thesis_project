import pandas as pd
import numpy as np

import time
start = time.time()
# your function here
print("Execution time:", time.time() - start)

# -------------------------------
# 1. LOAD DATA
# -------------------------------
# Load data (already cleaned and merged from Compustat/CRSP/Link)
compustat = pd.read_csv("__import_files__/compustat_20250217.csv", parse_dates=['date'])
crsp = pd.read_csv("__import_files__/crsp_ret_20250217.csv", parse_dates=['date'])

# -------------------------------
# 2. PREPARE GOODWILL CHANGE (GA)
# -------------------------------
# First, calculate lagged goodwill (by gvkey, fiscal year)
compustat['gdwl_prev'] = compustat.groupby('gvkey')['gdwl'].shift(1)

# Calculate Goodwill/Assets
compustat['GA_ratio'] = compustat['gdwl'] / compustat['at']

# Calculate Goodwill Change %
compustat['GA_change'] = (compustat['gdwl'] - compustat['gdwl_prev']) / compustat['gdwl_prev']

# Handling zeros automatically: If gdwl_prev == 0, this will be inf or NaN; we can leave them or cap if you prefer

# -------------------------------
# 3. FILTER DATA
# -------------------------------
# Basic filters:
# - Drop missing Goodwill/Assets change
compustat = compustat.dropna(subset=['GA_change'])

# Define fiscal year-end month for rebalancing
compustat['fyear'] = pd.to_datetime(compustat['fyear'].astype(str) + '-12-31')

# -------------------------------
# 4. PORTFOLIO ASSIGNMENT (JUNE each year based on previous year-end data)
# -------------------------------
# We align Compustat fiscal year-end to CRSP date
compustat['june_port_date'] = compustat['fyear'] + pd.DateOffset(months=6)  # Align to next June

# Filter to June dates only (for portfolio formation)
compustat_june = compustat[['gvkey', 'june_port_date', 'GA_change', 'at']].dropna()

# Assign GA quintiles (5 portfolios)
compustat_june['GA_quintile'] = compustat_june.groupby('june_port_date')['GA_change'].transform(
    lambda x: pd.qcut(x, 5, labels=False, duplicates='drop') + 1  # Quintiles labeled 1 (low GA) to 5 (high GA)
)

# -------------------------------
# 5. LINK TO CRSP AND MERGE
# -------------------------------
# First ensure we have a linking file between gvkey and permno
link_table = pd.read_csv("__import_files__/crsp_compustat_20250217.csv")

# Prepare CRSP: need month-end date
crsp['month_end'] = crsp['date'] + pd.offsets.MonthEnd(0)

# Merge Compustat GA quintiles with link table and then CRSP data
merged = compustat_june.merge(link_table, on='gvkey', how='left')
merged = merged.merge(crsp, left_on=['permno'], right_on=['permno'], how='left')

# Keep only relevant CRSP months (after portfolio formation date)
merged = merged[merged['month_end'] >= merged['june_port_date']]

# -------------------------------
# 6. VALUE & EQUAL-WEIGHTED RETURNS
# -------------------------------
# Define market cap (price * shares outstanding)
merged['market_cap'] = merged['prc'].abs() * merged['shrout']

# Monthly returns by GA portfolio
# First filter months to within 12 months after June formation date
merged['months_since_formation'] = ((merged['month_end'].dt.year - merged['june_port_date'].dt.year) * 12 + 
                                    (merged['month_end'].dt.month - merged['june_port_date'].dt.month))
merged = merged[(merged['months_since_formation'] >= 0) & (merged['months_since_formation'] <= 11)]

# Equal-weighted returns
ew_returns = merged.groupby(['month_end', 'GA_quintile'])['ret'].mean().reset_index()
ew_returns = ew_returns.pivot(index='month_end', columns='GA_quintile', values='ret')
ew_returns.columns = [f'GA_Q{int(col)}_EW' for col in ew_returns.columns]

# Value-weighted returns
def vw_ret(group):
    return np.sum(group['ret'] * group['market_cap']) / np.sum(group['market_cap'])

vw_returns = merged.groupby(['month_end', 'GA_quintile']).apply(vw_ret).reset_index()
vw_returns = vw_returns.pivot(index='month_end', columns='GA_quintile', values=0)
vw_returns.columns = [f'GA_Q{int(col)}_VW' for col in vw_returns.columns]

# -------------------------------
# 7. LONG-SHORT (High GA - Low GA)
# -------------------------------
# Calculate high-low GA spread
ew_returns['GA_HML_EW'] = ew_returns[f'GA_Q5_EW'] - ew_returns[f'GA_Q1_EW']
vw_returns['GA_HML_VW'] = vw_returns[f'GA_Q5_VW'] - vw_returns[f'GA_Q1_VW']

# -------------------------------
# 8. FINAL OUTPUT
# -------------------------------
# Merge EW and VW for final output
final_returns = ew_returns.merge(vw_returns, on='month_end', how='inner')

# Save to CSV
final_returns.to_csv('__import_files__/GA_portfolio_returns.csv')

# -------------------------------
# 9. Quick Check and Summary
# -------------------------------
print("\nSample of Final Portfolio Returns (First 5 rows):")
print(final_returns.head())

print("\nColumns of final output file:")
print(final_returns.columns.tolist())

print("\nREADY FOR MAIN(2) REGRESSIONS!")

