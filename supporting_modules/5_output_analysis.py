
import pandas as pd
import matplotlib.pyplot as plt
import os

# Load GA factor returns
ga_eq = pd.read_csv('data/ga_factor_returns_annual_equal.csv')
ga_val = pd.read_csv('data/ga_factor_returns_annual_value.csv')

# Output directory for plots
output_dir = "analysis_output"
os.makedirs(output_dir, exist_ok=True)

# Plot 1: GA factor returns over time
plt.figure(figsize=(10, 6))
plt.plot(ga_eq['year'], ga_eq['GA_factor'], label='Equal-weighted GA')
plt.plot(ga_val['year'], ga_val['GA_factor'], label='Value-weighted GA')
plt.axhline(0, color='gray', linestyle='--')
plt.title('GA Factor Returns Over Time')
plt.xlabel('Year')
plt.ylabel('Annual Return')
plt.legend()
plt.savefig(f'{output_dir}/GA_factor_returns_over_time.png')
plt.close()
print("✅ GA Factor returns over time saved.")

# Plot 2: Cumulative GA factor returns
ga_eq['cum_return'] = (1 + ga_eq['GA_factor']).cumprod()
ga_val['cum_return'] = (1 + ga_val['GA_factor']).cumprod()

plt.figure(figsize=(10, 6))
plt.plot(ga_eq['year'], ga_eq['cum_return'], label='Equal-weighted GA (Cumulative)')
plt.plot(ga_val['year'], ga_val['cum_return'], label='Value-weighted GA (Cumulative)')
plt.title('Cumulative GA Factor Returns')
plt.xlabel('Year')
plt.ylabel('Cumulative Return (Growth of $1)')
plt.legend()
plt.savefig(f'{output_dir}/GA_cumulative_returns.png')
plt.close()
print("✅ Cumulative GA returns saved.")

# Plot 3: Quintile portfolio returns (only equal-weighted as example)
for quintile in [1, 2, 3, 4, 5]:
    plt.plot(ga_eq['year'], ga_eq[str(quintile)], label=f'Q{quintile}')

plt.title('Quintile Portfolio Returns (Equal-weighted)')
plt.xlabel('Year')
plt.ylabel('Annual Return')
plt.legend()
plt.savefig(f'{output_dir}/quintile_portfolio_returns_equal.png')
plt.close()
print("✅ Quintile portfolio returns (equal-weighted) saved.")

# Plot 4: Q5 - Q1 Spread (GA factor itself for visual reference)
plt.figure(figsize=(10, 6))
plt.plot(ga_eq['year'], ga_eq['GA_factor'], label='Q5 - Q1 Spread (Equal-weighted GA)')
plt.plot(ga_val['year'], ga_val['GA_factor'], label='Q5 - Q1 Spread (Value-weighted GA)')
plt.axhline(0, color='gray', linestyle='--')
plt.title('GA Factor (Q5 - Q1 Spread) Over Time')
plt.xlabel('Year')
plt.ylabel('Annual Spread Return')
plt.legend()
plt.savefig(f'{output_dir}/GA_spread_over_time.png')
plt.close()
print("✅ GA spread (Q5-Q1) over time saved.")