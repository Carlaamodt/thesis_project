import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import os

# Ensure output directory exists
output_dir = "analysis_output"
os.makedirs(output_dir, exist_ok=True)

if __name__ == "__main__":

    ###########################################
    ### 1. Plot GA Factor and Cumulative Returns
    ###########################################

    print("📈 Plotting GA factor returns...")

    # Load GA factor returns
    ga_eq = pd.read_csv('data/ga_factor_returns_annual_equal.csv')
    ga_val = pd.read_csv('data/ga_factor_returns_annual_value.csv')

    # Plot GA factor returns over time
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

    # Plot cumulative returns
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

    # Plot quintile portfolio returns (Equal-weighted)
    plt.figure(figsize=(10, 6))
    for quintile in ['1', '2', '3', '4', '5']:
        plt.plot(ga_eq['year'], ga_eq[quintile], label=f'Q{quintile}')
    plt.title('Quintile Portfolio Returns (Equal-weighted)')
    plt.xlabel('Year')
    plt.ylabel('Annual Return')
    plt.legend()
    plt.savefig(f'{output_dir}/quintile_portfolio_returns_equal.png')
    plt.close()

    # Plot Q5 - Q1 Spread (GA factor itself for visual reference)
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

    print("✅ GA factor plots saved.")


    ##################################
    ### 2. Correlation Matrix of Factors
    ##################################

    print("\n📊 Correlation matrix of GA and FF factors...")

    # Load datasets (adjust paths if needed)
    ga_factor = pd.read_csv("data/ga_factor_returns_annual_equal.csv")
    ff_factors = pd.read_csv("data/FamaFrench_factors_with_momentum.csv", parse_dates=['quarter'])

    # Normalize column names to lowercase
    ga_factor.columns = ga_factor.columns.str.lower()
    ff_factors.columns = ff_factors.columns.str.lower()

    # Convert FF to annual
    ff_factors['year'] = ff_factors['quarter'].dt.year
    ff_factors_annual = ff_factors.groupby('year').mean().reset_index()

    # Fix year column types
    ga_factor['year'] = ga_factor['year'].astype(int)
    ff_factors_annual['year'] = ff_factors_annual['year'].astype(int)

    # Merge GA and FF on year
    merged = pd.merge(ga_factor, ff_factors_annual, on="year", how="inner")

    # Now check the merged columns to verify
    print("✅ Merged dataset columns:", merged.columns.tolist())

    # Proceed with correlation matrix
    factors_for_corr = merged[['ga_factor', 'mkt', 'smb', 'hml', 'rmw', 'cma', 'mom']]
    corr_matrix = factors_for_corr.corr()
    corr_matrix.to_excel(os.path.join(output_dir, "factor_correlation_matrix.xlsx"))

    # Plot heatmap
    plt.figure(figsize=(8, 6))
    sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', center=0)
    plt.title("Correlation Matrix of GA and FF Factors")
    plt.savefig(os.path.join(output_dir, "factor_correlation_heatmap.png"))
    plt.close()

    print("✅ Correlation matrix and heatmap saved.")

    ##################################
    ### ✅ Final message
    ##################################

    print("\n🎉 Output analysis completed! Check the 'analysis_output' folder.")
