import pandas as pd
import numpy as np
import statsmodels.api as sm
import os

#########################
### Load Factor Data ###
#########################

def load_data(weighting="equal"):
    """Loads GA factor returns and Fama-French factors, aligned annually."""
    print(f"📥 Loading {weighting}-weighted GA factor returns...")
    ga_factor = pd.read_csv(f"data/ga_factor_returns_annual_{weighting}.csv", parse_dates=['year'])

    print("📥 Loading Fama-French factors...")
    ff_factors = pd.read_csv("data/FamaFrench_factors_with_momentum.csv", parse_dates=['quarter'])

    # Convert Fama-French quarterly to annual
    ff_factors['year'] = ff_factors['quarter'].dt.year
    ff_factors = ff_factors.groupby('year').mean().reset_index()

    # Lowercase column names for consistency
    ga_factor.columns = ga_factor.columns.str.lower()
    ff_factors.columns = ff_factors.columns.str.lower()

    # Align date ranges - FIXED to handle Timestamp vs int
    start_year = ga_factor['year'].min().year if isinstance(ga_factor['year'].min(), pd.Timestamp) else int(ga_factor['year'].min())
    ff_factors = ff_factors[ff_factors['year'] >= start_year]

    print("✅ Fama-French dataset filtered. Date range:", ff_factors['year'].min(), "to", ff_factors['year'].max())
    return ga_factor, ff_factors


#############################
### Merge and Prepare Data ###
#############################

def merge_data(ga_factor, ff_factors):
    """Merges GA factor data with Fama-French factors on year."""
    print("🔄 Merging factor datasets...")

    # Ensure 'year' is int in both DataFrames
    ga_factor['year'] = ga_factor['year'].dt.year if np.issubdtype(ga_factor['year'].dtype, np.datetime64) else ga_factor['year']
    ff_factors['year'] = ff_factors['year'].astype(int)

    merged_df = pd.merge(ga_factor, ff_factors, on="year", how="inner")
    print("✅ Merged dataset. Total periods:", len(merged_df))
    return merged_df


#########################
### Run Factor Models ###
#########################

def run_factor_models(df, weighting):
    """Runs factor models with GA factor excess returns and collects results."""
    print(f"📊 Running regression models for {weighting}-weighted GA factor...")

    if 'ga_factor' not in df.columns or 'rf' not in df.columns:
        print("❌ Error: Missing GA factor or risk-free rate in data!")
        return

    df['ga_factor_excess'] = df['ga_factor'] - df['rf']

    factor_models = {
        "CAPM": df[["mkt"]],
        "Fama-French 3-Factor": df[["mkt", "smb", "hml"]],
        "Fama-French 5-Factor": df[["mkt", "smb", "hml", "rmw", "cma"]],
        "Fama-French 5 + Momentum": df[["mkt", "smb", "hml", "rmw", "cma", "mom"]],
    }

    all_results = []

    for model_name, factors in factor_models.items():
        X = sm.add_constant(factors, has_constant='add')
        y = df['ga_factor_excess']

        model = sm.OLS(y, X, missing='drop').fit()

        # Collect results in dict format
        res = {
            'Model': model_name,
            'Alpha (const)': model.params.get('const', float('nan')),
            'Alpha p-value': model.pvalues.get('const', float('nan')),
            'R-squared': model.rsquared,
            'Adj. R-squared': model.rsquared_adj,
            'Observations': int(model.nobs),
            'MKT beta': model.params.get('mkt', float('nan')),
            'MKT p-value': model.pvalues.get('mkt', float('nan')),
            'SMB beta': model.params.get('smb', float('nan')),
            'SMB p-value': model.pvalues.get('smb', float('nan')),
            'HML beta': model.params.get('hml', float('nan')),
            'HML p-value': model.pvalues.get('hml', float('nan')),
            'RMW beta': model.params.get('rmw', float('nan')),
            'RMW p-value': model.pvalues.get('rmw', float('nan')),
            'CMA beta': model.params.get('cma', float('nan')),
            'CMA p-value': model.pvalues.get('cma', float('nan')),
            'MOM beta': model.params.get('mom', float('nan')),
            'MOM p-value': model.pvalues.get('mom', float('nan')),
        }

        all_results.append(res)

        print(f"\n📊 {model_name} Regression Results for {weighting}-weighted GA Factor:\n")
        print(model.summary())

    return pd.DataFrame(all_results)


#########################
### Main Execution ###
#########################

def main():
    """Main function to run GA factor regressions, save results, and calculate quintile excess returns."""

    final_results = []

    # Run factor models for both equal- and value-weighted GA factor returns
    for weighting in ["equal", "value"]:
        ga_factor, ff_factors = load_data(weighting=weighting)
        df = merge_data(ga_factor, ff_factors)
        if df.empty:
            print(f"❌ Merged dataset empty for {weighting}-weighted!")
            continue
        model_results = run_factor_models(df, weighting)
        model_results['Weighting'] = weighting  # Add weighting label
        final_results.append(model_results)

    # Combine equal- and value-weighted regression results
    combined_results = pd.concat(final_results, ignore_index=True)

    # Save regression results to Excel
    output_path = "output/ga_factor_regression_results.xlsx"
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    combined_results.to_excel(output_path, index=False)
    print(f"\n✅ All regression results saved to {output_path}")

    # ------------------------------------------------------------
    # Now calculate GA quintile returns as excess returns
    # ------------------------------------------------------------
    print("\n💾 Saving GA quintile excess returns (Equal & Value weighted)...")

    # Load risk-free rate data
    rf = pd.read_csv('data/FamaFrench_factors_with_momentum.csv', parse_dates=['quarter'])
    rf['year'] = rf['quarter'].dt.year
    rf_annual = rf.groupby('year')['rf'].mean().reset_index()

    # Load GA factor returns
    ga_eq = pd.read_csv('data/ga_factor_returns_annual_equal.csv', parse_dates=['year'])
    ga_val = pd.read_csv('data/ga_factor_returns_annual_value.csv', parse_dates=['year'])

    # Ensure year is int for merge
    ga_eq['year'] = ga_eq['year'].dt.year if pd.api.types.is_datetime64_any_dtype(ga_eq['year']) else ga_eq['year']
    ga_val['year'] = ga_val['year'].dt.year if pd.api.types.is_datetime64_any_dtype(ga_val['year']) else ga_val['year']
    rf_annual['year'] = rf_annual['year'].astype(int)

    # Merge RF to GA factor returns
    ga_eq = pd.merge(ga_eq, rf_annual, on='year', how='left')
    ga_val = pd.merge(ga_val, rf_annual, on='year', how='left')

    # Calculate excess returns (quintile return - rf)
    for q in ['1', '2', '3', '4', '5']:
        ga_eq[q] = ga_eq[q] - ga_eq['rf']
        ga_val[q] = ga_val[q] - ga_val['rf']

    # Prepare final table of quintile excess returns
    quintile_returns_combined = ga_eq[['year', '1', '2', '3', '4', '5']].copy()
    quintile_returns_combined.columns = ['year', 'Q1_equal', 'Q2_equal', 'Q3_equal', 'Q4_equal', 'Q5_equal']

    value_subset = ga_val[['year', '1', '2', '3', '4', '5']].copy()
    value_subset.columns = ['year', 'Q1_value', 'Q2_value', 'Q3_value', 'Q4_value', 'Q5_value']

    # Merge Equal & Value returns
    quintile_returns_final = pd.merge(quintile_returns_combined, value_subset, on='year', how='inner')

    # Save to Excel
    quintile_returns_path = 'output/ga_quintile_returns_excess.xlsx'
    quintile_returns_final.to_excel(quintile_returns_path, index=False)

    print(f"✅ Quintile excess returns (Equal & Value weighted) saved to {quintile_returns_path}")

    # ------------------------------------------------------------
    # Final message
    # ------------------------------------------------------------
    print("\n🎉 All regressions and excess returns saved in 'output' folder.")

################################
### Run Main
################################

if __name__ == "__main__":
    main()