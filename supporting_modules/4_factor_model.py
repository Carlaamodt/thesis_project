import pandas as pd
import numpy as np
import statsmodels.api as sm
import os

#########################
### Load Factor Data ###
#########################

def load_data(weighting="equal", ga_choice="GA1_lagged"):
    """Loads monthly GA factor returns and Fama-French factors."""
    print(f"📥 Loading monthly {weighting}-weighted {ga_choice} factor returns...")
    ga_factor_path = f"output/factors/ga_factor_returns_monthly_{weighting}_{ga_choice}.csv"
    if not os.path.exists(ga_factor_path):
        print(f"❌ Error: GA factor return file not found at {ga_factor_path}")
        return None, None
    
    ga_factor = pd.read_csv(ga_factor_path, parse_dates=['date'])

    print("📥 Loading monthly Fama-French factors...")
    ff_factors_path = "data/FamaFrench_factors_with_momentum.csv"
    if not os.path.exists(ff_factors_path):
        print(f"❌ Error: Fama-French file not found at {ff_factors_path}")
        return None, None

    ff_factors = pd.read_csv(ff_factors_path, parse_dates=['date'])
    
    # Ensure lowercase for consistency
    ga_factor.columns = ga_factor.columns.str.lower()
    ff_factors.columns = ff_factors.columns.str.lower()

    # Check and display date ranges
    print("✅ GA factor date range:", ga_factor['date'].min(), "to", ga_factor['date'].max())
    print("✅ Fama-French factor date range:", ff_factors['date'].min(), "to", ff_factors['date'].max())

    # Align Fama-French dataset to start no earlier than GA factor
    start_date = ga_factor['date'].min()
    ff_factors = ff_factors[ff_factors['date'] >= start_date.replace(day=1)]  # Align to month start

    # Check required columns
    required_factors = ['mkt_rf', 'smb', 'hml', 'rmw', 'cma', 'mom', 'rf']
    missing_factors = [col for col in required_factors if col not in ff_factors.columns]
    if missing_factors:
        print(f"❌ Error: Missing columns in Fama-French dataset: {missing_factors}")
        return None, None

    print("✅ Fama-French dataset filtered. Period:", ff_factors['date'].min(), "to", ff_factors['date'].max())
    return ga_factor, ff_factors

#############################
### Merge and Prepare Data ###
#############################

def merge_data(ga_factor, ff_factors):
    """Merges GA factor data with Fama-French factors on date."""
    print("🔄 Merging GA factor with Fama-French factors...")
    
    # Adjust FF dates to month-end to match GA factors
    ff_factors['date'] = ff_factors['date'] + pd.offsets.MonthEnd(0)

    # Merge on date
    merged_df = pd.merge(ga_factor, ff_factors, on="date", how="inner")

    # Drop 2024 and beyond if any
    merged_df = merged_df[merged_df['date'].dt.year < 2024]
    print(f"✅ Merged dataset: {merged_df.shape[0]} months, from {merged_df['date'].min()} to {merged_df['date'].max()}")
    return merged_df

#########################
### Run Factor Models ###
#########################

def run_factor_models(df, weighting, ga_choice):
    """Runs factor regressions for GA factor returns with Newey-West robust standard errors and annualized alpha."""
    print(f"📊 Running factor regressions for {weighting}-weighted {ga_choice} factor...")

    # Check necessary columns
    if 'ga_factor' not in df.columns or 'rf' not in df.columns:
        print("❌ Error: GA factor or risk-free rate missing!")
        return None

    # Compute GA factor excess returns
    df['ga_factor_excess'] = df['ga_factor'] - df['rf']
    print("📈 GA factor excess return summary:\n", df['ga_factor_excess'].describe())

    # Skip if mostly NaNs
    if df['ga_factor_excess'].isna().mean() > 0.5:
        print(f"⚠️ Warning: Over 50% missing GA factor excess returns for {weighting}-weighted.")
        return None

    # Define models to run
    factor_models = {
        "CAPM": ["mkt_rf"],
        "Fama-French 3-Factor": ["mkt_rf", "smb", "hml"],
        "Fama-French 5-Factor": ["mkt_rf", "smb", "hml", "rmw", "cma"],
        "Fama-French 5 + Momentum": ["mkt_rf", "smb", "hml", "rmw", "cma", "mom"],
    }

    all_results = []

    # Loop through models
    for idx, (model_name, factor_list) in enumerate(factor_models.items(), 1):
        print(f"🔢 Running {model_name} ({idx}/{len(factor_models)})...")

        # Filter for available factors
        available_factors = [f for f in factor_list if f in df.columns]

        # Regression setup
        X = sm.add_constant(df[available_factors], has_constant='add')
        y = df['ga_factor_excess']

        # Fit OLS model with HAC (Newey-West) robust standard errors
        model = sm.OLS(y, X, missing='drop').fit(cov_type='HAC', cov_kwds={'maxlags': 3})

        # Calculate annualized alpha
        alpha_monthly = model.params.get('const', np.nan)
        alpha_annualized = alpha_monthly * 12

        # Collect result summary
        res = {
            'Model': model_name,
            'Weighting': weighting,
            'GA Choice': ga_choice,
            'Alpha (const)': alpha_monthly,
            'Alpha (annualized)': alpha_annualized,
            'Alpha p-value (HAC)': model.pvalues.get('const', np.nan),
            'Alpha t-stat (HAC)': model.tvalues.get('const', np.nan),
            'R-squared': model.rsquared,
            'Adj. R-squared': model.rsquared_adj,
            'Observations': int(model.nobs),
        }

        # Collect betas and stats for each factor
        for f in available_factors:
            res[f'{f.upper()} beta'] = model.params.get(f, np.nan)
            res[f'{f.upper()} p-value (HAC)'] = model.pvalues.get(f, np.nan)
            res[f'{f.upper()} t-stat (HAC)'] = model.tvalues.get(f, np.nan)

        all_results.append(res)

        # Print model summary for review
        print(f"\n📊 {model_name} Regression Results ({weighting}-weighted {ga_choice}) with Newey-West HAC SE:\n")
        print(model.summary())

    return pd.DataFrame(all_results)

#########################
### Main Execution ###
#########################

def main():
    """Main function to perform regression analysis and save results to multi-sheet Excel."""
    results_by_ga = {"GA1": None, "GA2": None, "GA3": None}  # Store results for each GA

    for weighting in ["equal", "value"]:
        for ga_choice in ["GA1_lagged", "GA2_lagged", "GA3_lagged"]:
            # Load data
            ga_factor, ff_factors = load_data(weighting=weighting, ga_choice=ga_choice)
            if ga_factor is None or ff_factors is None:
                print(f"❌ Data loading failed for {weighting}-weighted {ga_choice} factor.")
                continue  # Skip to next GA if file missing

            # Merge datasets
            df = merge_data(ga_factor, ff_factors)
            if len(df) == 0:
                print(f"❌ Merged dataset empty for {weighting}-weighted {ga_choice}.")
                continue

            # Run regressions
            model_results = run_factor_models(df, weighting, ga_choice)
            if model_results is not None:
                # Assign to corresponding GA sheet (strip '_lagged' for sheet name)
                ga_key = ga_choice.replace("_lagged", "")
                if results_by_ga[ga_key] is None:
                    results_by_ga[ga_key] = model_results
                else:
                    results_by_ga[ga_key] = pd.concat([results_by_ga[ga_key], model_results], ignore_index=True)

    # Save results to multi-sheet Excel
    output_path = "output/ga_factor_regression_results_monthly.xlsx"
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    with pd.ExcelWriter(output_path, engine='xlsxwriter') as writer:
        for ga_key, results in results_by_ga.items():
            if results is not None:
                results.to_excel(writer, sheet_name=ga_key, index=False)
            else:
                # Write empty sheet if no results
                pd.DataFrame().to_excel(writer, sheet_name=ga_key, index=False)
    
    print(f"\n✅ All monthly regression results saved to {output_path} with sheets GA1, GA2, GA3")

################################
### Run Main ###
################################

if __name__ == "__main__":
    main()