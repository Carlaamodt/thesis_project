import pandas as pd
import os
import statsmodels.api as sm
import numpy as np
import logging

EXPORT_LATEX = False  # Toggle LaTeX export on/off

#########################
### Setup Logging ###
#########################

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger()

#########################
### Load Factor Data ###
#########################

def load_data():
    """Loads monthly GA factor returns and WRDS-downloaded Fama-French factors (in decimals)."""
    # Load GA factors
    print("📥 Loading monthly GA factor returns...")
    ga_factor_path = "output/factors/ga_factors.csv"
    if not os.path.exists(ga_factor_path):
        print(f"❌ Error: GA factor return file not found at {ga_factor_path}")
        return None, None
    
    ga_factor = pd.read_csv(ga_factor_path, parse_dates=['date'])
    ga_factor['date'] = pd.to_datetime(ga_factor['date']) + pd.offsets.MonthEnd(0)
    ga_factor.set_index('date', inplace=True)
    ga_factor.columns = ga_factor.columns.str.lower()

    # Load local WRDS-downloaded Fama-French data
    print("📥 Loading WRDS-downloaded Fama-French 5-factor + Momentum data...")
    ff_factors_path = "data/FamaFrench_factors_with_momentum.csv"
    if not os.path.exists(ff_factors_path):
        print(f"❌ Error: Fama-French factor file not found at {ff_factors_path}")
        return None, None

    ff_factors = pd.read_csv(ff_factors_path, parse_dates=['date'])
    ff_factors['date'] = pd.to_datetime(ff_factors['date']) + pd.offsets.MonthEnd(0)
    ff_factors.set_index('date', inplace=True)
    ff_factors.columns = ff_factors.columns.str.lower()

    # Check required columns
    required_factors = ['mkt_rf', 'smb', 'hml', 'rmw', 'cma', 'mom', 'rf']
    missing_factors = [col for col in required_factors if col not in ff_factors.columns]
    if missing_factors:
        print(f"❌ Error: Missing columns in Fama-French dataset: {missing_factors}")
        return None, None

    # Filter dates to GA factor range
    start_date = ga_factor.index.min()
    ff_factors = ff_factors[ff_factors.index >= start_date]

    print("✅ GA factor date range:", ga_factor.index.min(), "to", ga_factor.index.max())
    print("✅ Fama-French factor date range:", ff_factors.index.min(), "to", ff_factors.index.max())
    return ga_factor, ff_factors

#############################
### Merge and Prepare Data ###
#############################

def merge_data(ga_factor, ff_factors):
    """Merges GA factor data with Fama-French factors on date index."""
    print("🔄 Merging GA factor with Fama-French factors...")

    merged_df = ga_factor.join(ff_factors, how='inner')

    # Drop future dates (2024+)
    merged_df = merged_df[merged_df.index.year < 2024]
    merged_df.sort_index(inplace=True)

    print(f"✅ Merged dataset: {merged_df.shape[0]} months, from {merged_df.index.min()} to {merged_df.index.max()}")

    # Check for any missing data
    missing = merged_df.isna().sum()
    if missing.sum() > 0:
        print(f"⚠️ Warning: Missing values in merged dataset:\n{missing[missing > 0]}")

    return merged_df

#########################
### Run Factor Models ###
#########################

def run_factor_models(df, ga_factor_column, ga_choice, weighting, size_group):
    """Runs factor regressions for GA factor returns with Newey-West robust standard errors and annualized alpha."""
    print(f"📊 Running factor regressions for {weighting}-weighted {ga_choice} factor (Size: {size_group})...")
    date_range = f"{df.index.min().strftime('%Y-%m')} to {df.index.max().strftime('%Y-%m')}"
    logger.info(f"Regression date range: {date_range}")

    # Compute GA factor excess returns directly in df
    logger.info(f"Running diagnostics for {ga_factor_column} (raw GA factor)")
    print("📊 Raw GA factor return summary:\n", df[ga_factor_column].describe())
    print(f"❓ Missing: {df[ga_factor_column].isna().sum()} / {len(df)} rows")

    # Check if 'rf' exists and compute excess returns
    if 'rf' not in df.columns:
        logger.error("Risk-free rate 'rf' not found in dataset!")
        return None
    df['ga_factor_excess'] = df[ga_factor_column] - df['rf']

    print("📈 GA factor excess return summary:\n", df['ga_factor_excess'].describe())
    logger.info("Newey-West standard errors using maxlags=3")

    # Skip if mostly NaNs
    if df['ga_factor_excess'].isna().mean() > 0.5:
        print(f"⚠️ Warning: Over 50% missing GA factor excess returns for {weighting}-weighted {ga_choice} (Size: {size_group}).")
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
        if not available_factors:
            logger.warning(f"No factors available for {model_name}")
            continue

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
            'Size Group': size_group,
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

        # Log model summary for review
        logger.info(f"{model_name} Regression Results ({weighting}-weighted {ga_choice}, Size: {size_group}) with Newey-West HAC SE:\n{model.summary()}")

        if EXPORT_LATEX:
            latex_dir = os.path.join("output", "latex")
            os.makedirs(latex_dir, exist_ok=True)
            latex_path = os.path.join(
                latex_dir,
                f"{ga_choice}_{weighting}_{size_group}_{model_name.replace(' ', '_')}.tex"
            )
            try:
                from statsmodels.iolib.summary2 import summary_col
                summary = summary_col([model], stars=True, model_names=[model_name])
                with open(latex_path, "w") as f:
                    f.write(summary.as_latex())
                logger.info(f"📄 Exported LaTeX table to {latex_path}")
            except Exception as e:
                logger.warning(f"⚠️ Failed to export LaTeX for {ga_choice}, {model_name}: {e}")

    return pd.DataFrame(all_results)

#########################
### Main Execution ###
#########################

def main():
    """Main function to perform regression analysis and save results to multi-sheet Excel."""
    # Define GA choices and factor columns
    ga_choices = {
        "goodwill_to_sales_lagged": "GA1",
        "goodwill_to_equity_lagged": "GA2",
        "goodwill_to_market_cap_lagged": "GA3"
    }
    
    # Store results for each GA metric
    results_by_ga = {"GA1": [], "GA2": [], "GA3": []}

    # Load data
    ga_factor, ff_factors = load_data()
    if ga_factor is None or ff_factors is None:
        print("❌ Data loading failed.")
        return

    # Merge datasets
    df = merge_data(ga_factor, ff_factors)
    if len(df) == 0:
        print("❌ Merged dataset empty.")
        return

    # Loop over each GA metric, weighting, and size group
    for ga_choice, ga_key in ga_choices.items():
        print(f"\n🔍 Processing GA metric: {ga_choice} (Sheet: {ga_key})")
        for weighting in ["ew", "vw"]:
            for size_group in ["all", "small", "big"]:
                # Construct the factor column name
                ga_factor_column = f"{ga_choice}_{weighting}" if size_group == "all" else f"{ga_choice}_{size_group}_{weighting}"
                
                if ga_factor_column not in df.columns:
                    print(f"⚠️ Warning: Factor column {ga_factor_column} not found in dataset. Skipping...")
                    continue

                # Run regressions
                model_results = run_factor_models(df, ga_factor_column, ga_choice, weighting, size_group)
                if model_results is not None:
                    print(f"✅ Adding results for {ga_factor_column} to {ga_key}")
                    results_by_ga[ga_key].append(model_results)
                else:
                    print(f"❌ No results for {ga_factor_column}")

    # Combine results for each GA metric and save to Excel
    output_path = "output/ga_factor_regression_results_monthly.xlsx"
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    with pd.ExcelWriter(output_path, engine='xlsxwriter') as writer:
        for ga_key in results_by_ga:
            if results_by_ga[ga_key]:
                combined_results = pd.concat(results_by_ga[ga_key], ignore_index=True)
                # Round numerical columns for better readability
                numerical_cols = [col for col in combined_results.columns if 'beta' in col or 'p-value' in col or 't-stat' in col or 'Alpha' in col or 'R-squared' in col]
                combined_results[numerical_cols] = combined_results[numerical_cols].round(4)
                print(f"📝 Writing {len(combined_results)} rows to sheet {ga_key}")
                combined_results.to_excel(writer, sheet_name=ga_key, index=False)
            else:
                print(f"⚠️ No results to write for sheet {ga_key}. Creating empty sheet.")
                pd.DataFrame().to_excel(writer, sheet_name=ga_key, index=False)
    
    print(f"\n✅ All monthly regression results saved to {output_path} with sheets GA1, GA2, GA3")

################################
### Run Main ###
################################

if __name__ == "__main__":
    main()