import pandas as pd
import os
import statsmodels.api as sm
import numpy as np
import logging

EXPORT_LATEX = False  # Toggle LaTeX export on/off

# ----------------------------
# Logging
# ----------------------------
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger()

# ----------------------------
# Compute Annualized Sharpe Ratio
# ----------------------------
def compute_annualized_sharpe(series: pd.Series) -> tuple:
    """
    Compute the annualized Sharpe ratio, mean return, and standard deviation for a series.
    
    Args:
        series (pd.Series): Time series of excess returns.
    
    Returns:
        tuple: (Sharpe ratio, annualized mean return, annualized standard deviation)
    """
    series = series.dropna()
    if len(series) < 12:  # Require at least 1 year of data
        logger.warning(f"Insufficient data for Sharpe ratio: {len(series)} observations")
        return np.nan, np.nan, np.nan
    mean_return = series.mean() * 12
    std_dev = series.std() * np.sqrt(12)
    sharpe = mean_return / std_dev if std_dev > 0 else np.nan
    return round(sharpe, 4), round(mean_return, 4), round(std_dev, 4)  # Fixed mean_ret to mean_return

# ----------------------------
# Load Fama-French Factors
# ----------------------------
def load_ff_factors():
    """Load Fama-French factors with momentum from CSV."""
    ff_path = "data/FamaFrench_factors_with_momentum.csv"
    if not os.path.exists(ff_path):
        raise FileNotFoundError(f"Missing file: {ff_path}")

    print("📥 Loading Fama-French factors with momentum...")
    ff = pd.read_csv(ff_path, parse_dates=['date'])
    ff['date'] = pd.to_datetime(ff['date']) + pd.offsets.MonthEnd(0)
    ff.set_index('date', inplace=True)
    ff.columns = ff.columns.str.lower()

    required = ['mkt_rf', 'smb', 'hml', 'rmw', 'cma', 'mom', 'rf']
    missing = [col for col in required if col not in ff.columns]
    if missing:
        raise ValueError(f"Missing columns in Fama-French file: {missing}")

    print(f"✅ Fama-French factors loaded: {len(ff)} months, from {ff.index.min()} to {ff.index.max()}")
    return ff

# ----------------------------
# Run Regressions
# ----------------------------
def run_factor_models(df, column, ga_choice, weighting, size_group, is_hedge=False):
    """Run factor model regressions with Newey-West robust standard errors."""
    logger.info(f"Running regression for {column} (GA: {ga_choice}, Weighting: {weighting}, Size: {size_group})")
    print(f"📊 Running regressions for {column}...")

    # Compute excess returns
    if is_hedge:
        df['ga_excess'] = df[column] - df['rf']
        print(f"📈 Excess return summary for {column}:\n{df['ga_excess'].describe()}")
    else:
        df['ga_excess'] = df[column]  # No rf deduction for individual portfolios
        print(f"📈 Raw return summary for {column}:\n{df['ga_excess'].describe()}")

    if df['ga_excess'].isna().mean() > 0.5:
        logger.warning(f"Too many NaNs in {column}. Skipping.")
        print(f"⚠️ Warning: Over 50% missing data for {column}. Skipping...")
        return None

    models = {
        "CAPM": ["mkt_rf"],
        "FF3": ["mkt_rf", "smb", "hml"],
        "FF5": ["mkt_rf", "smb", "hml", "rmw", "cma"],
        "FF5+MOM": ["mkt_rf", "smb", "hml", "rmw", "cma", "mom"]
    }

    results = []

    for model_name, factors in models.items():
        print(f"🔢 Running {model_name} model...")
        X = sm.add_constant(df[factors], has_constant='add')
        y = df['ga_excess']
        model = sm.OLS(y, X, missing='drop').fit(cov_type='HAC', cov_kwds={'maxlags': 3})

        label = f"{ga_choice}_{column}"
        res = {
            'Model': model_name,
            'Portfolio': column,
            'Weighting': weighting,
            'Size Group': size_group,
            'GA Choice': ga_choice,
            'Alpha (monthly)': model.params.get('const', np.nan),
            'Alpha (annualized)': model.params.get('const', np.nan) * 12,
            'Alpha t-stat (HAC)': model.tvalues.get('const', np.nan),
            'Alpha p-value (HAC)': model.pvalues.get('const', np.nan),
            'R-squared': model.rsquared,
            'Adj. R-squared': model.rsquared_adj,
            'Obs': int(model.nobs)
        }

        for factor in factors:
            res[f'{factor.upper()} beta'] = model.params.get(factor, np.nan)
            res[f'{factor.upper()} t-stat'] = model.tvalues.get(factor, np.nan)
            res[f'{factor.upper()} p-value'] = model.pvalues.get(factor, np.nan)

        results.append(res)

        # LaTeX export (optional)
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

    return pd.DataFrame(results)

# ----------------------------
# Process One GA File
# ----------------------------
def process_ga_file(filepath, ff_factors, ga_choice):
    """Process a single GA factor file and run regressions."""
    if not os.path.exists(filepath):
        logger.error(f"Missing GA file: {filepath}")
        print(f"❌ Missing GA file: {filepath}")
        return None

    print(f"📥 Loading GA factor data from {filepath}...")
    df = pd.read_csv(filepath, parse_dates=['crsp_date'])
    df['crsp_date'] = pd.to_datetime(df['crsp_date']) + pd.offsets.MonthEnd(0)
    df.set_index('crsp_date', inplace=True)
    df.columns = df.columns.str.lower()

    # Merge with Fama-French factors
    ff = ff_factors[ff_factors.index.isin(df.index)]
    merged = df.join(ff, how='inner').sort_index()

    # Filter out dates after 2023
    merged = merged[merged.index.year <= 2023]
    print(f"✅ Merged dataset: {len(merged)} months, from {merged.index.min()} to {merged.index.max()}")

    # Check for missing values after merging
    missing = merged.isna().sum()
    if missing.sum() > 0:
        logger.warning(f"Missing values in merged dataset:\n{missing[missing > 0]}")
        print(f"⚠️ Warning: Missing values in merged dataset:\n{missing[missing > 0]}")

    # Check for date continuity
    expected_dates = pd.date_range(start=merged.index.min(), end=merged.index.max(), freq='ME')
    missing_dates = expected_dates[~expected_dates.isin(merged.index)]
    if missing_dates.size > 0:
        logger.warning(f"Missing {len(missing_dates)} months in merged dataset: {missing_dates}")
        print(f"⚠️ Warning: Missing {len(missing_dates)} months in merged dataset.")

    # Run regressions and compute Sharpe ratios
    results = []
    sharpe_results = []
    for col in df.columns:
        if col.startswith("ga_factor") or col.startswith("factor_"):
            # Hedge portfolio
            weighting = "ew" if "ew" in col else "vw"
            if "small" in col:
                size_group = "small"
            elif "big" in col:
                size_group = "big"
            elif "sizeadj" in col:
                size_group = "sizeadj"
            else:
                size_group = "all"
            # Run regressions
            model_df = run_factor_models(merged.copy(), col, ga_choice, weighting, size_group, is_hedge=True)
            if model_df is not None:
                results.append(model_df)
            # Compute Sharpe ratio for excess returns
            merged['ga_excess'] = merged[col] - merged['rf']
            sharpe, mean_ret, std_ret = compute_annualized_sharpe(merged['ga_excess'])
            sharpe_results.append({
                'Portfolio': col,
                'GA Choice': ga_choice,
                'Weighting': weighting,
                'Size Group': size_group,
                'Sharpe Ratio': sharpe,
                'Mean Excess Return (Annualized)': mean_ret,
                'Std Dev (Annualized)': std_ret
            })

        elif col.startswith("single_d1") or col.startswith("single_d10") or col.startswith("d1") or col.startswith("d10"):
            # Individual portfolios: do NOT subtract RF
            weighting = "ew" if "ew" in col else "vw"
            size_group = "small" if "small" in col else "big" if "big" in col else "all"
            model_df = run_factor_models(merged.copy(), col, ga_choice, weighting, size_group, is_hedge=False)
            if model_df is not None:
                results.append(model_df)

    if results:
        regression_results = pd.concat(results, ignore_index=True)
    else:
        regression_results = pd.DataFrame()

    sharpe_df = pd.DataFrame(sharpe_results)
    return regression_results, sharpe_df

# ----------------------------
# Main
# ----------------------------
def main():
    """Main function to run regressions and compute Sharpe ratios for all GA factors."""
    ff_factors = load_ff_factors()

    ga_files = {
        "goodwill_to_sales_lagged": "output/factors/goodwill_to_sales_lagged_factors.csv",
        "goodwill_to_equity_lagged": "output/factors/goodwill_to_equity_lagged_factors.csv",
        "goodwill_to_market_cap_lagged": "output/factors/goodwill_to_market_cap_lagged_factors.csv"
    }

    results_by_ga = {}
    sharpe_by_ga = {}

    for ga_choice, path in ga_files.items():
        logger.info(f"Processing {ga_choice} from {path}")
        print(f"\n🔍 Processing GA metric: {ga_choice}")
        sheet_key = {
            "goodwill_to_sales_lagged": "GA1",
            "goodwill_to_equity_lagged": "GA2",
            "goodwill_to_market_cap_lagged": "GA3"
        }[ga_choice]

        regression_results, sharpe_results = process_ga_file(path, ff_factors, ga_choice)
        results_by_ga[sheet_key] = regression_results
        sharpe_by_ga[sheet_key] = sharpe_results

    # Export regression results to Excel
    output_path = "output/ga_factor_regression_results_monthly.xlsx"
    os.makedirs("output", exist_ok=True)

    with pd.ExcelWriter(output_path, engine='xlsxwriter') as writer:
        for sheet, df in results_by_ga.items():
            if df is not None and not df.empty:
                num_cols = [c for c in df.columns if df[c].dtype != 'object']
                df[num_cols] = df[num_cols].round(4)
                df.to_excel(writer, sheet_name=sheet, index=False)
                print(f"📝 Wrote regression results to sheet {sheet}")
            else:
                pd.DataFrame().to_excel(writer, sheet_name=sheet, index=False)
                print(f"⚠️ No regression results for sheet {sheet}. Created empty sheet.")

        # Export to CSV as well (one CSV per GA sheet)
        for sheet, df in results_by_ga.items():
            if df is not None and not df.empty:
                csv_path = f"output/{sheet.lower()}_regression_results.csv"
                df.to_csv(csv_path, index=False)
                logger.info(f"Saved regression CSV: {csv_path}")

    # Export Sharpe ratios to Excel
    sharpe_output_path = "output/ga_factor_sharpe_results.xlsx"
    with pd.ExcelWriter(sharpe_output_path, engine='xlsxwriter') as writer:
        for sheet, df in sharpe_by_ga.items():
            if df is not None and not df.empty:
                num_cols = [c for c in df.columns if df[c].dtype != 'object']
                df[num_cols] = df[num_cols].round(4)
                df.to_excel(writer, sheet_name=sheet, index=False)
                print(f"📝 Wrote Sharpe ratios to sheet {sheet}")
            else:
                pd.DataFrame().to_excel(writer, sheet_name=sheet, index=False)
                print(f"⚠️ No Sharpe ratios for sheet {sheet}. Created empty sheet.")

        # Export to CSV as well
        for sheet, df in sharpe_by_ga.items():
            if df is not None and not df.empty:
                csv_path = f"output/{sheet.lower()}_sharpe_results.csv"
                df.to_csv(csv_path, index=False)
                logger.info(f"Saved Sharpe CSV: {csv_path}")

    print(f"\n✅ Regression results saved to {output_path}")
    print(f"✅ Sharpe ratios saved to {sharpe_output_path}")

# ----------------------------
# Run it
# ----------------------------
if __name__ == "__main__":
    main()