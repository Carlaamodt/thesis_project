import pandas as pd
import os
import statsmodels.api as sm
import numpy as np
import logging

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# ----------------------------
# Load Fama-French Factors
# ----------------------------
def load_ff_factors(filepath: str) -> pd.DataFrame:
    """
    Load Fama-French factors with momentum from CSV.

    Args:
        filepath (str): Path to the FF factors CSV file.

    Returns:
        pd.DataFrame: FF factors with date index.
    """
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"Missing FF factors file: {filepath}")

    logger.info(f"Loading Fama-French factors from {filepath}...")
    ff = pd.read_csv(filepath, parse_dates=['date'])
    ff['date'] = pd.to_datetime(ff['date']) + pd.offsets.MonthEnd(0)
    ff.set_index('date', inplace=True)
    ff.columns = ff.columns.str.lower()

    required = ['mkt_rf', 'smb', 'hml', 'rmw', 'cma', 'mom', 'rf']
    missing = [col for col in required if col not in ff.columns]
    if missing:
        raise ValueError(f"Missing columns in FF factors file: {missing}")

    logger.info(f"Loaded FF factors: {len(ff)} months, from {ff.index.min()} to {ff.index.max()}")
    return ff

# ----------------------------
# Load GA Factor
# ----------------------------
def load_ga_factor(filepath: str, column: str = 'ga_factor_vw') -> pd.DataFrame:
    """
    Load the GA factor from CSV.

    Args:
        filepath (str): Path to the GA factor CSV file.
        column (str): Specific GA factor column to use (default: 'ga_factor_vw').

    Returns:
        pd.DataFrame: GA factor with date index.
    """
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"Missing GA factor file: {filepath}")

    logger.info(f"Loading GA factor from {filepath}...")
    df = pd.read_csv(filepath, parse_dates=['crsp_date'])
    df['crsp_date'] = pd.to_datetime(df['crsp_date']) + pd.offsets.MonthEnd(0)
    df.set_index('crsp_date', inplace=True)
    df.columns = df.columns.str.lower()

    if column not in df.columns:
        raise ValueError(f"Column '{column}' not found in GA factor file")

    logger.info(f"Loaded GA factor '{column}': {len(df)} months, from {df.index.min()} to {df.index.max()}")
    return df[[column]]

# ----------------------------
# Run Rolling Regression
# ----------------------------
def run_rolling_regression(df: pd.DataFrame, ga_column: str = 'ga_factor_vw', window: int = 36) -> pd.DataFrame:
    """
    Perform rolling 36-month regression of GA factor on FF5+MOM factors.

    Args:
        df (pd.DataFrame): Merged DataFrame with GA factor and FF factors.
        ga_column (str): Column name of the GA factor to regress (default: 'ga_factor_vw').
        window (int): Size of the rolling window in months (default: 36).

    Returns:
        pd.DataFrame: Rolling regression results with alpha, betas, and t-stats.
    """
    factors = ['mkt_rf', 'smb', 'hml', 'rmw', 'cma', 'mom']
    results = []

    # Ensure dates are sorted
    df = df.sort_index()
    dates = df.index.unique()

    # Rolling window
    for i in range(window, len(dates) + 1):
        window_df = df.iloc[i - window:i]
        end_date = window_df.index[-1]
        if len(window_df) < window:
            logger.warning(f"Window ending {end_date} has {len(window_df)} months (< {window})")
            continue

        y = window_df[ga_column]
        X = sm.add_constant(window_df[factors], has_constant='add')

        if y.isna().mean() > 0.5 or X.isna().any().mean() > 0.5:
            logger.warning(f"Skipping window ending {end_date}: Too many NaNs")
            continue

        model = sm.OLS(y, X, missing='drop').fit(cov_type='HAC', cov_kwds={'maxlags': 3})

        res = {
            'End Date': end_date,
            'Alpha': model.params.get('const', np.nan),
            'Alpha t-stat': model.tvalues.get('const', np.nan),
            'MKT-RF Beta': model.params.get('mkt_rf', np.nan),
            'MKT-RF t-stat': model.tvalues.get('mkt_rf', np.nan),
            'SMB Beta': model.params.get('smb', np.nan),
            'SMB t-stat': model.tvalues.get('smb', np.nan),
            'HML Beta': model.params.get('hml', np.nan),
            'HML t-stat': model.tvalues.get('hml', np.nan),
            'RMW Beta': model.params.get('rmw', np.nan),
            'RMW t-stat': model.tvalues.get('rmw', np.nan),
            'CMA Beta': model.params.get('cma', np.nan),
            'CMA t-stat': model.tvalues.get('cma', np.nan),
            'MOM Beta': model.params.get('mom', np.nan),
            'MOM t-stat': model.tvalues.get('mom', np.nan),
            'R-squared': model.rsquared,
            'Obs': int(model.nobs)
        }
        results.append(res)

    return pd.DataFrame(results)

# ----------------------------
# Main Function
# ----------------------------
def main():
    """
    Main function to run rolling regression for all three GA factors (GA1, GA2, GA3).
    """
    # File paths
    ff_filepath = "data/FamaFrench_factors_with_momentum.csv"
    ga_files = {
        "GA1_vw": "output/industry_adjusted/factors/goodwill_to_sales_lagged_industry_adj_factors.csv",
        "GA2_vw": "output/industry_adjusted/factors/goodwill_to_equity_lagged_industry_adj_factors.csv",
        "GA3_vw": "output/industry_adjusted/factors/goodwill_to_shareholder_equity_cap_lagged_industry_adj_factors.csv"
    }

    # Load FF factors once
    ff_factors = load_ff_factors(ff_filepath)

    # Process each GA factor
    results_dict = {}
    for ga_name, ga_filepath in ga_files.items():
        logger.info(f"\nProcessing {ga_name} from {ga_filepath}")
        
        # Load GA factor
        ga_df = load_ga_factor(ga_filepath, column='ga_factor_vw')
        
        # Merge with FF factors
        merged = ga_df.join(ff_factors, how='inner')
        merged = merged[merged.index.year <= 2023]  # Cap at 2023, start from earliest available
        logger.info(f"Merged dataset for {ga_name}: {len(merged)} months, from {merged.index.min()} to {merged.index.max()}")

        # Check for missing values
        missing = merged.isna().sum()
        if missing.sum() > 0:
            logger.warning(f"Missing values in merged dataset for {ga_name}:\n{missing[missing > 0]}")

        # Run rolling regression
        rolling_results = run_rolling_regression(merged, ga_column='ga_factor_vw', window=36)
        results_dict[ga_name] = rolling_results

    # Save results to Excel
    output_dir = "output/industry_adjusted/rolling"
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, "rolling_regression_results.xlsx")
    
    with pd.ExcelWriter(output_path, engine='xlsxwriter') as writer:
        for ga_name, df in results_dict.items():
            if not df.empty:
                df.to_excel(writer, sheet_name=ga_name, index=False)
                logger.info(f"Wrote {ga_name} results to sheet {ga_name}")
            else:
                pd.DataFrame().to_excel(writer, sheet_name=ga_name, index=False)
                logger.warning(f"No results for {ga_name}. Wrote empty sheet.")

    # Save CSV backups
    for ga_name, df in results_dict.items():
        if not df.empty:
            csv_path = os.path.join(output_dir, f"{ga_name.lower()}_rolling_regression.csv")
            df.to_csv(csv_path, index=False)
            logger.info(f"Saved {ga_name} CSV to {csv_path}")

    logger.info(f"All rolling regression results saved to {output_path}")

# ----------------------------
# Run the Script
# ----------------------------
if __name__ == "__main__":
    main()