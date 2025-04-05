import pandas as pd
import os
import statsmodels.api as sm
import numpy as np
import logging

EXPORT_LATEX = False

# ----------------------------
# Logging
# ----------------------------
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger()

# ----------------------------
# Load Fama-French Factors
# ----------------------------
def load_ff_factors():
    ff_path = "data/FamaFrench_factors_with_momentum.csv"
    if not os.path.exists(ff_path):
        raise FileNotFoundError(f"Missing file: {ff_path}")

    ff = pd.read_csv(ff_path, parse_dates=['date'])
    ff['date'] = pd.to_datetime(ff['date']) + pd.offsets.MonthEnd(0)
    ff.set_index('date', inplace=True)
    ff.columns = ff.columns.str.lower()

    required = ['mkt_rf', 'smb', 'hml', 'rmw', 'cma', 'mom', 'rf']
    missing = [col for col in required if col not in ff.columns]
    if missing:
        raise ValueError(f"Missing columns in Fama-French file: {missing}")

    return ff

# ----------------------------
# Run Regressions
# ----------------------------
def run_factor_models(df, column, ga_choice, weighting, size_group, is_hedge=False):
    logger.info(f"Running regression: {column}")
    
    if is_hedge:
        df['ga_excess'] = df[column] - df['rf']
    else:
        df['ga_excess'] = df[column]  # no rf deduction for individual portfolios

    if df['ga_excess'].isna().mean() > 0.5:
        logger.warning(f"Too many NaNs in {column}. Skipping.")
        return None

    models = {
        "CAPM": ["mkt_rf"],
        "FF3": ["mkt_rf", "smb", "hml"],
        "FF5": ["mkt_rf", "smb", "hml", "rmw", "cma"],
        "FF5+MOM": ["mkt_rf", "smb", "hml", "rmw", "cma", "mom"]
    }

    results = []

    for model_name, factors in models.items():
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

    return pd.DataFrame(results)

# ----------------------------
# Process One GA File
# ----------------------------
def process_ga_file(filepath, ff_factors, ga_choice):
    if not os.path.exists(filepath):
        logger.error(f"Missing GA file: {filepath}")
        return None

    df = pd.read_csv(filepath, parse_dates=['crsp_date'])
    df['crsp_date'] = pd.to_datetime(df['crsp_date']) + pd.offsets.MonthEnd(0)
    df.set_index('crsp_date', inplace=True)
    df.columns = df.columns.str.lower()

    ff = ff_factors[ff_factors.index.isin(df.index)]
    merged = df.join(ff, how='inner').sort_index()

    results = []
    for col in df.columns:
        if col.startswith("ga_factor") or col.startswith("factor_"):
            weighting = "ew" if "ew" in col else "vw"
            if "small" in col:
                size_group = "small"
            elif "big" in col:
                size_group = "big"
            elif "sizeadj" in col:
                size_group = "sizeadj"
            else:
                size_group = "all"
            model_df = run_factor_models(merged.copy(), col, ga_choice, weighting, size_group, is_hedge=True)
            if model_df is not None:
                results.append(model_df)

        elif col.startswith("single_d1") or col.startswith("single_d10") or col.startswith("d1") or col.startswith("d10"):
            # Individual portfolios: do NOT subtract RF
            weighting = "ew" if "ew" in col else "vw"
            size_group = "small" if "small" in col else "big" if "big" in col else "all"
            model_df = run_factor_models(merged.copy(), col, ga_choice, weighting, size_group, is_hedge=False)
            if model_df is not None:
                results.append(model_df)

    if results:
        return pd.concat(results, ignore_index=True)
    else:
        return pd.DataFrame()

# ----------------------------
# Main
# ----------------------------
def main():
    ff_factors = load_ff_factors()

    ga_files = {
        "goodwill_to_sales_lagged": "output/factors/goodwill_to_sales_lagged_factors.csv",
        "goodwill_to_equity_lagged": "output/factors/goodwill_to_equity_lagged_factors.csv",
        "goodwill_to_market_cap_lagged": "output/factors/goodwill_to_market_cap_lagged_factors.csv"
    }

    results_by_ga = {}

    for ga_choice, path in ga_files.items():
        logger.info(f"Processing {ga_choice} from {path}")
        sheet_key = {
            "goodwill_to_sales_lagged": "GA1",
            "goodwill_to_equity_lagged": "GA2",
            "goodwill_to_market_cap_lagged": "GA3"
        }[ga_choice]

        df_results = process_ga_file(path, ff_factors, ga_choice)
        results_by_ga[sheet_key] = df_results

    # Export to Excel
    output_path = "output/ga_factor_regression_results_monthly.xlsx"
    os.makedirs("output", exist_ok=True)

    with pd.ExcelWriter(output_path, engine='xlsxwriter') as writer:
        for sheet, df in results_by_ga.items():
            if df is not None and not df.empty:
                num_cols = [c for c in df.columns if df[c].dtype != 'object']
                df[num_cols] = df[num_cols].round(4)
                df.to_excel(writer, sheet_name=sheet, index=False)
            else:
                pd.DataFrame().to_excel(writer, sheet_name=sheet, index=False)

        # Export to CSV as well (one CSV per GA sheet)
    for sheet, df in results_by_ga.items():
        if df is not None and not df.empty:
            csv_path = f"output/{sheet.lower()}_regression_results.csv"
            df.to_csv(csv_path, index=False)
            logger.info(f"Saved CSV: {csv_path}")

    print(f"\n✅ Regression results saved to {output_path}")

# ----------------------------
# Run it
# ----------------------------
if __name__ == "__main__":
    main()