import pandas as pd
import numpy as np
import os
import statsmodels.api as sm
import matplotlib.pyplot as plt
import logging

# --- Configuration ---
ROLLING_WINDOW = 36  # months
FACTORS = ['mkt_rf', 'smb', 'hml', 'rmw', 'cma', 'mom']
GA_COLUMNS = ['ga_factor_ew', 'ga_factor_vw']
DATA_PATH = 'output/industry_adjusted/factors/goodwill_to_equity_lagged_industry_adj_factors.csv'
FF_PATH = 'data/FamaFrench_factors_with_momentum.csv'
SAVE_DIR = 'output/industry_adjusted/regression'
os.makedirs(SAVE_DIR, exist_ok=True)

# --- Logging ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger()

# --- Load Data ---
logger.info("Loading GA factor returns...")
ga_df = pd.read_csv(DATA_PATH, parse_dates=['crsp_date'])
ga_df.set_index('crsp_date', inplace=True)

df_factors = pd.read_csv(FF_PATH, parse_dates=['date'])
df_factors['date'] = pd.to_datetime(df_factors['date']) + pd.offsets.MonthEnd(0)
df_factors.set_index('date', inplace=True)
df_factors.columns = df_factors.columns.str.lower()

# --- Merge Datasets ---
df = ga_df.join(df_factors, how='inner')
df = df[df.index.year >= 2004]

# --- Rolling Regressions ---
def run_rolling_regression(target_col):
    logger.info(f"Running rolling regression for {target_col}...")
    results = []
    for i in range(ROLLING_WINDOW, len(df)):
        window_df = df.iloc[i - ROLLING_WINDOW:i].dropna(subset=[target_col] + FACTORS)
        if len(window_df) < ROLLING_WINDOW:
            continue
        X = sm.add_constant(window_df[FACTORS])
        y = window_df[target_col]
        model = sm.OLS(y, X).fit()
        result = {
            'date': df.index[i],
            'alpha': model.params['const'],
            'alpha_tstat': model.tvalues['const']
        }
        for factor in FACTORS:
            result[f'{factor}_beta'] = model.params.get(factor, np.nan)
        results.append(result)
    return pd.DataFrame(results)

# --- Process Each GA Column ---
for col in GA_COLUMNS:
    out_df = run_rolling_regression(col)
    out_df.set_index('date', inplace=True)
    csv_path = os.path.join(SAVE_DIR, f'rolling_{col}.csv')
    out_df.to_csv(csv_path)
    logger.info(f"Saved rolling regression results to {csv_path}")

    # Plot alpha
    plt.figure(figsize=(12, 4))
    plt.plot(out_df.index, out_df['alpha'] * 12, label='Annualized Alpha')
    plt.axhline(0, color='black', linewidth=0.8, linestyle='--')
    plt.title(f"Rolling Annualized Alpha ({col})")
    plt.xlabel("Date")
    plt.ylabel("Alpha")
    plt.legend()
    plt.tight_layout()
    plot_path = os.path.join(SAVE_DIR, f'rolling_alpha_{col}.png')
    plt.savefig(plot_path)
    plt.close()
    logger.info(f"Saved alpha plot to {plot_path}")
