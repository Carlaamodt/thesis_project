import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import statsmodels.api as sm
import os
import logging
import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Set output directory for File 7 outputs
output_dir = "file 7 output"
os.makedirs(output_dir, exist_ok=True)
subdirs = ['deciles', 'factors', 'industry', 'plots']
for subdir in subdirs:
    os.makedirs(os.path.join(output_dir, subdir), exist_ok=True)

# Set seaborn style for better visuals
sns.set(style="whitegrid")

########################################
### Load FF48 Mapping ###
########################################

def load_ff48_mapping_from_excel(filepath: str) -> dict:
    """
    Load Fama-French 48 industry classification mapping from an Excel file.
    """
    logger.info(f"Loading Fama-French 48 industry classification from {filepath}")
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"Excel file not found at {filepath}")
    
    df = pd.read_excel(filepath)
    if not all(col in df.columns for col in ['Industry number', 'Industry code', 'Industry description']):
        raise ValueError("Excel file must contain columns: 'Industry number', 'Industry code', 'Industry description'")
    
    df['Industry number'] = df['Industry number'].ffill().astype(int)
    ff48_mapping = {}
    for _, row in df.iterrows():
        code = row['Industry code']
        industry_number = int(row['Industry number'])
        if isinstance(code, str) and '-' in code:
            try:
                start, end = map(int, code.strip().split('-'))
                ff48_mapping[range(start, end + 1)] = industry_number
            except ValueError:
                logger.warning(f"Skipping invalid SIC range: {code}")
    logger.info(f"Loaded {len(ff48_mapping)} SIC ranges into FF48 mapping.")
    return ff48_mapping

def map_sic_to_ff48(sic):
    """Map a 4-digit SIC code to Fama-French 48 industry."""
    if pd.isna(sic):
        sic_int = 0
    else:
        try:
            sic_int = int(float(sic))
        except ValueError:
            logger.warning(f"Invalid SIC value: {sic}, setting to 0")
            sic_int = 0

    for sic_range, industry in FF48_MAPPING.items():
        if sic_int in sic_range:
            return industry
    return 48  # Default to "Other" if no match

########################################
### Load Data ###
########################################

def load_data(directory="data/", ga_choice="goodwill_to_equity_lagged"):
    """
    Load all necessary datasets for analysis.
    """
    logger.info(f"Loading data for {ga_choice}...")
    
    # Processed data from 2_data_processing.py
    proc_filepath = os.path.join(directory, "processed_data.csv")
    required_cols = [
        'permno', 'crsp_date', 'ret', 'market_cap', 'sich', 'siccd',
        'goodwill_to_sales_lagged', 'goodwill_to_equity_lagged', 'goodwill_to_shareholder_equity_cap_lagged'
    ]
    dtypes = {'sich': 'str', 'siccd': 'str'}
    if not os.path.exists(proc_filepath):
        logger.error(f"Processed data file not found: {proc_filepath}")
        raise FileNotFoundError(f"Processed data file not found: {proc_filepath}")
    proc_df = pd.read_csv(proc_filepath, parse_dates=['crsp_date'], usecols=required_cols, 
                          dtype=dtypes, low_memory=False)
    proc_df = proc_df.drop_duplicates(subset=['permno', 'crsp_date'])
    
    # Compute year for grouping
    proc_df['year'] = proc_df['crsp_date'].dt.year
    logger.info(f"Processed data rows: {len(proc_df)}, unique permno: {proc_df['permno'].nunique()}")
    logger.info(f"Columns in processed data: {list(proc_df.columns)}")

    # Decile data from 5_factor_construction_industry.py
    decile_filepath = os.path.join("output", "industry_adjusted", "deciles", f"{ga_choice}_industry_adj_deciles.csv")
    if not os.path.exists(decile_filepath):
        decile_filepath = os.path.join("output", "deciles", f"{ga_choice}_deciles.csv")
        if not os.path.exists(decile_filepath):
            logger.warning(f"Decile file not found in either path: {os.path.join('output', 'industry_adjusted', 'deciles')} or {os.path.join('output', 'deciles')}. Will assign deciles from raw data.")
            decile_df = pd.DataFrame()
        else:
            decile_df = pd.read_csv(decile_filepath, parse_dates=['crsp_date'])
            decile_df = decile_df[decile_df['decile'].between(1, 10)]
            logger.info(f"Decile data loaded from: {decile_filepath}, rows: {len(decile_df)}, unique permno: {decile_df['permno'].nunique()}")
            logger.info(f"Columns in decile data: {list(decile_df.columns)}")
    else:
        decile_df = pd.read_csv(decile_filepath, parse_dates=['crsp_date'])
        decile_df = decile_df[decile_df['decile'].between(1, 10)]
        logger.info(f"Decile data loaded from: {decile_filepath}, rows: {len(decile_df)}, unique permno: {decile_df['permno'].nunique()}")
        logger.info(f"Columns in decile data: {list(decile_df.columns)}")

    # FF factors from 3_download_fama_french.py
    ff_filepath = os.path.join(directory, "FamaFrench_factors_with_momentum.csv")
    if not os.path.exists(ff_filepath):
        logger.error(f"Fama-French factors file not found: {ff_filepath}")
        raise FileNotFoundError(f"Fama-French factors file not found: {ff_filepath}")
    ff_df = pd.read_csv(ff_filepath, parse_dates=['date'])
    ff_df['date'] = ff_df['date'] + pd.offsets.MonthEnd(0)
    ff_df.columns = ff_df.columns.str.lower()
    logger.info(f"FF factors rows: {len(ff_df)}")

    # GA factors from 5_factor_construction_industry.py
    ga_factors_path = os.path.join("output", "industry_adjusted", "factors", f"{ga_choice}_industry_adj_factors.csv")
    if not os.path.exists(ga_factors_path):
        ga_factors_path = os.path.join("output", "factors", f"{ga_choice}_factors.csv")
        if not os.path.exists(ga_factors_path):
            logger.warning(f"GA factors file not found in either path: {os.path.join('output', 'industry_adjusted', 'factors')} or {os.path.join('output', 'factors')}. Diagnostics may be limited.")
            ga_factors_df = pd.DataFrame()
        else:
            ga_factors_df = pd.read_csv(ga_factors_path, parse_dates=['crsp_date'])
            ga_factors_df.rename(columns={'crsp_date': 'date'}, inplace=True)
            logger.info(f"GA factors loaded from: {ga_factors_path}, rows: {len(ga_factors_df)}")
    else:
        ga_factors_df = pd.read_csv(ga_factors_path, parse_dates=['crsp_date'])
        ga_factors_df.rename(columns={'crsp_date': 'date'}, inplace=True)
        logger.info(f"GA factors loaded from: {ga_factors_path}, rows: {len(ga_factors_df)}")

    return proc_df, decile_df, ff_df, ga_factors_df

########################################
### Decile Analysis ###
########################################

def decile_analysis(proc_df, decile_df, ga_choice="goodwill_to_equity_lagged"):
    """
    Analyze portfolio characteristics across deciles.
    """
    logger.info(f"Analyzing {ga_choice} deciles...")
    
    if decile_df.empty:
        logger.warning(f"No decile data—assigning from raw data.")
        df = proc_df.copy()
        if df[ga_choice].nunique() < 10:
            logger.warning(f"Too few unique {ga_choice} values ({df[ga_choice].nunique()})—skipping.")
            return
        df['decile'] = pd.qcut(df[ga_choice], 10, labels=False, duplicates='drop') + 1
    else:
        df = decile_df.merge(proc_df[['permno', 'crsp_date', ga_choice, 'market_cap', 'ret', 'year']], 
                             on=['permno', 'crsp_date'], how='left')
    
    logger.info(f"Columns in df after setup: {list(df.columns)}")
    
    # Ensure ga_choice column is present
    if ga_choice not in df.columns:
        logger.error(f"Column {ga_choice} not found in df. Available columns: {list(df.columns)}")
        raise KeyError(f"Column {ga_choice} not found in df")
    
    # GA stats per decile
    ga_stats = df.groupby(['year', 'decile']).agg(
        avg_ga=(ga_choice, 'mean'),
        median_ga=(ga_choice, 'median'),
        std_ga=(ga_choice, 'std')
    ).reset_index()
    filepath = os.path.join(output_dir, 'deciles', f"{ga_choice}_ga_stats_per_decile.xlsx")
    ga_stats.to_excel(filepath, index=False)
    logger.info(f"Saved GA stats: {filepath}")

    # Market equity (ME) stats
    me_stats = df.groupby(['year', 'decile']).agg(
        avg_me=('market_cap', 'mean'),
        median_me=('market_cap', 'median'),
        std_me=('market_cap', 'std')
    ).reset_index()
    filepath = os.path.join(output_dir, 'deciles', f"{ga_choice}_market_cap_per_decile.xlsx")
    me_stats.to_excel(filepath, index=False)
    logger.info(f"Saved market cap stats: {filepath}")

    # Combined portfolio characteristics
    combined_stats = pd.merge(ga_stats, me_stats, on=['year', 'decile'], how='left')
    filepath = os.path.join(output_dir, 'deciles', f"{ga_choice}_portfolio_characteristics.xlsx")
    combined_stats.to_excel(filepath, index=False)
    logger.info(f"Saved portfolio characteristics: {filepath}")

    # Visualization: Average GA per decile over time
    plt.figure(figsize=(12, 6))
    sns.lineplot(data=ga_stats, x='year', y='avg_ga', hue='decile', palette='tab10')
    plt.title(f'Average {ga_choice} per Decile Over Time')
    plt.xlabel('Year')
    plt.ylabel(f'Average {ga_choice}')
    plt.legend(title='Decile')
    filepath = os.path.join(output_dir, 'plots', f"{ga_choice}_avg_ga_per_decile_over_time.png")
    plt.savefig(filepath, dpi=300, bbox_inches='tight')
    plt.close()
    logger.info(f"Saved plot: {filepath}")

    # Visualization: Distribution of Returns per Decile
    plt.figure(figsize=(12, 6))
    sns.boxplot(data=df, x='decile', y='ret')
    plt.title(f'Distribution of Returns per {ga_choice} Decile')
    plt.xlabel('Decile')
    plt.ylabel('Monthly Return')
    filepath = os.path.join(output_dir, 'plots', f"{ga_choice}_return_distribution_per_decile.png")
    plt.savefig(filepath, dpi=300, bbox_inches='tight')
    plt.close()
    logger.info(f"Saved plot: {filepath}")

########################################
### GA Factor Diagnostics ###
########################################

def ga_factor_diagnostics(proc_df, ff_df, ga_factors_df, ga_choice="goodwill_to_equity_lagged"):
    """
    Perform diagnostics on the GA factor, including trends and correlations.
    """
    logger.info(f"{ga_choice} factor diagnostics...")
    
    # GA trend over time
    ga_trend = proc_df.groupby('year').agg(
        total_obs=(ga_choice, 'count'),
        ga_zero=(ga_choice, lambda x: (x == 0).sum()),
        ga_nonzero=(ga_choice, lambda x: (x != 0).sum()),
        mean_ga=(ga_choice, 'mean'),
        median_ga=(ga_choice, 'median'),
        std_ga=(ga_choice, 'std')
    ).reset_index()
    filepath = os.path.join(output_dir, 'factors', f"{ga_choice}_zero_nonzero_trend.xlsx")
    ga_trend.to_excel(filepath, index=False)
    logger.info(f"Saved GA trend: {filepath}")
    
    plt.figure(figsize=(10, 6))
    sns.lineplot(data=ga_trend, x='year', y='mean_ga', label=f'Mean {ga_choice}')
    sns.lineplot(data=ga_trend, x='year', y='median_ga', label=f'Median {ga_choice}')
    plt.legend()
    plt.title(f'Mean and Median {ga_choice} Over Time')
    plt.xlabel('Year')
    plt.ylabel(ga_choice)
    filepath = os.path.join(output_dir, 'plots', f"{ga_choice}_mean_median_trend.png")
    plt.savefig(filepath, dpi=300, bbox_inches='tight')
    plt.close()
    logger.info(f"Saved GA trend plot: {filepath}")

    if ga_factors_df.empty:
        logger.warning(f"No GA factors data—skipping factor-specific diagnostics.")
        return
    
    # Prepare equal-weighted and value-weighted GA factors
    ga_eq = ga_factors_df[['date', 'ga_factor_ew']].rename(columns={'ga_factor_ew': 'ga_factor'})
    ga_val = ga_factors_df[['date', 'ga_factor_vw']].rename(columns={'ga_factor_vw': 'ga_factor'})

    # Cumulative returns
    ga_eq['cum_return'] = (1 + ga_eq['ga_factor']).cumprod()
    ga_val['cum_return'] = (1 + ga_val['ga_factor']).cumprod()
    
    merged = pd.merge(ga_eq[['date', 'ga_factor']], ff_df, on='date', how='inner')
    merged['mkt_cum_return'] = (1 + merged['mkt_rf']).cumprod() if 'mkt_rf' in merged.columns else np.nan
    
    plt.figure(figsize=(10, 6))
    plt.plot(ga_eq['date'], ga_eq['cum_return'], label='Equal-weighted GA (Low - High)')
    plt.plot(ga_val['date'], ga_val['cum_return'], label='Value-weighted GA (Low - High)')
    if merged['mkt_cum_return'].notna().any():
        plt.plot(merged['date'], merged['mkt_cum_return'], label='Market', linestyle='--', color='green')
    plt.title(f'{ga_choice} Cumulative Returns vs. Market (Low GA - High GA)')
    plt.xlabel('Date')
    plt.ylabel('Cumulative Return')
    plt.legend()
    plt.grid(True, alpha=0.7)
    filepath = os.path.join(output_dir, 'plots', f"{ga_choice}_cumulative_returns.png")
    plt.savefig(filepath, dpi=300, bbox_inches='tight')
    plt.close()
    logger.info(f"Saved cumulative returns plot: {filepath}")

    # Correlation with FF factors
    ff_cols = ['mkt_rf', 'smb', 'hml', 'rmw', 'cma', 'mom']
    available_ff_cols = [col for col in ff_cols if col in merged.columns]
    corr_matrix = merged[['ga_factor'] + available_ff_cols].corr()
    filepath = os.path.join(output_dir, 'factors', f"{ga_choice}_correlation_matrix.xlsx")
    corr_matrix.to_excel(filepath)
    logger.info(f"Saved correlation matrix: {filepath}")

    plt.figure(figsize=(8, 6))
    sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', center=0, fmt='.2f')
    plt.title(f"{ga_choice} Correlation with FF Factors (Low GA - High GA)")
    filepath = os.path.join(output_dir, 'plots', f"{ga_choice}_correlation_heatmap.png")
    plt.savefig(filepath, dpi=300, bbox_inches='tight')
    plt.close()
    logger.info(f"Saved correlation heatmap: {filepath}")

    # Factor summary
    ga_summary = pd.DataFrame({
        'Equal Mean': [ga_eq['ga_factor'].mean()],
        'Equal Std': [ga_eq['ga_factor'].std()],
        'Equal Sharpe (Annualized)': [(ga_eq['ga_factor'].mean() * 12) / (ga_eq['ga_factor'].std() * np.sqrt(12))],
        'Value Mean': [ga_val['ga_factor'].mean()],
        'Value Std': [ga_val['ga_factor'].std()],
        'Value Sharpe (Annualized)': [(ga_val['ga_factor'].mean() * 12) / (ga_val['ga_factor'].std() * np.sqrt(12))]
    })
    filepath = os.path.join(output_dir, 'factors', f"{ga_choice}_factor_summary.xlsx")
    ga_summary.to_excel(filepath, index=False)
    logger.info(f"Saved factor summary: {filepath}")

    # Volatility Plot
    plt.figure(figsize=(10, 6))
    plt.plot(ga_eq['date'], ga_eq['ga_factor'].rolling(window=12).std() * np.sqrt(12), label='Equal-weighted Volatility')
    plt.plot(ga_val['date'], ga_val['ga_factor'].rolling(window=12).std() * np.sqrt(12), label='Value-weighted Volatility')
    plt.title(f'Annualized Volatility of {ga_choice} Factor (Low GA - High GA)')
    plt.xlabel('Date')
    plt.ylabel('Annualized Volatility')
    plt.legend()
    plt.grid(True, alpha=0.7)
    filepath = os.path.join(output_dir, 'plots', f"{ga_choice}_factor_volatility.png")
    plt.savefig(filepath, dpi=300, bbox_inches='tight')
    plt.close()
    logger.info(f"Saved factor volatility plot: {filepath}")

########################################
### Market Return Plot + Decile Alphas ###
########################################

def market_return_reference_plot(ff_df):
    """
    Plot the cumulative market return as a reference.
    """
    if 'mkt_rf' not in ff_df.columns or 'rf' not in ff_df.columns:
        logger.warning("Missing market or risk-free columns in FF data.")
        return

    ff_df = ff_df.copy()
    ff_df['mkt'] = ff_df['mkt_rf'] + ff_df['rf']  # Reconstruct actual return
    ff_df['mkt_cum_return'] = (1 + ff_df['mkt']).cumprod()

    plt.figure(figsize=(10, 6))
    plt.plot(ff_df['date'], ff_df['mkt_cum_return'], label='Market Return (Actual)', linestyle='-')
    plt.title("Market Cumulative Return (Not Risk-Adjusted)")
    plt.xlabel("Date")
    plt.ylabel("Cumulative Return")
    plt.grid(True, alpha=0.6)
    plt.legend()
    filepath = os.path.join(output_dir, 'plots', "market_actual_cumulative_return.png")
    plt.savefig(filepath, dpi=300, bbox_inches='tight')
    plt.close()
    logger.info(f"Saved market cumulative return plot: {filepath}")

def decile_alpha_regressions(proc_df, decile_df, ff_df, ga_choice="goodwill_to_equity_lagged"):
    """
    Run CAPM regressions for each decile to estimate alphas.
    """
    logger.info(f"Running alpha regressions for each decile of {ga_choice}...")

    results = []

    for weighting in ["equal", "value"]:
        for decile in range(1, 11):
            sub_df = decile_df[decile_df['decile'] == decile].copy() if not decile_df.empty else proc_df[proc_df['decile'] == decile].copy()
            sub_df = sub_df.merge(proc_df[['permno', 'crsp_date', 'ret', 'market_cap']], 
                                  on=['permno', 'crsp_date'], how='left')
            sub_df = sub_df[sub_df['ret'].notna() & sub_df['market_cap'].notna()]
            
            if sub_df.empty:
                logger.warning(f"Skipping decile {decile} ({weighting}) — no data for regression.")
                continue

            if weighting == "equal":
                ret_series = sub_df.groupby('crsp_date')['ret'].mean()
            else:
                ret_series = sub_df.groupby('crsp_date').apply(
                    lambda x: np.average(x['ret'], weights=x['market_cap']) if len(x) >= 5 else np.nan
                )

            ret_series.name = 'ret'
            ret_df = ret_series.dropna().reset_index()

            if 'index' in ret_df.columns:
                ret_df.rename(columns={'index': 'crsp_date'}, inplace=True)

            if 'crsp_date' not in ret_df.columns:
                logger.error(f"crsp_date column not found in return data for decile {decile} ({weighting})")
                continue

            merged = pd.merge(ret_df, ff_df, left_on='crsp_date', right_on='date', how='inner')
            if 'rf' not in merged.columns:
                logger.error("Missing risk-free rate in FF data.")
                continue

            merged['ga_factor_excess'] = merged['ret'] - merged['rf']

            X = sm.add_constant(merged[['mkt_rf']], has_constant='add')
            y = merged['ga_factor_excess']
            model = sm.OLS(y, X, missing='drop').fit(cov_type='HAC', cov_kwds={'maxlags': 3})

            results.append({
                'Decile': decile,
                'Weighting': weighting,
                'Alpha (Annualized)': model.params.get('const', np.nan) * 12,
                'Alpha t-stat': model.tvalues.get('const', np.nan),
                'Alpha p-value': model.pvalues.get('const', np.nan),
                'R-squared': model.rsquared,
                'Observations': int(model.nobs)
            })

    if not results:
        logger.warning(f"No decile alpha regressions completed for {ga_choice}")
        return

    results_df = pd.DataFrame(results)
    filepath = os.path.join(output_dir, 'deciles', f"decile_alpha_regressions_{ga_choice}.xlsx")
    results_df.to_excel(filepath, index=False)
    logger.info(f"Saved decile alpha regressions: {filepath}")

    # Visualization: Alpha per Decile
    plt.figure(figsize=(10, 6))
    sns.barplot(data=results_df, x='Decile', y='Alpha (Annualized)', hue='Weighting')
    plt.title(f'Annualized CAPM Alpha per {ga_choice} Decile')
    plt.xlabel('Decile')
    plt.ylabel('Annualized Alpha')
    filepath = os.path.join(output_dir, 'plots', f"{ga_choice}_decile_alphas.png")
    plt.savefig(filepath, dpi=300, bbox_inches='tight')
    plt.close()
    logger.info(f"Saved decile alpha plot: {filepath}")

########################################
### Industry Analysis ###
########################################

def industry_analysis(proc_df, decile_df, ff_df, ga_choice="goodwill_to_equity_lagged", industries=None):
    """
    Analyze GA factor performance across industries.
    """
    logger.info(f"{ga_choice} industry analysis...")
    
    if decile_df.empty:
        logger.warning(f"No decile data—assigning from raw data.")
        df = proc_df.copy()
        if df[ga_choice].nunique() < 10:
            logger.warning(f"Too few unique {ga_choice} values ({df[ga_choice].nunique()})—skipping.")
            return
        df['decile'] = pd.qcut(df[ga_choice], 10, labels=False, duplicates='drop') + 1
    else:
        df = proc_df.merge(decile_df[['permno', 'crsp_date', 'decile']], 
                           on=['permno', 'crsp_date'], how='inner')

    # Industry classification using FF48 mapping
    df['sic'] = df['sich'].fillna(df['siccd'])
    df['ff48_industry'] = df['sic'].apply(map_sic_to_ff48)
    
    # Compute industry statistics
    industry_stats = df.groupby('ff48_industry').agg(
        num_firms=('permno', 'nunique'),
        total_market_cap=('market_cap', 'sum'),
        median_market_cap=('market_cap', 'median'),
        median_annual_return=('ret', lambda x: x.groupby(df['year']).mean().median())
    ).reset_index()
    # Calculate total market cap of universe
    total_universe_market_cap = df['market_cap'].sum()
    industry_stats['weight_in_universe'] = industry_stats['total_market_cap'] / total_universe_market_cap
    
    # Save industry statistics
    filepath = os.path.join(output_dir, 'industry', f"{ga_choice}_industry_statistics.xlsx")
    industry_stats.to_excel(filepath, index=False)
    logger.info(f"Saved industry statistics to: {filepath}")
    
    if industries is None:
        industries = industry_stats['ff48_industry'].unique()
    logger.info(f"Industries: {list(industries)}")

    df = df[df['ret'].notna() & (df['market_cap'] > 0)].copy()
    logger.info(f"Data for factor computation: {len(df)} rows")

    industry_factors = {}
    for weighting in ["equal", "value"]:
        factor_returns = []
        for industry in industries:
            ind_df = df[df['ff48_industry'] == industry].copy()
            if len(ind_df) < 20:
                logger.warning(f"Skipping {industry}: insufficient data ({len(ind_df)} rows)")
                continue
            ind_df = ind_df[ind_df['decile'].isin([1, 10])].copy()
            if len(ind_df['decile'].unique()) < 2:
                logger.warning(f"Skipping {industry}: missing deciles 1 or 10 ({ind_df['decile'].unique()})")
                continue
            
            if weighting == "equal":
                factor = ind_df.groupby(['crsp_date', 'decile'])['ret'].mean().unstack()
            else:
                factor = ind_df.groupby(['crsp_date', 'decile']).apply(
                    lambda x: np.average(x['ret'], weights=x['market_cap']) if len(x) >= 5 else np.nan,
                    include_groups=False
                ).unstack()
            
            if 10 not in factor.columns or 1 not in factor.columns:
                logger.warning(f"Skipping {industry}: missing decile 1 or 10 in returns")
                continue
            
            factor['ga_factor'] = factor[1] - factor[10]  # Low-minus-high
            factor = factor[['ga_factor']].dropna().reset_index()
            factor['industry'] = industry
            factor_returns.append(factor)
        
        if not factor_returns:
            logger.warning(f"No {weighting}-weighted factors computed for {ga_choice}")
            continue
        
        combined = pd.concat(factor_returns, axis=0)
        combined.rename(columns={'crsp_date': 'date'}, inplace=True)
        filepath = os.path.join(output_dir, 'industry', 
                                f"ga_factor_returns_{weighting}_{ga_choice}_by_industry.csv")
        combined.to_csv(filepath, index=False)
        logger.info(f"Saved {weighting}-weighted factors: {filepath}")
        industry_factors[weighting] = combined

        # Visualization: Industry Factor Returns Over Time
        plt.figure(figsize=(12, 6))
        for industry in combined['industry'].unique():
            ind_data = combined[combined['industry'] == industry]
            plt.plot(ind_data['date'], ind_data['ga_factor'], label=industry)
        plt.title(f'{weighting.capitalize()}-Weighted {ga_choice} Factor Returns by Industry')
        plt.xlabel('Date')
        plt.ylabel('Factor Return (Low - High)')
        plt.legend()
        plt.grid(True, alpha=0.7)
        filepath = os.path.join(output_dir, 'plots', 
                                f"{ga_choice}_{weighting}_factor_returns_by_industry.png")
        plt.savefig(filepath, dpi=300, bbox_inches='tight')
        plt.close()
        logger.info(f"Saved industry factor returns plot: {filepath}")

    # Regression analysis
    factor_models = {"CAPM": ["mkt_rf"], "FF5": ["mkt_rf", "smb", 'hml', "rmw", "cma"]}
    all_results = []
    for weighting, factors in industry_factors.items():
        for industry in factors['industry'].unique():
            ind_df = factors[factors['industry'] == industry].merge(ff_df, on='date', how='inner')
            if len(ind_df) < 12:
                logger.warning(f"Skipping {industry} ({weighting}): too few months ({len(ind_df)})")
                continue
            ind_df['ga_factor_excess'] = ind_df['ga_factor'] - ind_df['rf']
            
            for model_name, factor_list in factor_models.items():
                X = sm.add_constant(ind_df[factor_list], has_constant='add')
                y = ind_df['ga_factor_excess']
                model = sm.OLS(y, X, missing='drop').fit(cov_type='HAC', cov_kwds={'maxlags': 3})
                all_results.append({
                    'Industry': industry,
                    'Weighting': weighting,
                    'Model': model_name,
                    'Alpha (Annualized)': model.params.get('const', np.nan) * 12,
                    'Alpha t-stat': model.tvalues.get('const', np.nan),
                    'Alpha p-value': model.pvalues.get('const', np.nan),
                    'R-squared': model.rsquared,
                    'Observations': int(model.nobs)
                })
    
    if not all_results:
        logger.warning(f"No regression results for {ga_choice}")
        return
    
    results_df = pd.DataFrame(all_results)
    filepath = os.path.join(output_dir, 'industry', f"industry_regression_results_{ga_choice}.xlsx")
    results_df.to_excel(filepath, index=False)
    logger.info(f"Saved industry regression results: {filepath}")

    # Visualization: Industry Alphas
    plt.figure(figsize=(12, 6))
    sns.barplot(data=results_df[results_df['Model'] == 'CAPM'], x='Industry', y='Alpha (Annualized)', hue='Weighting')
    plt.title(f'CAPM Annualized Alpha for {ga_choice} Factor by Industry')
    plt.xlabel('Industry')
    plt.ylabel('Annualized Alpha')
    plt.xticks(rotation=45)
    filepath = os.path.join(output_dir, 'plots', f"{ga_choice}_industry_alphas.png")
    plt.savefig(filepath, dpi=300, bbox_inches='tight')
    plt.close()
    logger.info(f"Saved industry alpha plot: {filepath}")

########################################
### Main Execution ###
########################################

def main(ga_choice="goodwill_to_equity_lagged", industries=None):
    """
    Main function to run all diagnostic analyses.
    """
    global FF48_MAPPING
    FF48_MAPPING = load_ff48_mapping_from_excel("/Users/carlaamodt/thesis_project/Excel own/Industry classificationFF.xlsx")
    try:
        proc_df, decile_df, ff_df, ga_factors_df = load_data(ga_choice=ga_choice)
        decile_analysis(proc_df, decile_df, ga_choice=ga_choice)
        ga_factor_diagnostics(proc_df, ff_df, ga_factors_df, ga_choice=ga_choice)
        industry_analysis(proc_df, decile_df, ff_df, ga_choice=ga_choice, industries=industries)
        market_return_reference_plot(ff_df)
        decile_alpha_regressions(proc_df, decile_df, ff_df, ga_choice=ga_choice)
        logger.info(f"All {ga_choice} outputs generated in {output_dir}!")
    except Exception as e:
        logger.error(f"Failed: {e}")
        raise

if __name__ == "__main__":
    main(ga_choice="goodwill_to_equity_lagged")