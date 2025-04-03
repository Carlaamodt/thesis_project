import pandas as pd
import numpy as np
from datetime import datetime
import os
import logging

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Set pandas options for better display
pd.set_option('display.max_columns', None)

def load_data(filepath):
    """
    Load the processed data from a CSV file and validate required columns.
    
    Args:
        filepath (str): Path to the input CSV file.
    
    Returns:
        pd.DataFrame: Loaded DataFrame with validated columns.
    """
    logger.info(f"Loading data from {filepath}...")
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"Data file not found at {filepath}")
    
    required_cols = [
        'crsp_date', 'ret', 'market_cap', 'exchcd', 'permno',
        'goodwill_to_sales_lagged', 'goodwill_to_equity_lagged', 'goodwill_to_market_cap_lagged'
    ]
    
    try:
        df = pd.read_csv(filepath, parse_dates=['crsp_date'])
        missing_cols = [col for col in required_cols if col not in df.columns]
        if missing_cols:
            raise ValueError(f"Missing required columns: {missing_cols}")
        
        logger.info(f"Loaded {len(df)} rows, {df['permno'].nunique()} unique firms")
        logger.info(f"Years present: {sorted(df['crsp_date'].dt.year.unique())}")
        logger.info(f"Missing values:\n{df.isna().sum()}")
        return df
    except Exception as e:
        logger.error(f"Failed to load data: {e}")
        raise
def compute_june_me_ff_style(df):
    """
    Compute the forward-filled June market equity (june_me_ff_style) for each firm.
    
    Args:
        df (pd.DataFrame): Input DataFrame with crsp_date, permno, and market_cap.
    
    Returns:
        pd.DataFrame: DataFrame with june_me_ff_style column added.
    """
    logger.info("Computing june_me_ff_style...")
    # Extract June market cap
    df['year'] = df['crsp_date'].dt.year
    june_me = df[df['crsp_date'].dt.month == 6][['permno', 'year', 'market_cap']].rename(columns={'market_cap': 'june_me'})
    
    # Merge June market cap back into the main DataFrame
    df = df.merge(june_me, on=['permno', 'year'], how='left')
    
    # Forward-fill june_me within each permno
    df = df.sort_values(['permno', 'crsp_date'])
    df['june_me_ff_style'] = df.groupby('permno')['june_me'].ffill()
    
    logger.info(f"Missing june_me_ff_style: {df['june_me_ff_style'].isna().sum()}")
    return df
# Function to compute NYSE median market cap in June each year
def compute_nyse_size_breakpoints(df):
    """
    Compute the NYSE median market cap in June for each year.
    
    Args:
        df (pd.DataFrame): DataFrame with June data only.
    
    Returns:
        pd.DataFrame: DataFrame with year and nyse_median.
    """
    june_df = df[(df['crsp_date'].dt.month == 6) & (df['exchcd'] == 1)]  # NYSE only
    nyse_medians = june_df.groupby(june_df['crsp_date'].dt.year)['market_cap'].median().reset_index()
    nyse_medians.columns = ['year', 'nyse_median']
    logger.info(f"NYSE medians computed:\n{nyse_medians}")
    return nyse_medians

# Function to compute decile breakpoints using NYSE firms
def get_nyse_decile_breakpoints(df, metric, year):
    """
    Compute decile breakpoints for a GA metric using NYSE firms in June.
    
    Args:
        df (pd.DataFrame): DataFrame with June data.
        metric (str): GA metric to sort on.
        year (int): Year to compute breakpoints for.
    
    Returns:
        np.ndarray: Array of breakpoints (10th to 90th percentiles), or None if insufficient data.
    """
    nyse_df = df[(df['exchcd'] == 1) & (df['crsp_date'].dt.month == 6) & (df['year'] == year)]
    nyse_df = nyse_df[nyse_df[metric].notna()]
    if len(nyse_df) < 20:  # Require at least 20 firms
        logger.warning(f"Year {year}: Too few NYSE firms ({len(nyse_df)}) for {metric} breakpoints")
        return None
    breakpoints = np.percentile(nyse_df[metric], np.arange(10, 100, 10))
    logger.info(f"Year {year}: {metric} breakpoints: {breakpoints}")
    return breakpoints

# Function to assign deciles
def assign_deciles(values, breakpoints):
    """
    Assign deciles to values based on breakpoints.
    
    Args:
        values (pd.Series): Values to assign deciles to.
        breakpoints (np.ndarray): Breakpoints for deciles.
    
    Returns:
        pd.Series: Decile assignments (1 to 10).
    """
    if breakpoints is None:
        return pd.Series(np.nan, index=values.index)
    return pd.cut(values, bins=[-np.inf] + list(breakpoints) + [np.inf], labels=False, include_lowest=True) + 1

# Function to compute portfolio returns
def compute_portfolio_returns(group, weight_col=None):
    """
    Compute portfolio returns (equal-weighted or value-weighted) and number of firms.
    
    Args:
        group (pd.DataFrame): Group of firms to compute returns for.
        weight_col (str, optional): Column to use for value-weighting.
    
    Returns:
        tuple: (portfolio return, number of firms)
    """
    if len(group) == 0:
        return np.nan, 0
    if weight_col is None:  # Equal-weighted
        return group['ret'].mean(), len(group)
    else:  # Value-weighted
        weights = group[weight_col] / group[weight_col].sum()
        return (group['ret'] * weights).sum(), len(group)
    
def count_firms_per_year(df, ga_column):
    logger.info(f"Counting firms per year for: {ga_column}")
    
    df['year'] = df['crsp_date'].dt.year
    firm_counts = df[df[ga_column].notna()].groupby('year').agg(
        num_firms=('permno', 'nunique'),
        num_unique_ga=(ga_column, 'nunique'),
        num_rows=('permno', 'count')
    ).reset_index()
    
    logger.info("Firms per year:\n%s", firm_counts)
    
    os.makedirs("analysis_output", exist_ok=True)
    output_path = f"analysis_output/firms_unique_{ga_column}_per_year.xlsx"
    firm_counts.to_excel(output_path, index=False)
    logger.info(f"Saved firm counts to {output_path}")
    
    return firm_counts

# Main function to process the data
def main(filepath):
    # Load data
    df = load_data(filepath)
    df_full = df.copy()
    df_full['year'] = df_full['crsp_date'].dt.year
    
    # Compute june_me_ff_style
    df = compute_june_me_ff_style(df)
    
    # Step 1: Data Preparation
    # Compute NYSE medians using June data (before excluding June)
    nyse_medians = compute_nyse_size_breakpoints(df)
    df = df.merge(nyse_medians, on='year', how='left')
    
    # Assign Small/Big based on NYSE median using june_me_ff_style
    df['size_group'] = np.where(df['june_me_ff_style'].isna(), np.nan,
                               np.where(df['june_me_ff_style'] <= df['nyse_median'], 'Small', 'Big'))
    logger.info(f"Size groups assigned:\n{df['size_group'].value_counts(dropna=False)}")
    
    # Filter to 2004–2023, exclude June, apply constraints
    df = df[(df['crsp_date'].dt.year >= 2004) & (df['crsp_date'].dt.year <= 2023)]
    df = df[df['crsp_date'].dt.month != 6]  # Exclude June per Fama-French
    df = df[df['ret'].notna() & (df['ret'] > -1) & (df['ret'] < 1) & df['june_me_ff_style'].notna()]
    logger.info(f"After filters: {len(df)} rows, {df['permno'].nunique()} unique firms")
    
    # Add year and month for grouping
    df['year'] = df['crsp_date'].dt.year
    df['month'] = df['crsp_date'].dt.month
    
    # List of GA metrics to process
    ga_metrics = ['goodwill_to_sales_lagged', 'goodwill_to_equity_lagged', 'goodwill_to_market_cap_lagged']
    
    # Main loop over GA metrics
    for ga_metric in ga_metrics:
        logger.info(f"Processing {ga_metric}...")
        count_firms_per_year(df_full, ga_metric)
        
        # Step 2: Factor Construction
        output_data = []
        
        # Group by year to assign deciles annually
        for year in range(2004, 2024):
            # Get NYSE breakpoints for this year (use original df with June data)
            breakpoints = get_nyse_decile_breakpoints(df_full[df_full['crsp_date'].dt.year == year], ga_metric, year)
            
            if breakpoints is None:
                continue
            
            # Assign deciles to all firms for July of this year to June of next year
            year_df = pd.concat([
                df[(df['crsp_date'].dt.year == year) & (df['crsp_date'].dt.month >= 7)],
                df[(df['crsp_date'].dt.year == year + 1) & (df['crsp_date'].dt.month <= 6)]
            ])
            year_df['decile'] = assign_deciles(year_df[ga_metric], breakpoints)
            
            # Monthly returns
            monthly_groups = year_df.groupby(['crsp_date', 'decile', 'size_group'])
            
            for (crsp_date, decile, size_group), group in monthly_groups:
                # Single-sort (ignore size_group for now)
                single_sort_group = year_df[(year_df['crsp_date'] == crsp_date) & 
                                           (year_df['decile'] == decile)]
                ew_ret, ew_n = compute_portfolio_returns(single_sort_group)
                vw_ret, vw_n = compute_portfolio_returns(single_sort_group, 'june_me_ff_style')
                
                # Double-sort (Small/Big)
                ew_ret_size, ew_n_size = compute_portfolio_returns(group)
                vw_ret_size, vw_n_size = compute_portfolio_returns(group, 'june_me_ff_style')
                
                output_data.append({
                    'crsp_date': crsp_date,
                    'decile': decile,
                    'size_group': size_group,
                    'single_ew_ret': ew_ret,
                    'single_vw_ret': vw_ret,
                    'single_ew_n': ew_n,
                    'single_vw_n': vw_n,
                    'size_ew_ret': ew_ret_size,
                    'size_vw_ret': vw_ret_size,
                    'size_ew_n': ew_n_size,
                    'size_vw_n': vw_n_size
                })
        
        # Convert to DataFrame
        factor_df = pd.DataFrame(output_data)
        factor_df['crsp_date'] = pd.to_datetime(factor_df['crsp_date']).dt.strftime('%Y-%m-%d')
        
        # Step 3: Compute Factors
        # Single-Sort Factor (D1 - D10) and store D1, D10 returns
        single_sort = factor_df.groupby('crsp_date', group_keys=False).apply(
            lambda x: pd.Series({
                'single_d1_ew': x.loc[x['decile'] == 1, 'single_ew_ret'].values[0] if 1 in x['decile'].values else np.nan,
                'single_d10_ew': x.loc[x['decile'] == 10, 'single_ew_ret'].values[0] if 10 in x['decile'].values else np.nan,
                'single_d1_vw': x.loc[x['decile'] == 1, 'single_vw_ret'].values[0] if 1 in x['decile'].values else np.nan,
                'single_d10_vw': x.loc[x['decile'] == 10, 'single_vw_ret'].values[0] if 10 in x['decile'].values else np.nan,
                'ga_factor_ew': (
                    x.loc[x['decile'] == 1, 'single_ew_ret'].values[0] -
                    x.loc[x['decile'] == 10, 'single_ew_ret'].values[0]
                ) if (1 in x['decile'].values and 10 in x['decile'].values) else np.nan,
                'ga_factor_vw': (
                    x.loc[x['decile'] == 1, 'single_vw_ret'].values[0] -
                    x.loc[x['decile'] == 10, 'single_vw_ret'].values[0]
                ) if (1 in x['decile'].values and 10 in x['decile'].values) else np.nan,
                'n_firms_d1_ew': x.loc[x['decile'] == 1, 'single_ew_n'].values[0] if 1 in x['decile'].values else 0,
                'n_firms_d10_ew': x.loc[x['decile'] == 10, 'single_ew_n'].values[0] if 10 in x['decile'].values else 0,
                'n_firms_d1_vw': x.loc[x['decile'] == 1, 'single_vw_n'].values[0] if 1 in x['decile'].values else 0,
                'n_firms_d10_vw': x.loc[x['decile'] == 10, 'single_vw_n'].values[0] if 10 in x['decile'].values else 0
            })
        ).reset_index()
        
        # Double-Sort Factor (Small and Big separately)
        double_sort = factor_df.groupby(['crsp_date', 'size_group'], group_keys=False).apply(
            lambda x: pd.Series({
                'd1_ew': x.loc[x['decile'] == 1, 'size_ew_ret'].values[0] if 1 in x['decile'].values else np.nan,
                'd10_ew': x.loc[x['decile'] == 10, 'size_ew_ret'].values[0] if 10 in x['decile'].values else np.nan,
                'd1_vw': x.loc[x['decile'] == 1, 'size_vw_ret'].values[0] if 1 in x['decile'].values else np.nan,
                'd10_vw': x.loc[x['decile'] == 10, 'size_vw_ret'].values[0] if 10 in x['decile'].values else np.nan,
                'factor_ew': (
                    x.loc[x['decile'] == 1, 'size_ew_ret'].values[0] -
                    x.loc[x['decile'] == 10, 'size_ew_ret'].values[0]
                ) if (1 in x['decile'].values and 10 in x['decile'].values) else np.nan,
                'factor_vw': (
                    x.loc[x['decile'] == 1, 'size_vw_ret'].values[0] -
                    x.loc[x['decile'] == 10, 'size_vw_ret'].values[0]
                ) if (1 in x['decile'].values and 10 in x['decile'].values) else np.nan,
                'n_firms_d1_ew': x.loc[x['decile'] == 1, 'size_ew_n'].values[0] if 1 in x['decile'].values else 0,
                'n_firms_d10_ew': x.loc[x['decile'] == 10, 'size_ew_n'].values[0] if 10 in x['decile'].values else 0,
                'n_firms_d1_vw': x.loc[x['decile'] == 1, 'size_vw_n'].values[0] if 1 in x['decile'].values else 0,
                'n_firms_d10_vw': x.loc[x['decile'] == 10, 'size_vw_n'].values[0] if 10 in x['decile'].values else 0
            })
        ).reset_index()
        
        # Pivot double-sort for Small and Big
        double_sort_pivot = double_sort.pivot(index='crsp_date', columns='size_group', 
                                              values=['d1_ew', 'd10_ew', 'd1_vw', 'd10_vw', 'factor_ew', 'factor_vw', 
                                                      'n_firms_d1_ew', 'n_firms_d10_ew', 'n_firms_d1_vw', 'n_firms_d10_vw'])
        double_sort_pivot.columns = [f'{col[0]}_{col[1].lower()}' for col in double_sort_pivot.columns]
        
        # Size-Adjusted 10x2
        size_adj = factor_df.groupby(['crsp_date', 'decile']).apply(
            lambda x: pd.Series({
                'sizeadj_ew': x['size_ew_ret'].mean(),
                'sizeadj_vw': x['size_vw_ret'].mean(),
                'n_firms_ew': x['size_ew_n'].sum(),
                'n_firms_vw': x['size_vw_n'].sum()
            })
        ).reset_index()
        
        # Pivot size-adjusted returns for all deciles
        size_adj_pivot = size_adj.pivot(index='crsp_date', columns='decile', 
                                        values=['sizeadj_ew', 'sizeadj_vw', 'n_firms_ew', 'n_firms_vw'])
        size_adj_pivot.columns = [f'{col[0]}_d{int(col[1])}' for col in size_adj_pivot.columns]
        
        # Size-adjusted factor (D1 - D10)
        size_adj_factor = size_adj.groupby('crsp_date').apply(
            lambda x: pd.Series({
                'ga_factor_sizeadj_ew': (x[x['decile'] == 1]['sizeadj_ew'].values[0] - 
                                        x[x['decile'] == 10]['sizeadj_ew'].values[0]) if (1 in x['decile'].values and 10 in x['decile'].values) else np.nan,
                'ga_factor_sizeadj_vw': (x[x['decile'] == 1]['sizeadj_vw'].values[0] - 
                                        x[x['decile'] == 10]['sizeadj_vw'].values[0]) if (1 in x['decile'].values and 10 in x['decile'].values) else np.nan
            })
        ).reset_index()
        
        # Merge all results
        final_df = single_sort.merge(double_sort_pivot, on='crsp_date', how='outer')\
                              .merge(size_adj_pivot, on='crsp_date', how='outer')\
                              .merge(size_adj_factor, on='crsp_date', how='outer')
        
        # Step 4: Save to CSV
        final_df.to_csv(f'{ga_metric}_factors.csv', index=False)
        logger.info(f"Saved {ga_metric}_factors.csv")
    # ============================
    # 📊 Post-hoc Sharpe diagnostics
    # ============================
    import math

    def compute_annualized_sharpe(series):
        series = series.dropna()
        mean_return = series.mean() * 12
        std_dev = series.std() * math.sqrt(12)
        sharpe = mean_return / std_dev if std_dev > 0 else np.nan
        return round(sharpe, 4), round(mean_return, 4), round(std_dev, 4)

    # Inside the ga_metric loop, after final_df is created
    print(f"\n📈 Sharpe Ratios for {ga_metric}")
    print("---------------------------------------")
    for col in final_df.columns:
        if col.startswith("ga_factor") or col.startswith("factor_"):
            sharpe, mean_ret, std_ret = compute_annualized_sharpe(final_df[col])
            print(f"{col:<25} | Sharpe: {sharpe:<6} | Mean: {mean_ret:<6} | Std: {std_ret:<6}")
    # ============================
    logger.info("Factor construction complete!")
     
# Run the script
if __name__ == "__main__":
    # Specify your filepath here
    filepath = "/Users/carlaamodt/thesis_project/data/processed_data.csv"
    main(filepath)