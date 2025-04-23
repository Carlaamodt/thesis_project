import pandas as pd
import numpy as np
import os
import logging
import argparse
from typing import Tuple, Optional, Union
import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Set pandas options for better display (used for debugging)
pd.set_option('display.max_columns', None)

def load_ff48_mapping_from_excel(filepath: str) -> dict:
    """
    Load Fama-French 48 industry classification mapping from an Excel file.

    Args:
        filepath (str): Path to the Excel file.

    Returns:
        dict: Dictionary mapping SIC code ranges (as Python ranges) to FF48 industry number.
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
    sic = int(sic) if pd.notna(sic) else 0
    for sic_range, industry in FF48_MAPPING.items():
        if sic in sic_range:
            return industry
    return 48  # Default to "Other" if no match

def load_data(filepath: str) -> pd.DataFrame:
    """
    Load the processed data from a CSV file and validate required columns.
    
    Args:
        filepath (str): Path to the input CSV file.
    
    Returns:
        pd.DataFrame: Loaded DataFrame with validated columns.
    
    Raises:
        FileNotFoundError: If the filepath does not exist.
        ValueError: If required columns are missing or data types are invalid.
    """
    logger.info(f"Loading data from {filepath}...")
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"Data file not found at {filepath}")
    
    required_cols = [
        'crsp_date', 'ret', 'market_cap', 'exchcd', 'permno',
        'goodwill_to_sales_lagged', 'goodwill_to_equity_lagged', 'goodwill_to_shareholder_equity_cap_lagged',
        'sich', 'siccd'
    ]
    
    try:
        df = pd.read_csv(filepath, parse_dates=['crsp_date'])
        missing_cols = [col for col in required_cols if col not in df.columns]
        if missing_cols:
            raise ValueError(f"Missing required columns: {missing_cols}")
        
        # Validate data types
        if not pd.api.types.is_datetime64_any_dtype(df['crsp_date']):
            raise ValueError("crsp_date must be a datetime column")
        numeric_cols = ['ret', 'market_cap', 'exchcd']
        for col in numeric_cols:
            if not pd.api.types.is_numeric_dtype(df[col]):
                raise ValueError(f"{col} must be numeric, got {df[col].dtype}")
        
        # Check for duplicates
        duplicates = df.duplicated().sum()
        if duplicates > 0:
            logger.warning(f"Found {duplicates} duplicate rows in the data")
        
        logger.info(f"Loaded {len(df)} rows, {df['permno'].nunique()} unique firms")
        logger.info(f"Years present: {sorted(df['crsp_date'].dt.year.unique())}")
        logger.info(f"Missing values:\n{df.isna().sum()}")
        logger.debug(f"First few rows:\n{df.head()}")
        return df
    except Exception as e:
        logger.error(f"Failed to load data: {e}")
        raise

def compute_industry_adjusted_gte(df: pd.DataFrame) -> pd.DataFrame:
    logger.info("Computing industry-adjusted GA metrics...")
    df['sic'] = df['sich'].fillna(df['siccd'])
    df['ff48_industry'] = df['sic'].apply(map_sic_to_ff48)
    
    # June t only
    june_df = df[df['crsp_date'].dt.month == 6].copy()
    
    # Metrics to adjust
    ga_metrics = ['goodwill_to_sales_lagged', 'goodwill_to_equity_lagged', 'goodwill_to_shareholder_equity_cap_lagged']
    for metric in ga_metrics:
        june_df[f'{metric}_industry_adj'] = june_df[metric]
    
    # Industry averages
    for metric in ga_metrics:
        industry_avg = (june_df.groupby(['year', 'ff48_industry'])
                        .agg({metric: 'mean', 'permno': 'count'})
                        .rename(columns={metric: f'{metric}_avg', 'permno': 'num_firms'}))
        industry_avg = industry_avg[industry_avg['num_firms'] >= 3]
        june_df = june_df.merge(industry_avg[[f'{metric}_avg']], 
                               on=['year', 'ff48_industry'], how='left')
        june_df[f'{metric}_industry_adj'] = june_df[metric] - june_df[f'{metric}_avg']
        june_df = june_df.drop(columns=[f'{metric}_avg'])
    
    # Merge back
    merge_cols = ['permno', 'year'] + [f'{m}_industry_adj' for m in ga_metrics]
    df = df.merge(june_df[merge_cols], on=['permno', 'year'], how='left')
    return df
def compute_june_me_ff_style(df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute the forward-filled June market equity (june_me_ff_style) for each firm.
    Impute missing values using the most recent available market_cap.
    
    Args:
        df (pd.DataFrame): Input DataFrame with crsp_date, permno, and market_cap.
    
    Returns:
        pd.DataFrame: DataFrame with june_me_ff_style column added.
    """
    logger.info("Computing june_me_ff_style...")
    df = df.sort_values(['permno', 'crsp_date']).copy()
    
    # Extract June market cap
    df['year'] = df['crsp_date'].dt.year
    df['is_june'] = df['crsp_date'].dt.month == 6
    june_me = df[df['is_june']][['permno', 'year', 'market_cap']].rename(columns={'market_cap': 'june_me'})
    
    # Merge June market cap back into the main DataFrame
    df = df.merge(june_me, on=['permno', 'year'], how='left')
    
    # Forward-fill june_me within each permno
    df['june_me_ff_style'] = df.groupby('permno')['june_me'].ffill()
    
    # Impute missing june_me_ff_style using the most recent market_cap
    missing_june_me = df['june_me_ff_style'].isna()
    if missing_june_me.any():
        logger.info(f"Imputing {missing_june_me.sum()} missing june_me_ff_style values...")
        df['market_cap_ff'] = df.groupby('permno')['market_cap'].ffill()
        df['june_me_ff_style'] = df['june_me_ff_style'].fillna(df['market_cap_ff'])
        logger.info(f"After imputation, missing june_me_ff_style: {df['june_me_ff_style'].isna().sum()}")
    
    # Validate june_me_ff_style
    invalid_weights = (df['june_me_ff_style'] <= 0) & (df['june_me_ff_style'].notna())
    if invalid_weights.any():
        logger.warning(f"Found {invalid_weights.sum()} rows with non-positive june_me_ff_style")
        df.loc[invalid_weights, 'june_me_ff_style'] = np.nan
    
    # Drop temporary columns
    df = df.drop(columns=['is_june', 'market_cap_ff', 'june_me'], errors='ignore')
    return df

def compute_sample_median_market_cap(df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute the median market cap across all firms in the sample in June for each year.
    
    Args:
        df (pd.DataFrame): DataFrame with crsp_date and market_cap.
    
    Returns:
        pd.DataFrame: DataFrame with year and sample_median.
    """
    logger.info("Computing sample median market cap...")
    # Use all firms in June (not just NYSE)
    june_df = df[df['crsp_date'].dt.month == 6]
    sample_medians = june_df.groupby(june_df['crsp_date'].dt.year)['market_cap'].median().reset_index()
    sample_medians.columns = ['year', 'sample_median']
    
    # Check for years with insufficient firms
    firm_counts = june_df.groupby(june_df['crsp_date'].dt.year)['permno'].nunique()
    low_counts = firm_counts[firm_counts < 5]
    if not low_counts.empty:
        logger.warning(f"Years with fewer than 5 firms:\n{low_counts}")
    
    # Fill missing years with the previous year's median
    all_years = pd.DataFrame({'year': range(df['crsp_date'].dt.year.min(), df['crsp_date'].dt.year.max() + 1)})
    sample_medians = all_years.merge(sample_medians, on='year', how='left').ffill()
    
    logger.info(f"Sample medians computed:\n{sample_medians}")
    return sample_medians

def get_sample_decile_breakpoints(df: pd.DataFrame, metric: str, year: int) -> Optional[np.ndarray]:
    """
    Compute decile breakpoints for a metric using all firms in the sample in June.
    
    Args:
        df (pd.DataFrame): DataFrame with June data.
        metric (str): Metric to sort on (e.g., gte_industry_adj).
        year (int): Year to compute breakpoints for.
    
    Returns:
        np.ndarray: Array of breakpoints (10th to 90th percentiles), or None if insufficient data.
    """
    sample_df = df[(df['crsp_date'].dt.month == 6) & (df['year'] == year)]
    sample_df = sample_df[sample_df[metric].notna()]
    
    if len(sample_df) < 10:  # Relaxed from 20 to 10 firms
        logger.warning(f"Year {year}: Too few firms ({len(sample_df)}) for {metric} breakpoints")
        return None
    
    if sample_df[metric].nunique() < 5:
        logger.warning(f"Year {year}: Too few unique {metric} values ({sample_df[metric].nunique()}) for deciles")
        return None
    
    breakpoints = np.percentile(sample_df[metric], np.arange(10, 100, 10))
    logger.info(f"Year {year}: {metric} breakpoints: {breakpoints}")
    return breakpoints

def assign_deciles(values: pd.Series, breakpoints: Optional[np.ndarray]) -> pd.Series:
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
    
    deciles = pd.cut(values, bins=[-np.inf] + list(breakpoints) + [np.inf], 
                     labels=False, include_lowest=True) + 1
    decile_counts = pd.Series(deciles).value_counts(dropna=False).sort_index()
    logger.info(f"Decile distribution:\n{decile_counts}")
    return deciles

def compute_portfolio_returns(group: pd.DataFrame, weight_col: Optional[str] = None, 
                             min_firms: int = 5) -> Tuple[float, int]:
    """
    Compute portfolio returns (equal-weighted or value-weighted) and number of firms.
    
    Args:
        group (pd.DataFrame): Group of firms to compute returns for.
        weight_col (str, optional): Column to use for value-weighting.
        min_firms (int): Minimum number of firms required for a valid return.
    
    Returns:
        tuple: (portfolio return, number of firms)
    """
    n_firms = len(group)
    if n_firms < min_firms:
        logger.warning(f"Group {group.name} has too few firms: {n_firms}")
        return np.nan, n_firms
    
    if weight_col is None:  # Equal-weighted
        return group['ret'].mean(), n_firms
    else:  # Value-weighted
        if group[weight_col].isna().all() or (group[weight_col] <= 0).all():
            logger.warning(f"Group {group.name} has invalid {weight_col} values")
            return np.nan, n_firms
        weights = group[weight_col] / group[weight_col].sum()
        return (group['ret'] * weights).sum(), n_firms

def count_firms_per_year(df: pd.DataFrame, metric: str) -> pd.DataFrame:
    """
    Count the number of firms with non-missing metric values per year.
    
    Args:
        df (pd.DataFrame): DataFrame with year and metric.
        metric (str): Metric column to analyze.
    
    Returns:
        pd.DataFrame: DataFrame with firm counts per year.
    """
    logger.info(f"Counting firms per year for: {metric}")
    firm_counts = df[df[metric].notna()].groupby('year').agg(
        num_firms=('permno', 'nunique'),
        num_unique_metric=(metric, 'nunique'),
        num_rows=('permno', 'count')
    ).reset_index()
    
    logger.info(f"Firms per year for {metric}:\n{firm_counts}")
    
    os.makedirs("analysis_output/industry_adjusted", exist_ok=True)
    output_path = f"analysis_output/industry_adjusted/firms_unique_{metric}_per_year.csv"
    firm_counts.to_csv(output_path, index=False)
    logger.info(f"Saved firm counts to {output_path}")
    
    return firm_counts

def main(filepath: str, ff48_filepath: str) -> None:
    global FF48_MAPPING
    FF48_MAPPING = load_ff48_mapping_from_excel(ff48_filepath)
    df = load_data(filepath)
    df['year'] = df['crsp_date'].dt.year
    df_full = df.copy()
    df_full = compute_industry_adjusted_gte(df_full)
    df = compute_june_me_ff_style(df_full)
    sample_medians = compute_sample_median_market_cap(df)
    df = df.merge(sample_medians, on='year', how='left')
    df['size_group'] = np.where(df['june_me_ff_style'].isna(), np.nan,
                               np.where(df['june_me_ff_style'] <= df['sample_median'], 'Small', 'Big'))
    logger.info(f"Size groups assigned:\n{df['size_group'].value_counts(dropna=False)}")
    df = df[(df['crsp_date'].dt.year >= 2004) & (df['crsp_date'].dt.year <= 2023)]
    df = df[df['crsp_date'].dt.month != 6]
    df = df[df['ret'].notna() & (df['ret'] > -1) & (df['ret'] < 1)]
    logger.info(f"After filters: {len(df)} rows, {df['permno'].nunique()} unique firms")
    df['month'] = df['crsp_date'].dt.month
    ga_metrics = ['goodwill_to_sales_lagged', 'goodwill_to_equity_lagged', 'goodwill_to_shareholder_equity_cap_lagged']
    for ga_metric in ga_metrics:
        metric = f'{ga_metric}_industry_adj'
        logger.info(f"Processing {metric}...")
        count_firms_per_year(df_full, metric)
        missing_ga = df_full[metric].isna().sum()
        logger.info(f"Firms with missing {metric} in full dataset: {missing_ga} ({missing_ga / len(df_full) * 100:.2f}%)")
        missing_ga_filtered = df[metric].isna().sum()
        logger.info(f"Firms with missing {metric} in filtered dataset: {missing_ga_filtered} ({missing_ga_filtered / len(df) * 100:.2f}%)")
        decile_assignments = []
        output_data = []
        for year in range(2004, 2024):
            breakpoints = get_sample_decile_breakpoints(df_full[df_full['crsp_date'].dt.year == year], metric, year)
            if breakpoints is None:
                continue
            year_df = pd.concat([
                df[(df['crsp_date'].dt.year == year) & (df['crsp_date'].dt.month >= 7)],
                df[(df['crsp_date'].dt.year == year + 1) & (df['crsp_date'].dt.month <= 6)]
            ])
            year_df['decile'] = assign_deciles(year_df[metric], breakpoints)
            decile_summary = year_df.groupby('decile')[metric].mean()
            logger.info(f"Year {year} avg {metric} per decile:\n{decile_summary}")
            decile_assignments.append(year_df[['permno', 'crsp_date', 'decile']])
            # Rest of the loop (single_sort, double_sort, etc.) remains unchanged
            single_sort_groups = year_df.groupby(['crsp_date', 'decile'])
            single_sort_results = single_sort_groups.apply(
                lambda g: pd.Series({
                    'single_ew_ret': compute_portfolio_returns(g)[0],
                    'single_ew_n': compute_portfolio_returns(g)[1],
                    'single_vw_ret': compute_portfolio_returns(g, 'june_me_ff_style')[0],
                    'single_vw_n': compute_portfolio_returns(g, 'june_me_ff_style')[1]
                })
            ).reset_index()
            double_sort_groups = year_df.groupby(['crsp_date', 'decile', 'size_group'])
            double_sort_results = double_sort_groups.apply(
                lambda g: pd.Series({
                    'size_ew_ret': compute_portfolio_returns(g)[0],
                    'size_ew_n': compute_portfolio_returns(g)[1],
                    'size_vw_ret': compute_portfolio_returns(g, 'june_me_ff_style')[0],
                    'size_vw_n': compute_portfolio_returns(g, 'june_me_ff_style')[1]
                })
            ).reset_index()
            merged_results = single_sort_results.merge(double_sort_results, on=['crsp_date', 'decile'], how='left')
            output_data.append(merged_results)
        if not output_data:
            logger.warning(f"No data processed for {metric}, skipping...")
            continue
        factor_df = pd.concat(output_data, ignore_index=True)
        factor_df['crsp_date'] = pd.to_datetime(factor_df['crsp_date']).dt.strftime('%Y-%m-%d')
        decile_df = pd.concat(decile_assignments, ignore_index=True)
        os.makedirs('output/industry_adjusted/deciles', exist_ok=True)
        decile_df.to_csv(f'output/industry_adjusted/deciles/{metric}_deciles.csv', index=False)
        logger.info(f"Saved decile assignments to output/industry_adjusted/deciles/{metric}_deciles.csv")
        # Factor construction (single_sort, double_sort, size_adj) remains unchanged
        single_sort = factor_df.groupby('crsp_date').apply(
            lambda x: pd.Series({
                'single_d1_ew': x[x['decile'] == 1]['single_ew_ret'].iloc[0] if 1 in x['decile'].values else np.nan,
                'single_d10_ew': x[x['decile'] == 10]['single_ew_ret'].iloc[0] if 10 in x['decile'].values else np.nan,
                'single_d1_vw': x[x['decile'] == 1]['single_vw_ret'].iloc[0] if 1 in x['decile'].values else np.nan,
                'single_d10_vw': x[x['decile'] == 10]['single_vw_ret'].iloc[0] if 10 in x['decile'].values else np.nan,
                'ga_factor_ew': (x[x['decile'] == 1]['single_ew_ret'].iloc[0] - 
                                 x[x['decile'] == 10]['single_ew_ret'].iloc[0]) if (1 in x['decile'].values and 10 in x['decile'].values) else np.nan,
                'ga_factor_vw': (x[x['decile'] == 1]['single_vw_ret'].iloc[0] - 
                                 x[x['decile'] == 10]['single_vw_ret'].iloc[0]) if (1 in x['decile'].values and 10 in x['decile'].values) else np.nan,
                'n_firms_d1_ew': x[x['decile'] == 1]['single_ew_n'].iloc[0] if 1 in x['decile'].values else 0,
                'n_firms_d10_ew': x[x['decile'] == 10]['single_ew_n'].iloc[0] if 10 in x['decile'].values else 0,
                'n_firms_d1_vw': x[x['decile'] == 1]['single_vw_n'].iloc[0] if 1 in x['decile'].values else 0,
                'n_firms_d10_vw': x[x['decile'] == 10]['single_vw_n'].iloc[0] if 10 in x['decile'].values else 0
            })
        ).reset_index()
        double_sort = factor_df.groupby(['crsp_date', 'size_group']).apply(
            lambda x: pd.Series({
                'd1_ew': x[x['decile'] == 1]['size_ew_ret'].iloc[0] if 1 in x['decile'].values else np.nan,
                'd10_ew': x[x['decile'] == 10]['size_ew_ret'].iloc[0] if 10 in x['decile'].values else np.nan,
                'd1_vw': x[x['decile'] == 1]['size_vw_ret'].iloc[0] if 1 in x['decile'].values else np.nan,
                'd10_vw': x[x['decile'] == 10]['size_vw_ret'].iloc[0] if 10 in x['decile'].values else np.nan,
                'factor_ew': (x[x['decile'] == 1]['size_ew_ret'].iloc[0] - 
                              x[x['decile'] == 10]['size_ew_ret'].iloc[0]) if (1 in x['decile'].values and 10 in x['decile'].values) else np.nan,
                'factor_vw': (x[x['decile'] == 1]['size_vw_ret'].iloc[0] - 
                              x[x['decile'] == 10]['size_vw_ret'].iloc[0]) if (1 in x['decile'].values and 10 in x['decile'].values) else np.nan,
                'n_firms_d1_ew': x[x['decile'] == 1]['size_ew_n'].iloc[0] if 1 in x['decile'].values else 0,
                'n_firms_d10_ew': x[x['decile'] == 10]['size_ew_n'].iloc[0] if 10 in x['decile'].values else 0,
                'n_firms_d1_vw': x[x['decile'] == 1]['size_vw_n'].iloc[0] if 1 in x['decile'].values else 0,
                'n_firms_d10_vw': x[x['decile'] == 10]['size_vw_n'].iloc[0] if 10 in x['decile'].values else 0
            })
        ).reset_index()
        double_sort_pivot = double_sort.pivot(index='crsp_date', columns='size_group',
                                              values=['d1_ew', 'd10_ew', 'd1_vw', 'd10_vw', 'factor_ew', 'factor_vw',
                                                      'n_firms_d1_ew', 'n_firms_d10_ew', 'n_firms_d1_vw', 'n_firms_d10_vw'])
        double_sort_pivot.columns = [f'{col[0]}_{col[1].lower()}' for col in double_sort_pivot.columns]
        size_adj = factor_df.groupby(['crsp_date', 'decile']).agg({
            'size_ew_ret': 'mean',
            'size_vw_ret': 'mean',
            'size_ew_n': 'sum',
            'size_vw_n': 'sum'
        }).reset_index().rename(columns={
            'size_ew_ret': 'sizeadj_ew',
            'size_vw_ret': 'sizeadj_vw',
            'size_ew_n': 'n_firms_ew',
            'size_vw_n': 'n_firms_vw'
        })
        size_adj_pivot = size_adj.pivot(index='crsp_date', columns='decile',
                                        values=['sizeadj_ew', 'sizeadj_vw', 'n_firms_ew', 'n_firms_vw'])
        size_adj_pivot.columns = [f'{col[0]}_d{int(col[1])}' for col in size_adj_pivot.columns]
        size_adj_factor = size_adj.groupby('crsp_date').apply(
            lambda x: pd.Series({
                'ga_factor_sizeadj_ew': (x[x['decile'] == 1]['sizeadj_ew'].iloc[0] - 
                                         x[x['decile'] == 10]['sizeadj_ew'].iloc[0]) if (1 in x['decile'].values and 10 in x['decile'].values) else np.nan,
                'ga_factor_sizeadj_vw': (x[x['decile'] == 1]['sizeadj_vw'].iloc[0] - 
                                         x[x['decile'] == 10]['sizeadj_vw'].iloc[0]) if (1 in x['decile'].values and 10 in x['decile'].values) else np.nan
            })
        ).reset_index()
        final_df = single_sort.merge(double_sort_pivot, on='crsp_date', how='outer')\
                              .merge(size_adj_pivot, on='crsp_date', how='outer')\
                              .merge(size_adj_factor, on='crsp_date', how='outer')
        os.makedirs('output/industry_adjusted/factors', exist_ok=True)
        output_path = f'output/industry_adjusted/factors/{metric}_factors.csv'
        final_df.to_csv(output_path, index=False)
        logger.info(f"Saved factors to {output_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Construct industry-adjusted GTE factors from processed data.")
    parser.add_argument(
        '--filepath',
        type=str,
        required=False,
        default="/Users/carlaamodt/thesis_project/data/processed_data.csv",
        help="Path to the processed data CSV file (default: %(default)s)"
    )
    parser.add_argument(
        '--ff48_filepath',
        type=str,
        required=False,
        default="/Users/carlaamodt/thesis_project/Excel own/Industry classificationFF.xlsx",
        help="Path to the Fama-French 48 industry classification Excel file (default: %(default)s)"
    )
    args = parser.parse_args()
    main(args.filepath, args.ff48_filepath)