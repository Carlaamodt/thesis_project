import pandas as pd
import numpy as np
import os
import logging

##################################
### Setup Logging
##################################

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger()

##################################
### Load Processed Data
##################################

def load_processed_data(directory="data/", filename="processed_data.csv", ga_choice="GA1_lagged"):
    filepath = os.path.join(directory, filename)
    logger.info(f"Loading processed data from {filepath} with {ga_choice} ...")
    required_cols = ['permno', 'gvkey', 'crsp_date', 'FF_year', ga_choice, 'ret', 'prc', 'csho']
    try:
        df = pd.read_csv(filepath, parse_dates=['crsp_date'], usecols=required_cols, low_memory=False)
        df = df.drop_duplicates(subset=['permno', 'crsp_date'])
        logger.info(f"Loaded rows: {len(df)}, unique permno: {df['permno'].nunique()}")
        if len(df) < 100000:
            logger.warning(f"Low row count: {len(df)}")
        return df
    except Exception as e:
        logger.error(f"Load failed: {e}")
        raise

##################################
### GA Lagged Diagnostics
##################################

def ga_lagged_diagnostics(df, ga_column):
    logger.info(f"Running {ga_column} Diagnostics...")
    df_goodwill = df[df[ga_column].notna()].copy()
    if df_goodwill.empty:
        logger.error(f"No firms with {ga_column} data!")
        raise ValueError(f"No {ga_column} data.")
    
    print(f"\n📊 Goodwill firms: {df_goodwill['permno'].nunique()} (rows: {len(df_goodwill)})")
    nans = df_goodwill[ga_column].isna().sum()
    df_goodwill[ga_column] = df_goodwill[ga_column].fillna(0)
    zeros = df_goodwill[ga_column].eq(0).sum()
    non_zeros = df_goodwill[ga_column].ne(0).sum()
    
    print(f"❌ Missing {ga_column} (pre-fill NaN): {nans}")
    print(f"🔢 Zero {ga_column} (including filled NaN): {zeros}")
    print(f"🔢 Non-zero {ga_column}: {non_zeros}")
    print(f"🆔 Unique {ga_column} (total): {df_goodwill[ga_column].nunique()}")
    print(f"🆔 Unique {ga_column} (non-zero): {df_goodwill.loc[df_goodwill[ga_column] != 0, ga_column].nunique()}")
    print(f"✅ Total rows check: {nans + (zeros - nans) + non_zeros} (should equal {len(df_goodwill)})")
    
    if non_zeros > 0:
        stats = df_goodwill[ga_column].describe(percentiles=[0.01, 0.05, 0.25, 0.5, 0.75, 0.95, 0.99])
        extremes = df_goodwill[ga_column].quantile([0.001, 0.999]).to_dict()
        print(f"⚠️ Extreme {ga_column} (0.1%, 99.9%): {extremes}")
        print(f"\n📊 {ga_column} Stats:\n", stats)
    print("✅ Diagnostics Complete.\n")

##################################
### Count Firms Per Year
##################################

def count_firms_per_year(df, ga_column):
    logger.info("Counting firms per FF_year...")
    firm_counts = df[df[ga_column].notna()].groupby('FF_year').agg(
        num_firms=('permno', 'nunique'),
        num_unique_ga=(ga_column, 'nunique'),
        num_rows=('permno', 'count')
    ).reset_index()
    if firm_counts['num_firms'].min() < 20:
        logger.warning(f"Years with <20 firms: {firm_counts[firm_counts['num_firms'] < 20]['FF_year'].tolist()}")
    logger.info("Firms per FF_year:\n%s", firm_counts)
    os.makedirs("analysis_output", exist_ok=True)
    firm_counts.to_excel(f"analysis_output/firms_unique_{ga_column}_per_year.xlsx", index=False)
    logger.info(f"Saved to analysis_output/firms_unique_{ga_column}_per_year.xlsx")
    return firm_counts

##################################
### Assign GA Deciles
##################################

def assign_ga_deciles(df, ga_column):
    logger.info(f"Assigning {ga_column} deciles...")
    df['decile'] = 0
    decile_assignments = []
    for year, group in df.groupby('FF_year'):
        goodwill_firms = group[group[ga_column].notna()].copy()
        valid_ga = goodwill_firms[goodwill_firms[ga_column] != 0].copy()
        logger.info(f"Year {year}: Goodwill firms = {len(goodwill_firms)}, Valid GA = {len(valid_ga)}")
        if len(valid_ga) >= 20 and valid_ga[ga_column].nunique() > 5:
            try:
                valid_ga.loc[:, 'decile'] = pd.qcut(valid_ga[ga_column], 10, labels=False, duplicates='drop') + 1
                decile_assignments.append(valid_ga[['permno', 'crsp_date', 'decile']])
            except ValueError as e:
                logger.warning(f"Year {year}: Decile failed - {e}")
        else:
            logger.warning(f"Year {year}: Insufficient data (firms: {len(valid_ga)}, unique GA: {valid_ga[ga_column].nunique()})")
    if not decile_assignments:
        logger.error("No decile assignments!")
        raise ValueError("Decile assignment failed.")
    deciles_df = pd.concat(decile_assignments, axis=0)
    df = df.merge(deciles_df, on=['permno', 'crsp_date'], how='left', suffixes=('', '_new'))
    df['decile'] = df['decile_new'].fillna(df['decile']).astype(int)
    df.drop(columns=['decile_new'], inplace=True)
    logger.info("✅ Decile distribution:\n%s", df['decile'].value_counts().sort_index())
    os.makedirs("analysis_output", exist_ok=True)
    df.to_csv(f'analysis_output/decile_assignments_{ga_column}.csv', index=False)
    logger.info(f"Saved to analysis_output/decile_assignments_{ga_column}.csv")
    return df

##################################
### Create GA Factor Returns
##################################

def create_ga_factor(df, ga_column):
    logger.info(f"Creating {ga_column} factor returns...")
    df['crsp_date'] = pd.to_datetime(df['crsp_date']) + pd.offsets.MonthEnd(0)
    df = df[df['decile'].isin([1, 10]) & df['ret'].notna() & (df['ret'].abs() <= 5)].copy()
    if len(df) < 100:
        logger.error(f"Too few rows for factor: {len(df)}")
        raise ValueError("Insufficient data.")
    
    df['ME'] = np.where((df['prc'] > 0) & (df['csho'] > 0), np.abs(df['prc']) * df['csho'], np.nan)
    df = df[df['ME'].notna() & (df['ME'] > 0)]

    equal_weighted = df.groupby(['crsp_date', 'decile']).agg(
        ret_mean=('ret', 'mean'),
        num_firms=('permno', 'count')
    ).unstack()
    min_firms = equal_weighted['num_firms'].min().min()
    if min_firms < 5:
        sparse_dates = equal_weighted['num_firms'][(equal_weighted['num_firms'][1] < 5) |
                                                  (equal_weighted['num_firms'][10] < 5)].index.tolist()
        logger.warning(f"Months with <5 firms in D1 or D10: {len(sparse_dates)} dates - {sparse_dates[:5]}...")
    equal_weighted['ga_factor'] = equal_weighted['ret_mean'][10] - equal_weighted['ret_mean'][1]
    equal_weighted = equal_weighted[['ga_factor']].dropna()
    # Reset index and rename
    equal_weighted.reset_index(inplace=True)
    equal_weighted.rename(columns={'crsp_date': 'date'}, inplace=True)

    # Double-check no NaNs remain
    if equal_weighted['ga_factor'].isna().any():
        logger.warning("NaNs found in ga_factor before saving!")
        equal_weighted = equal_weighted.dropna(subset=['ga_factor'])

    os.makedirs('output/factors', exist_ok=True)
    equal_weighted.to_csv(f'output/factors/ga_factor_returns_monthly_equal_{ga_column}.csv', index=False)

    def weighted_ret(x):
        if len(x) < 5 or x['ME'].sum() <= 0:
            return np.nan
        return np.average(x['ret'], weights=x['ME'])

    value_weighted = df.groupby(['crsp_date', 'decile'])[['ret', 'ME']].apply(weighted_ret, include_groups=False).unstack()
    value_weighted['ga_factor'] = value_weighted[10] - value_weighted[1]
    value_weighted = value_weighted[['ga_factor']].dropna()

    equal_weighted.reset_index(inplace=True)
    value_weighted.reset_index(inplace=True)
    equal_weighted.rename(columns={'crsp_date': 'date'}, inplace=True)
    value_weighted.rename(columns={'crsp_date': 'date'}, inplace=True)

    os.makedirs('output/factors', exist_ok=True)
    equal_weighted.to_csv(f'output/factors/ga_factor_returns_monthly_equal_{ga_column}.csv', index=False)
    value_weighted.to_csv(f'output/factors/ga_factor_returns_monthly_value_{ga_column}.csv', index=False)

    for name, factor in [("Equal-weighted", equal_weighted), ("Value-weighted", value_weighted)]:
        stats = factor['ga_factor'].describe()
        annualized_mean = stats['mean'] * 12
        annualized_std = stats['std'] * np.sqrt(12)
        logger.info(f"{name} {ga_column} factor stats:\n{stats}\n"
                    f"Annualized Mean: {annualized_mean:.4f}, Annualized Std: {annualized_std:.4f}")
        if stats['count'] < 12 or stats['std'] == 0:
            logger.warning(f"{name} factor might be unreliable: {stats}")

    logger.info(f"✅ {ga_column} factor returns saved to 'output/factors/'")
    df[['permno', 'crsp_date', 'FF_year', 'decile', ga_column, 'ME']].to_csv(
        f'analysis_output/decile_assignments_{ga_column}.csv', index=False
    )
    return equal_weighted, value_weighted

##################################
### Main Pipeline
##################################

def main(ga_choice="GA1_lagged"):
    try:
        df = load_processed_data(ga_choice=ga_choice)
        ga_lagged_diagnostics(df, ga_column=ga_choice)
        count_firms_per_year(df, ga_column=ga_choice)
        df = assign_ga_deciles(df, ga_column=ga_choice)
        equal_weighted, value_weighted = create_ga_factor(df, ga_column=ga_choice)
        logger.info(f"Final dataset shape: {df.shape}")
    except Exception as e:
        logger.error(f"Pipeline failed: {e}")
        raise

if __name__ == "__main__":
    main(ga_choice="GA1_lagged")  # Swap to "GA2_lagged" or "GA3_lagged" here