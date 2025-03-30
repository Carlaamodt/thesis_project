import pandas as pd
import numpy as np
import os
import logging

##################################
### Setup Logging
##################################

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger()
SAVE_INTERMEDIATE_FILES = True  # Toggle to save intermediate CSVs

##################################
### Load Processed Data
##################################

def load_processed_data(directory="data/", filename="processed_data.csv", ga_choice="goodwill_to_sales_lagged"):
    filepath = os.path.join(directory, filename)
    logger.info(f"Loading processed data from {filepath} with {ga_choice} ...")
    required_cols = ['permno', 'gvkey', 'crsp_date', 'FF_year', ga_choice, 'ret', 'prc', 'csho', 'at', 'market_cap', 'is_nyse']
    try:
        df = pd.read_csv(filepath, parse_dates=['crsp_date'], usecols=required_cols, low_memory=False)
        df = df.drop_duplicates(subset=['permno', 'crsp_date'])
        df = df[df['crsp_date'].dt.year < 2024]  # Filter out 2024
        logger.info(f"Loaded rows: {len(df)}, unique permno: {df['permno'].nunique()}")
        logger.info(f"Years present: {sorted(df['crsp_date'].dt.year.unique())}")
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
    
    # Clip extreme values for robustness
    df_goodwill[ga_column] = df_goodwill[ga_column].clip(lower=-10, upper=10)
    
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
    df = df[df['FF_year'] < 2024].copy()
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

def assign_ga_deciles(df, ga_column, size_group=None):
    logger.info(f"Assigning {ga_column} deciles (Size: {size_group if size_group else 'All'})...")
    original_df = df.copy()  # Preserve original for merging
    df = df[df['FF_year'] < 2024].copy()
    if size_group:
        df = df[df['size_group'] == size_group].copy()
    logger.info(f"Missing values in {ga_column}: {df[ga_column].isna().sum()} / {len(df)} rows ({df[ga_column].isna().mean():.2%})")

    decile_assignments = []
    for year, group in df.groupby('FF_year'):
        goodwill_firms = group[
            group[ga_column].notna() &
            (group[ga_column] > 0) &
            (group['at'] > 0)
        ].copy()
        excluded = group.shape[0] - goodwill_firms.shape[0]
        logger.info(f"Year {year}: Excluded {excluded} firms due to missing or invalid goodwill or total assets.")
        logger.info(f"Year {year}: Goodwill firms = {len(goodwill_firms)}")

        if len(goodwill_firms) >= 20 and goodwill_firms[ga_column].nunique() > 5:
            try:
                goodwill_firms['decile'] = pd.qcut(goodwill_firms[ga_column], 10, labels=False, duplicates='drop') + 1
                goodwill_firms['decile'] = goodwill_firms['decile'].astype(int)
                decile_assignments.append(goodwill_firms[['permno', 'crsp_date', 'decile']])
            except ValueError as e:
                logger.warning(f"Year {year}: Decile assignment failed - {e}")
        else:
            logger.warning(f"Year {year}: Insufficient data (firms: {len(goodwill_firms)}, unique GA: {goodwill_firms[ga_column].nunique()})")

    if decile_assignments:
        deciles_df = pd.concat(decile_assignments, axis=0)
        original_df = original_df.drop(columns=['decile'], errors='ignore')
        original_df = original_df.merge(deciles_df, on=['permno', 'crsp_date'], how='left')
        if 'decile' in original_df.columns:
            original_df['decile'] = original_df['decile'].astype(pd.Int64Dtype())
            if original_df['decile'].isna().all():
                logger.error(f"No valid deciles assigned for {ga_column} (Size: {size_group if size_group else 'All'})")
            else:
                logger.info("✅ Decile distribution:\n%s", original_df['decile'].value_counts().sort_index())
        else:
            logger.warning(f"⚠️ No decile column created for {ga_column} (Size: {size_group if size_group else 'All'})")
        
        if SAVE_INTERMEDIATE_FILES:
            os.makedirs("analysis_output", exist_ok=True)
            original_df.to_csv(f'analysis_output/decile_assignments_{ga_column}_{size_group if size_group else "all"}.csv', index=False)
            logger.info(f"Saved to analysis_output/decile_assignments_{ga_column}_{size_group if size_group else 'all'}.csv")
        return original_df
    else:
        logger.warning(f"⚠️ No decile assignments made for {ga_column} (Size: {size_group if size_group else 'All'})")
        return original_df

##################################
### Create GA Factor Returns
##################################

def create_ga_factor(df, ga_column, size_group=None):
    logger.info(f"Creating {ga_column} factor returns (Size: {size_group if size_group else 'All'})...")
    df['crsp_date'] = pd.to_datetime(df['crsp_date']) + pd.offsets.MonthEnd(0)
    df = df[df['crsp_date'].dt.month != 6]  # Drop June returns (before size sort)

    # Apply return cap |ret| <= 1
    df = df[df['decile'].isin([1, 10]) & df['ret'].notna() & (df['ret'].abs() <= 1)].copy()
    dropped = df['ret'].abs().gt(1).sum()
    logger.info(f"Dropped rows due to |ret| > 1: {dropped} out of {len(df) + dropped}")
    
    if len(df) < 100:
        logger.error(f"Too few rows for factor: {len(df)}")
        raise ValueError("Insufficient data.")
    
    df['ME'] = df['june_me_ff_style']
    df = df[df['ME'].notna() & (df['ME'] > 0)]

    # Equal-weighted portfolio
    equal_weighted = df.groupby(['crsp_date', 'decile']).agg(
        ret_mean=('ret', 'mean'),
        num_firms=('permno', 'count')
    ).unstack()
    d1_counts = equal_weighted['num_firms'].get(1, pd.Series())
    d10_counts = equal_weighted['num_firms'].get(10, pd.Series())
    logger.info(f"Mean firms per month — D1: {d1_counts.mean():.2f}, D10: {d10_counts.mean():.2f}")

    min_firms = equal_weighted['num_firms'].min().min()
    if min_firms < 10:
        sparse_dates = equal_weighted['num_firms'][(equal_weighted['num_firms'][1] < 10) |
                                                  (equal_weighted['num_firms'][10] < 10)].index.tolist()
        logger.warning(f"Months with <10 firms in D1 or D10: {len(sparse_dates)} dates - {sparse_dates[:5]}...")
    
    equal_weighted['ga_factor'] = equal_weighted['ret_mean'][1] - equal_weighted['ret_mean'][10]
    logger.info("GA Factor = D1 (Low GA) – D10 (High GA): Long low goodwill, short high goodwill")
    equal_weighted = equal_weighted[['ga_factor']].dropna()
    equal_weighted.reset_index(inplace=True)
    equal_weighted.rename(columns={'crsp_date': 'date'}, inplace=True)

    # Value-weighted portfolio
    def weighted_ret(x):
        if len(x) < 10 or x['ME'].sum() <= 0:
            return np.nan
        return np.average(x['ret'], weights=x['ME'])

    value_weighted = df.groupby(['crsp_date', 'decile'])[['ret', 'ME']].apply(weighted_ret, include_groups=False).unstack()
    value_weighted['ga_factor'] = value_weighted[1] - value_weighted[10]
    value_weighted = value_weighted[['ga_factor']].dropna()
    value_weighted.reset_index(inplace=True)
    value_weighted.rename(columns={'crsp_date': 'date'}, inplace=True)

    # Save results
    suffix = f"_{size_group.lower()}" if size_group else ""
    if SAVE_INTERMEDIATE_FILES:
        os.makedirs('output/factors', exist_ok=True)
        equal_weighted.to_csv(f'output/factors/ga_factor_returns_monthly_equal_{ga_column}{suffix}.csv', index=False)
        value_weighted.to_csv(f'output/factors/ga_factor_returns_monthly_value_{ga_column}{suffix}.csv', index=False)
        logger.info(f"Saved EW to 'output/factors/ga_factor_returns_monthly_equal_{ga_column}{suffix}.csv'")
        logger.info(f"Saved VW to 'output/factors/ga_factor_returns_monthly_value_{ga_column}{suffix}.csv'")

    # Factor statistics
    for name, factor in [("Equal-weighted", equal_weighted), ("Value-weighted", value_weighted)]:
        stats = factor['ga_factor'].describe()
        annualized_mean = stats['mean'] * 12
        annualized_std = stats['std'] * np.sqrt(12)
        sharpe_ratio = annualized_mean / annualized_std if annualized_std != 0 else np.nan
        logger.info(f"{name} Sharpe Ratio: {sharpe_ratio:.4f}")
        logger.info(f"{name} {ga_column} factor stats (Size: {size_group if size_group else 'All'}):\n{stats}\n"
                    f"Annualized Mean: {annualized_mean:.4f}, Annualized Std: {annualized_std:.4f}")
        if stats['count'] < 12 or stats['std'] == 0:
            logger.warning(f"{name} factor might be unreliable: {stats}")

    return equal_weighted, value_weighted

##################################
### Main Pipeline
##################################

def main():
    try:
        ga_choices = [
            "goodwill_to_sales_lagged",
            "goodwill_to_equity_lagged",
            "goodwill_to_market_cap_lagged"
        ]
        all_factor_returns_list = []

        for ga_choice in ga_choices:
            print(f"\n🔄 Processing {ga_choice}...")
            df_base = load_processed_data(ga_choice=ga_choice)
            ga_lagged_diagnostics(df_base, ga_column=ga_choice)
            count_firms_per_year(df_base, ga_column=ga_choice)

            # Fama-French June-end Size Sorting (applied once, reused)
            df_base['month'] = df_base['crsp_date'].dt.month
            df_base['year'] = df_base['crsp_date'].dt.year
            june_me = df_base[df_base['month'] == 6][['permno', 'year', 'market_cap']].rename(columns={'market_cap': 'june_me'})
            nyse_medians = df_base[(df_base['month'] == 6) & (df_base['is_nyse'] == 1)].groupby('year')['market_cap'].median().reset_index()
            nyse_medians.rename(columns={'market_cap': 'nyse_median'}, inplace=True)
            df_base = df_base.merge(june_me, on=['permno', 'year'], how='left')
            df_base = df_base.merge(nyse_medians, on='year', how='left')
            df_base['size_group'] = np.where(df_base['june_me'] <= df_base['nyse_median'], 'Small', 'Big')
            df_base['june_me_ff_style'] = df_base.groupby('permno')['june_me'].ffill()

            # Single-sort (all firms)
            df_all = assign_ga_deciles(df_base.copy(), ga_column=ga_choice)
            ew_all, vw_all = create_ga_factor(df_all, ga_column=ga_choice)

            # Double-sort (Small and Big)
            factor_dict = {}
            for size_group in ['Small', 'Big']:
                df_size = assign_ga_deciles(df_base.copy(), ga_column=ga_choice, size_group=size_group)
                if 'decile' in df_size.columns and not df_size['decile'].isna().all():
                    ew_size, vw_size = create_ga_factor(df_size, ga_column=ga_choice, size_group=size_group)
                    factor_dict[size_group] = {'ew': ew_size, 'vw': vw_size}
                else:
                    logger.warning(f"⚠️ Skipping {ga_choice} - {size_group}: No valid deciles assigned")

            # Combine factor returns
            factor_returns = ew_all[['date']].copy()
            factor_returns[f'{ga_choice}_ew'] = ew_all['ga_factor']
            factor_returns[f'{ga_choice}_vw'] = vw_all['ga_factor']
            for size_group in ['Small', 'Big']:
                ew = factor_dict.get(size_group, {}).get('ew')
                vw = factor_dict.get(size_group, {}).get('vw')
                if ew is not None:
                    factor_returns[f'{ga_choice}_{size_group.lower()}_ew'] = ew['ga_factor']
                if vw is not None:
                    factor_returns[f'{ga_choice}_{size_group.lower()}_vw'] = vw['ga_factor']
            
            factor_returns.set_index('date', inplace=True)
            all_factor_returns_list.append(factor_returns)
            logger.info(f"✅ Finished processing {ga_choice} — rows: {df_base.shape[0]}")

        # Final Export
        all_factor_returns = pd.concat(all_factor_returns_list, axis=1, join='outer')
        all_factor_returns.reset_index(inplace=True)
        os.makedirs('output/factors', exist_ok=True)
        all_factor_returns.to_csv('output/factors/ga_factors.csv', index=False)
        logger.info("✅ All factor returns saved to 'output/factors/ga_factors.csv'")
        print("\n✅ All GA factors processed successfully!")

    except Exception as e:
        logger.error(f"❌ Pipeline failed: {e}")
        raise

if __name__ == "__main__":
    main()