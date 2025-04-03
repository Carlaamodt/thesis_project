import pandas as pd
import numpy as np
import os
import logging

# ----------------------------------------
# Setup Logging
# ----------------------------------------

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger()
SAVE_INTERMEDIATE_FILES = True  # Save decile assignments and factor returns

# ----------------------------------------
# Load Processed Data
# ----------------------------------------

def load_processed_data(directory="data/", filename="processed_data.csv", ga_choice="goodwill_to_sales_lagged"):
    filepath = os.path.join(directory, filename)
    logger.info(f"Loading processed data from {filepath} with {ga_choice} ...")
    
    required_cols = [
        'permno', 'gvkey', 'crsp_date', 'FF_year',
        ga_choice, 'ret', 'prc', 'csho', 'at',
        'market_cap', 'is_nyse'
    ]
    
    try:
        df = pd.read_csv(filepath, parse_dates=['crsp_date'], usecols=required_cols, low_memory=False)
        df = df.drop_duplicates(subset=['permno', 'crsp_date'])
        df = df[df['crsp_date'].dt.year < 2024]  # Drop 2024-forward
        logger.info(f"Loaded rows: {len(df)}, unique permno: {df['permno'].nunique()}")
        logger.info(f"Years present: {sorted(df['crsp_date'].dt.year.unique())}")
        if len(df) < 100000:
            logger.warning(f"Low row count: {len(df)}")
        return df
    except Exception as e:
        logger.error(f"Failed to load data: {e}")
        raise

# ----------------------------------------
# GA Diagnostics
# ----------------------------------------

def ga_lagged_diagnostics(df, ga_column):
    logger.info(f"Running diagnostics for: {ga_column}")
    df_goodwill = df[df[ga_column].notna()].copy()

    if df_goodwill.empty:
        logger.error(f"No firms with {ga_column} data")
        raise ValueError(f"No valid data for {ga_column}")

    print(f"\n📊 Goodwill firms: {df_goodwill['permno'].nunique()} (rows: {len(df_goodwill)})")
    nans = df_goodwill[ga_column].isna().sum()
    df_goodwill[ga_column] = df_goodwill[ga_column].fillna(0)
    zeros = df_goodwill[ga_column].eq(0).sum()
    non_zeros = df_goodwill[ga_column].ne(0).sum()
    
    df_goodwill[ga_column] = df_goodwill[ga_column].clip(lower=-10, upper=10)

    print(f"❌ Missing {ga_column} (pre-fill NaN): {nans}")
    print(f"🔢 Zero {ga_column} (incl. filled NaN): {zeros}")
    print(f"🔢 Non-zero {ga_column}: {non_zeros}")
    print(f"🆔 Unique {ga_column}: total={df_goodwill[ga_column].nunique()}, non-zero={df_goodwill[df_goodwill[ga_column] != 0][ga_column].nunique()}")
    print(f"✅ Row check: {nans + (zeros - nans) + non_zeros} == {len(df_goodwill)}")

    if non_zeros > 0:
        stats = df_goodwill[ga_column].describe(percentiles=[0.01, 0.05, 0.25, 0.5, 0.75, 0.95, 0.99])
        extremes = df_goodwill[ga_column].quantile([0.001, 0.999]).to_dict()
        print(f"⚠️ Extremes (0.1%, 99.9%): {extremes}")
        print(f"\n📊 {ga_column} Stats:\n{stats}")
    print("✅ Diagnostics Complete.\n")

# ----------------------------------------
# Count Firms Per Year
# ----------------------------------------

def count_firms_per_year(df, ga_column):
    logger.info(f"Counting firms by FF_year for: {ga_column}")
    df = df[df['FF_year'] < 2024].copy()
    
    firm_counts = df[df[ga_column].notna()].groupby('FF_year').agg(
        num_firms=('permno', 'nunique'),
        num_unique_ga=(ga_column, 'nunique'),
        num_rows=('permno', 'count')
    ).reset_index()

    if firm_counts['num_firms'].min() < 20:
        logger.warning("Years with <20 firms: %s", firm_counts[firm_counts['num_firms'] < 20]['FF_year'].tolist())

    logger.info("Firms per FF_year:\n%s", firm_counts)

    os.makedirs("analysis_output", exist_ok=True)
    output_path = f"analysis_output/firms_unique_{ga_column}_per_year.xlsx"
    firm_counts.to_excel(output_path, index=False)
    logger.info(f"Saved firm counts to {output_path}")

    return firm_counts

# ----------------------------------------
# Assign Deciles by GA Variable
# ----------------------------------------

def assign_ga_deciles(df, ga_column, size_group=None):
    size_tag = size_group if size_group else 'All'
    logger.info(f"Assigning deciles for {ga_column} (Size: {size_tag})")
    
    working_df = df.copy()
    if size_group:
        working_df = working_df[working_df['size_group'] == size_group].copy()

    decile_assignments = []

    for year, group in working_df.groupby('FF_year'):
        filtered = group[
            group[ga_column].notna() &
            (group[ga_column] > 0) &
            (group['at'] > 0)
        ].copy()
        
        logger.info(f"Year {year}: Valid firms: {len(filtered)} / Excluded: {group.shape[0] - filtered.shape[0]}")

        if len(filtered) >= 20 and filtered[ga_column].nunique() > 5:
            try:
                filtered['decile'] = pd.qcut(filtered[ga_column], 10, labels=False, duplicates='drop') + 1
                filtered['decile'] = filtered['decile'].astype(int)
                decile_assignments.append(filtered[['permno', 'crsp_date', 'decile']])
            except Exception as e:
                logger.warning(f"Year {year}: Decile assignment failed — {e}")
        else:
            logger.warning(f"Year {year}: Insufficient firms or GA variation")

    if decile_assignments:
        deciles_df = pd.concat(decile_assignments, axis=0)
        df = df.drop(columns=['decile'], errors='ignore')
        df = df.merge(deciles_df, on=['permno', 'crsp_date'], how='left')
        
        if df['decile'].notna().any():
            logger.info(f"✅ Decile assignment complete. Distribution:\n{df['decile'].value_counts().sort_index()}")
        else:
            logger.error("⚠️ All deciles are NaN!")

        if SAVE_INTERMEDIATE_FILES:
            os.makedirs("analysis_output", exist_ok=True)
            output_name = f"decile_assignments_{ga_column}_{size_tag.lower()}.csv"
            df.to_csv(f"analysis_output/{output_name}", index=False)
            logger.info(f"Saved deciles to analysis_output/{output_name}")
    else:
        logger.warning(f"⚠️ No deciles assigned for {ga_column} (Size: {size_tag})")

    return df

# ----------------------------------------
# Create GA Factor Returns (EW & VW)
# ----------------------------------------

def create_ga_factor(df, ga_column, size_group=None):
    label = size_group if size_group else "All"
    logger.info(f"Creating GA factor returns — {ga_column} (Size: {label})")

    df = df[df['decile'].isin([1, 10])].copy()
    df = df[df['ret'].notna() & (df['ret'].abs() <= 1)]
    df['crsp_date'] = pd.to_datetime(df['crsp_date']) + pd.offsets.MonthEnd(0)
    df = df[df['crsp_date'].dt.month != 6]  # Drop June to match Fama-French
    df['ME'] = df['june_me_ff_style']
    df = df[df['ME'].notna() & (df['ME'] > 0)]

    if df.empty:
        logger.warning(f"No data for {ga_column} (Size: {label})")
        return pd.DataFrame(), pd.DataFrame()

    # Equal-weighted
    ew = df.groupby(['crsp_date', 'decile'])['ret'].mean().unstack()
    ew['ga_factor'] = ew[1] - ew[10]
    ew = ew[['ga_factor']].dropna().rename_axis('date').reset_index()

    # Value-weighted
    def vw_func(x):
        return np.average(x['ret'], weights=x['ME']) if x['ME'].sum() > 0 else np.nan
    
    vw = df.groupby(['crsp_date', 'decile'], group_keys=False)[['ret', 'ME']].apply(vw_func, include_groups=False).unstack()
    vw['ga_factor'] = vw[1] - vw[10]
    vw = vw[['ga_factor']].dropna().rename_axis('date').reset_index()

    # Save if needed
    suffix = f"_{size_group.lower()}" if size_group else ""
    if SAVE_INTERMEDIATE_FILES:
        os.makedirs('output/factors', exist_ok=True)
        ew.to_csv(f'output/factors/ga_factor_returns_monthly_equal_{ga_column}{suffix}.csv', index=False)
        vw.to_csv(f'output/factors/ga_factor_returns_monthly_value_{ga_column}{suffix}.csv', index=False)
        logger.info(f"Saved EW to 'ga_factor_returns_monthly_equal_{ga_column}{suffix}.csv'")
        logger.info(f"Saved VW to 'ga_factor_returns_monthly_value_{ga_column}{suffix}.csv'")

    # Stats
    for tag, factor in [("Equal-weighted", ew), ("Value-weighted", vw)]:
        stats = factor['ga_factor'].describe()
        sharpe = (stats['mean'] * 12) / (stats['std'] * np.sqrt(12)) if stats['std'] > 0 else np.nan
        logger.info(f"{tag} {ga_column} (Size: {label}) — Sharpe: {sharpe:.4f}, N={int(stats['count'])}")

    return ew, vw

# ----------------------------------------
# Size-adjusted GA Factor (10x2)
# ----------------------------------------

def create_ga_factor_size_adjusted(df, ga_column):
    logger.info(f"Creating 10x2 size-adjusted GA factor for {ga_column}...")

    # Ensure 'ME' column exists and filter data
    df = df[df['decile'].isin([1, 10])].copy()
    df = df[df['ret'].notna() & (df['ret'].abs() <= 1)]
    df['crsp_date'] = pd.to_datetime(df['crsp_date']) + pd.offsets.MonthEnd(0)
    df = df[df['crsp_date'].dt.month != 6]  # Drop June to match Fama-French
    df['ME'] = df['june_me_ff_style']  # Set 'ME' from 'june_me_ff_style'
    df = df[df['ME'].notna() & (df['ME'] > 0)]  # Filter for valid ME

    if df.empty:
        logger.warning(f"No data for size-adjusted {ga_column}")
        return pd.DataFrame(), pd.DataFrame()

    # Group and pivot for equal-weighted returns
    ew = df.groupby(['crsp_date', 'decile', 'size_group']).agg(
        ret_mean=('ret', 'mean'),
        num_firms=('permno', 'count')
    ).reset_index()

    # Pivot so we get (decile, size_group) as columns
    ew = ew.pivot(index='crsp_date', columns=['decile', 'size_group'], values='ret_mean')

    # Handle missing combinations with .get() and fallbacks
    d1_small = ew.get((1, 'Small'), pd.Series(index=ew.index))
    d1_big   = ew.get((1, 'Big'), pd.Series(index=ew.index))
    d10_small = ew.get((10, 'Small'), pd.Series(index=ew.index))
    d10_big   = ew.get((10, 'Big'), pd.Series(index=ew.index))

    # Compute D1-D10 factor
    ew['ga_factor'] = ((d1_small + d1_big) / 2) - ((d10_small + d10_big) / 2)
    ew = ew[['ga_factor']].dropna().reset_index().rename(columns={'crsp_date': 'date'})

    # Value-weighted version
    def weighted_ret(x):
        if len(x) < 2:  # Changed to 2
            logger.warning(f"Group {x.name} has too few firms: {len(x)}")
            return np.nan
        if x['ME'].sum() <= 0:
            logger.warning(f"Group {x.name} has invalid ME sum: {x['ME'].sum()}")
            logger.warning(f"ME values in group {x.name}:\n{x['ME']}")
            return np.nan
        return np.average(x['ret'], weights=x['ME'])

    vw = df.groupby(['crsp_date', 'decile', 'size_group']).apply(weighted_ret, include_groups=False).unstack()
    
    # Handle missing combinations safely
    d1_small_vw = vw.get((1, 'Small'), pd.Series(index=vw.index))
    d1_big_vw   = vw.get((1, 'Big'), pd.Series(index=vw.index))
    d10_small_vw = vw.get((10, 'Small'), pd.Series(index=vw.index))
    d10_big_vw   = vw.get((10, 'Big'), pd.Series(index=vw.index))

    # Compute D1-D10 factor
    vw['ga_factor'] = ((d1_small_vw + d1_big_vw) / 2) - ((d10_small_vw + d10_big_vw) / 2)
    # Impute missing values
    vw['ga_factor'] = vw['ga_factor'].fillna(method='ffill').fillna(method='bfill')
    vw = vw[['ga_factor']].reset_index().rename(columns={'crsp_date': 'date'})

    # Save intermediate results
    if SAVE_INTERMEDIATE_FILES:
        os.makedirs('output/factors', exist_ok=True)
        ew.to_csv(f'output/factors/ga_factor_returns_monthly_equal_{ga_column}_sizeadj.csv', index=False)
        vw.to_csv(f'output/factors/ga_factor_returns_monthly_value_{ga_column}_sizeadj.csv', index=False)
        logger.info(f"Saved 10x2 EW to 'output/factors/ga_factor_returns_monthly_equal_{ga_column}_sizeadj.csv'")
        logger.info(f"Saved 10x2 VW to 'output/factors/ga_factor_returns_monthly_value_{ga_column}_sizeadj.csv'")

    # Factor stats
    for name, factor in [("Size-adjusted EW", ew), ("Size-adjusted VW", vw)]:
        stats = factor['ga_factor'].describe()
        ann_mean = stats['mean'] * 12
        ann_std = stats['std'] * np.sqrt(12)
        sharpe = ann_mean / ann_std if ann_std != 0 else np.nan
        logger.info(f"{name} Sharpe Ratio: {sharpe:.4f}")
        logger.info(f"{name} {ga_column} factor stats:\n{stats}\nAnnualized Mean: {ann_mean:.4f}, Annualized Std: {ann_std:.4f}")

    return ew, vw

# ----------------------------------------
# Main Pipeline
# ----------------------------------------

def main():
    try:
        ga_columns = [
            "goodwill_to_sales_lagged",
            "goodwill_to_equity_lagged",
            "goodwill_to_market_cap_lagged"
        ]

        all_factor_dfs = []

        for ga_col in ga_columns:
            print(f"\n🔄 Processing GA variable: {ga_col}")
            df = load_processed_data(ga_choice=ga_col)
            ga_lagged_diagnostics(df, ga_col)
            count_firms_per_year(df, ga_col)

            # Apply Fama-French style June size sort
            df['month'] = df['crsp_date'].dt.month
            df['year'] = df['crsp_date'].dt.year
            june_me = df[df['month'] == 6][['permno', 'year', 'market_cap']].rename(columns={'market_cap': 'june_me'})
            nyse_median = df[(df['month'] == 6) & (df['is_nyse'] == 1)].groupby('year')['market_cap'].median().reset_index()
            nyse_median.rename(columns={'market_cap': 'nyse_median'}, inplace=True)
            df = df.merge(june_me, on=['permno', 'year'], how='left')
            df = df.merge(nyse_median, on='year', how='left')
            df['size_group'] = np.where(df['june_me'] <= df['nyse_median'], 'Small', 'Big')
            df['june_me_ff_style'] = df.groupby('permno')['june_me'].ffill()

            # 🟢 Method 1: Single-sort
            df_single = assign_ga_deciles(df.copy(), ga_col)
            ew_single, vw_single = create_ga_factor(df_single, ga_col)

            # 🟢 Method 2: Double-sort (Small/Big)
            df_small = assign_ga_deciles(df.copy(), ga_col, size_group='Small')
            df_big = assign_ga_deciles(df.copy(), ga_col, size_group='Big')

            ew_small, vw_small = create_ga_factor(df_small, ga_col, size_group='Small')
            ew_big, vw_big = create_ga_factor(df_big, ga_col, size_group='Big')

            # 🟢 Method 3: Size-adjusted 10×2
            df_10x2 = assign_ga_deciles(df.copy(), ga_col)
            ew_sizeadj, vw_sizeadj = create_ga_factor_size_adjusted(df_10x2, ga_col)

            # Ensure 'date' columns are consistently formatted
            for factor_df in [ew_single, vw_single, ew_small, vw_small, ew_big, vw_big, ew_sizeadj, vw_sizeadj]:
                if not factor_df.empty:
                    factor_df['date'] = pd.to_datetime(factor_df['date']).dt.normalize()  # Remove time zone and microseconds

            # 📦 Combine all into single DataFrame
            factor_df = ew_single[['date']].copy()
            factor_df[f'{ga_col}_ew'] = ew_single['ga_factor']
            factor_df[f'{ga_col}_vw'] = vw_single['ga_factor']
            factor_df[f'{ga_col}_small_ew'] = ew_small['ga_factor']
            factor_df[f'{ga_col}_small_vw'] = vw_small['ga_factor']
            factor_df[f'{ga_col}_big_ew'] = ew_big['ga_factor']
            factor_df[f'{ga_col}_big_vw'] = vw_big['ga_factor']
            factor_df[f'{ga_col}_sizeadj_ew'] = ew_sizeadj['ga_factor']
            factor_df[f'{ga_col}_sizeadj_vw'] = vw_sizeadj['ga_factor']

            # Debug: Inspect factor_df
            logger.info(f"factor_df for {ga_col} - Shape: {factor_df.shape}")
            logger.info(f"factor_df for {ga_col} - Date range: {factor_df['date'].min()} to {factor_df['date'].max()}")
            logger.info(f"factor_df for {ga_col} - First few rows:\n{factor_df.head()}")
            # Add: Check non-NaN counts for each column
            logger.info(f"factor_df for {ga_col} - Non-NaN counts:\n{factor_df.notna().sum()}")

            factor_df['date'] = pd.to_datetime(factor_df['date']).dt.normalize()
            factor_df.set_index('date', inplace=True)
            all_factor_dfs.append(factor_df)

        # 🔄 Combine all GA variables across columns
        final_df = pd.concat(all_factor_dfs, axis=1, join='outer').reset_index()

        # Debug: Inspect final_df
        logger.info(f"final_df - Shape: {final_df.shape}")
        logger.info(f"final_df - Columns: {final_df.columns.tolist()}")
        logger.info(f"final_df - First few rows:\n{final_df.head()}")
        # Add: Check non-NaN counts for each column
        logger.info(f"final_df - Non-NaN counts:\n{final_df.notna().sum()}")
        # Add: Check data types
        logger.info(f"final_df - Data types:\n{final_df.dtypes}")

        # 💾 Export to ga.factors
        os.makedirs('output/factors', exist_ok=True)
        output_path = '/Users/carlaamodt/thesis_project/ga.factors.csv'  # Changed to .csv
        final_df.to_csv(output_path, index=False, encoding='utf-8', lineterminator='\n')
        logger.info(f"✅ All GA factors saved to '{output_path}'")
        print("\n✅ All GA factors processed successfully!")

    except Exception as e:
        logger.error(f"❌ Pipeline failed: {e}")
        raise

if __name__ == "__main__":
    main()