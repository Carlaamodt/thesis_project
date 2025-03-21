import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
import glob
import datetime

# Set output directory
output_dir = "analysis_output"
os.makedirs(output_dir, exist_ok=True)

########################################
### Load Raw and Processed Data ###
########################################

def load_data(directory="data/"):
    files = {
        "compustat": load_latest_file(f"{directory}/compustat_*.csv"),
        "crsp_ret": load_latest_file(f"{directory}/crsp_ret_*.csv"),
        "crsp_delist": load_latest_file(f"{directory}/crsp_delist_*.csv"),
        "crsp_compustat": load_latest_file(f"{directory}/crsp_compustat_*.csv"),
        "processed": f"{directory}/processed_data.csv"
    }
    if None in files.values() or not os.path.exists(files["processed"]):
        raise FileNotFoundError(f"Missing files: {', '.join(k for k, v in files.items() if v is None or not os.path.exists(v))}")
    
    print("📥 Loading data...")
    compustat = pd.read_csv(files["compustat"], parse_dates=['date'])
    crsp_ret = pd.read_csv(files["crsp_ret"], parse_dates=['date'])
    crsp_delist = pd.read_csv(files["crsp_delist"], parse_dates=['dlstdt'])
    crsp_compustat = pd.read_csv(files["crsp_compustat"], parse_dates=['linkdt', 'linkenddt'])
    processed = files["processed"]
    
    return compustat, crsp_ret, crsp_delist, crsp_compustat, processed

def load_latest_file(pattern):
    files = sorted(glob.glob(pattern))
    return files[-1] if files else None

########################################
### Raw Data Overview ###
########################################

def raw_data_overview(compustat, crsp_ret, crsp_delist, crsp_compustat):
    print("\n🔍 Raw Data Overview:")
    
    print("\nCompustat:")
    print(f"Rows: {len(compustat)}, Unique gvkey: {compustat['gvkey'].nunique()}")
    print(f"Date range: {compustat['date'].min()} to {compustat['date'].max()}")
    goodwill_stats = {
        "Total firms": compustat['gvkey'].nunique(),
        "Firms with goodwill > 0": compustat.loc[compustat['gdwl'] > 0, 'gvkey'].nunique(),
        "Firms with goodwill NaN": compustat.loc[compustat['gdwl'].isna(), 'gvkey'].nunique()
    }
    print(f"Goodwill stats: {goodwill_stats}")
    print(f"Missing values (key vars):\n{compustat[['gdwl', 'at', 'ceq', 'csho']].isna().sum()}")

    print("\nCRSP Returns:")
    print(f"Rows: {len(crsp_ret)}, Unique permno: {crsp_ret['permno'].nunique()}")
    print(f"Date range: {crsp_ret['date'].min()} to {crsp_ret['date'].max()}")
    print(f"Missing returns: {crsp_ret['ret'].isna().sum()}")

    print("\nCRSP Delistings:")
    print(f"Rows: {len(crsp_delist)}, Unique permno: {crsp_delist['permno'].nunique()}")
    print(f"Date range: {crsp_delist['dlstdt'].min()} to {crsp_delist['dlstdt'].max()}")

    print("\nCRSP-Compustat Link:")
    print(f"Rows: {len(crsp_compustat)}, Unique gvkey-permno pairs: {crsp_compustat[['gvkey', 'permno']].drop_duplicates().shape[0]}")
    print(f"Link date range: {crsp_compustat['linkdt'].min()} to {crsp_compustat['linkenddt'].max()}")

########################################
### Processed Data Exploration ###
########################################

def processed_data_exploration(processed_path, chunk_size=500_000):
    print("\n🎯 Processed Data Exploration:")
    
    # Define dtypes for problematic columns (assuming naicsh, sich, etc.)
    dtypes = {
        'naicsh': 'str', 'sich': 'str', 'dlstcd': 'str', 'linktype': 'str'
    }
    
    # Basic stats with chunking
    total_rows, unique_gvkeys, unique_permnos, years = 0, set(), set(), set()
    for chunk in pd.read_csv(processed_path, chunksize=chunk_size, parse_dates=['crsp_date'], 
                            dtype=dtypes, low_memory=False):
        total_rows += len(chunk)
        unique_gvkeys.update(chunk['gvkey'].unique())
        unique_permnos.update(chunk['permno'].unique())
        years.update(chunk['crsp_date'].dt.year.unique())
    
    print(f"Rows: {total_rows}")
    print(f"Unique gvkeys: {len(unique_gvkeys)}")
    print(f"Unique permnos: {len(unique_permnos)}")
    print(f"Years covered: {sorted(years)}")

    # Variable coverage over time
    coverage = []
    for chunk in pd.read_csv(processed_path, chunksize=chunk_size, parse_dates=['crsp_date'], 
                            dtype=dtypes, low_memory=False):
        chunk_coverage = chunk.groupby(chunk['crsp_date'].dt.year).agg(
            firms=('gvkey', 'nunique'),
            returns_na=('ret', lambda x: x.isna().sum()),
            goodwill_na=('gdwl', lambda x: x.isna().sum())
        )
        coverage.append(chunk_coverage)
    coverage_df = pd.concat(coverage).groupby(level=0).sum()
    coverage_df.to_csv(f"{output_dir}/data_coverage_by_year.csv")
    print(f"\nCoverage by year saved to {output_dir}/data_coverage_by_year.csv")

    # Distribution plots (sample)
    sample_df = pd.read_csv(processed_path, nrows=100_000, parse_dates=['crsp_date'], 
                           dtype=dtypes, low_memory=False)
    plt.figure(figsize=(10, 6))
    sns.histplot(sample_df['ret'].dropna(), bins=50, kde=True)
    plt.title("Distribution of Monthly Returns (Sample)")
    plt.savefig(f"{output_dir}/returns_distribution.png")
    plt.close()

    plt.figure(figsize=(10, 6))
    sns.lineplot(data=coverage_df['firms'], label="Unique Firms")
    plt.title("Number of Firms Over Time")
    plt.savefig(f"{output_dir}/firms_over_time.png")
    plt.close()

########################################
### Additional Explorative Elements ###
########################################

def explorative_analysis(compustat, processed_path):
    print("\n🔬 Additional Explorations:")
    
    # Goodwill prevalence over time (raw data)
    goodwill_trend = compustat.groupby(compustat['date'].dt.year).agg(
        firms_with_gdwl=('gdwl', lambda x: (x > 0).sum()),
        total_firms=('gvkey', 'nunique')
    )
    goodwill_trend['pct_with_gdwl'] = goodwill_trend['firms_with_gdwl'] / goodwill_trend['total_firms']
    goodwill_trend.to_csv(f"{output_dir}/goodwill_prevalence.csv")
    print(f"Goodwill prevalence saved to {output_dir}/goodwill_prevalence.csv")

    # Exchange distribution (processed data, sample)
    dtypes = {'naicsh': 'str', 'sich': 'str', 'dlstcd': 'str', 'linktype': 'str'}
    sample_df = pd.read_csv(processed_path, nrows=100_000, parse_dates=['crsp_date'], 
                           dtype=dtypes, low_memory=False)
    exch_dist = sample_df.groupby('exchcd')['permno'].nunique().reset_index()
    exch_dist.columns = ['Exchange Code', 'Unique Permnos']
    exch_dist.to_csv(f"{output_dir}/exchange_distribution.csv", index=False)
    print(f"Exchange distribution (sample):\n{exch_dist}")

########################################
### Main Execution ###
########################################

def main():
    try:
        compustat, crsp_ret, crsp_delist, crsp_compustat, processed_path = load_data()
        raw_data_overview(compustat, crsp_ret, crsp_delist, crsp_compustat)
        processed_data_exploration(processed_path)
        explorative_analysis(compustat, processed_path)
        print("✅ Data overview and exploration completed!")
    except Exception as e:
        print(f"❌ Failed: {e}")
        raise

if __name__ == "__main__":
    main()