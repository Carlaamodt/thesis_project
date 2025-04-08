import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
import glob
import datetime

# Set output directory with subfolders
output_dir = "analysis_output/file4_analysis"
os.makedirs(output_dir, exist_ok=True)

# Subdirectories for each file's analysis
file1_dir = os.path.join(output_dir, "file1")
file2_dir = os.path.join(output_dir, "file2")
file3_dir = os.path.join(output_dir, "file3")
os.makedirs(file1_dir, exist_ok=True)
os.makedirs(file2_dir, exist_ok=True)
os.makedirs(file3_dir, exist_ok=True)

# Set seaborn style for better visuals
sns.set(style="whitegrid")

########################################
### Load Raw and Processed Data ###
########################################

def load_data(directory="data/"):
    files = {
        "compustat": load_latest_file(f"{directory}/compustat_*.csv"),
        "crsp_ret": load_latest_file(f"{directory}/crsp_ret_*.csv"),
        "crsp_delist": load_latest_file(f"{directory}/crsp_delist_*.csv"),
        "crsp_compustat": load_latest_file(f"{directory}/crsp_compustat_*.csv"),
        "processed": f"{directory}/processed_data.csv",
        "fama_french": f"{directory}/FamaFrench_factors_with_momentum.csv"
    }
    missing_files = [k for k, v in files.items() if v is None or not os.path.exists(v)]
    if missing_files:
        raise FileNotFoundError(f"Missing files: {', '.join(missing_files)}")
    
    print("📥 Loading data...")
    compustat = pd.read_csv(files["compustat"], parse_dates=['date'])
    crsp_ret = pd.read_csv(files["crsp_ret"], parse_dates=['date'])
    crsp_delist = pd.read_csv(files["crsp_delist"], parse_dates=['dlstdt'])
    crsp_compustat = pd.read_csv(files["crsp_compustat"], parse_dates=['linkdt', 'linkenddt'])
    processed = files["processed"]
    fama_french = pd.read_csv(files["fama_french"], parse_dates=['date'])
    
    return compustat, crsp_ret, crsp_delist, crsp_compustat, processed, fama_french

def load_latest_file(pattern):
    files = sorted(glob.glob(pattern))
    return files[-1] if files else None

########################################
### Raw Data Overview (File 1) ###
########################################

def raw_data_overview(compustat, crsp_ret, crsp_delist, crsp_compustat):
    print("\n🔍 Raw Data Overview (File 1 - Data Extraction):")
    
    # Compustat Overview
    print("\nCompustat:")
    print(f"Rows: {len(compustat)}, Unique gvkey: {compustat['gvkey'].nunique()}")
    print(f"Date range: {compustat['date'].min()} to {compustat['date'].max()}")
    goodwill_stats = {
        "Total firms": compustat['gvkey'].nunique(),
        "Firms with goodwill > 0": compustat.loc[compustat['gdwl'] > 0, 'gvkey'].nunique(),
        "Firms with goodwill NaN": compustat.loc[compustat['gdwl'].isna(), 'gvkey'].nunique()
    }
    print(f"Goodwill stats: {goodwill_stats}")
    
    # Print available columns to debug
    print("\nAvailable columns in Compustat:", compustat.columns.tolist())
    
    # Define key variables, excluding 'sale' if not present
    key_vars = ['gdwl', 'at', 'ceq', 'csho']
    available_vars = [var for var in key_vars if var in compustat.columns]
    if len(available_vars) < len(key_vars):
        print(f"⚠️ Some key variables are missing: {[var for var in key_vars if var not in compustat.columns]}")
    
    # Missing values for available key variables
    print(f"Missing values (key vars):\n{compustat[available_vars].isna().sum()}")
    
    # Summary statistics for available key variables
    compustat_stats = compustat[available_vars].describe()
    compustat_stats.to_csv(os.path.join(file1_dir, "compustat_key_vars_stats.csv"))
    print(f"\nCompustat key variables stats saved to {file1_dir}/compustat_key_vars_stats.csv")
    print(compustat_stats)

    # Plot distribution of goodwill
    plt.figure(figsize=(10, 6))
    sns.histplot(compustat[compustat['gdwl'] > 0]['gdwl'], bins=50, kde=True, log_scale=True)
    plt.title("Distribution of Goodwill (gdwl > 0, Log Scale)")
    plt.xlabel("Goodwill (Log Scale)")
    plt.ylabel("Frequency")
    plt.savefig(os.path.join(file1_dir, "goodwill_distribution.png"))
    plt.close()

    # CRSP Returns Overview
    print("\nCRSP Returns:")
    print(f"Rows: {len(crsp_ret)}, Unique permno: {crsp_ret['permno'].nunique()}")
    print(f"Date range: {crsp_ret['date'].min()} to {crsp_ret['date'].max()}")
    print(f"Missing returns: {crsp_ret['ret'].isna().sum()}")

    # CRSP Delistings Overview
    print("\nCRSP Delistings:")
    print(f"Rows: {len(crsp_delist)}, Unique permno: {crsp_delist['permno'].nunique()}")
    print(f"Date range: {crsp_delist['dlstdt'].min()} to {crsp_delist['dlstdt'].max()}")

    # CRSP-Compustat Link Overview
    print("\nCRSP-Compustat Link:")
    print(f"Rows: {len(crsp_compustat)}, Unique gvkey-permno pairs: {crsp_compustat[['gvkey', 'permno']].drop_duplicates().shape[0]}")
    print(f"Link date range: {crsp_compustat['linkdt'].min()} to {crsp_compustat['linkenddt'].max()}")

########################################
### Processed Data Exploration (File 2) ###
########################################

def processed_data_exploration(processed_path, chunk_size=500_000):
    print("\n🎯 Processed Data Exploration (File 2 - Data Processing):")
    
    dtypes = {
        'naicsh': 'str', 'sich': 'str', 'dlstcd': 'str', 'linktype': 'str'
    }
    
    # Basic stats
    total_rows, unique_gvkeys, unique_permnos, years = 0, set(), set(), set()
    for chunk in pd.read_csv(processed_path, chunksize=chunk_size, parse_dates=['crsp_date'], dtype=dtypes, low_memory=False):
        total_rows += len(chunk)
        unique_gvkeys.update(chunk['gvkey'].unique())
        unique_permnos.update(chunk['permno'].unique())
        years.update(chunk['crsp_date'].dt.year.unique())
    
    print(f"Rows: {total_rows}")
    print(f"Unique gvkeys: {len(unique_gvkeys)}")
    print(f"Unique permnos: {len(unique_permnos)}")
    print(f"Years covered: {sorted(years)}")

    # Coverage stats by year
    coverage = []
    for chunk in pd.read_csv(processed_path, chunksize=chunk_size, parse_dates=['crsp_date'], dtype=dtypes, low_memory=False):
        chunk_coverage = chunk.groupby(chunk['crsp_date'].dt.year).agg(
            firms=('gvkey', 'nunique'),
            returns_na=('ret', lambda x: x.isna().sum()),
            goodwill_na=('gdwl', lambda x: x.isna().sum())
        )
        coverage.append(chunk_coverage)
    coverage_df = pd.concat(coverage).groupby(level=0).sum()
    coverage_df.to_csv(os.path.join(file2_dir, "data_coverage_by_year.csv"))
    print(f"\nCoverage by year saved to {file2_dir}/data_coverage_by_year.csv")

    # Plot firm count over time
    plt.figure(figsize=(10, 6))
    sns.lineplot(data=coverage_df['firms'], label="Unique Firms")
    plt.title("Number of Firms Over Time")
    plt.xlabel("Year")
    plt.ylabel("Number of Firms")
    plt.savefig(os.path.join(file2_dir, "firms_over_time.png"))
    plt.close()

    # Non-zero GA1 (goodwill_to_sales_lagged) and GA2 (goodwill_to_equity_lagged)
    ga1_col = 'goodwill_to_sales_lagged'
    ga2_col = 'goodwill_to_equity_lagged'
    non_zero_counts_ga1 = []
    non_zero_counts_ga2 = []
    for chunk in pd.read_csv(processed_path, chunksize=chunk_size, parse_dates=['crsp_date'], dtype=dtypes, low_memory=False):
        # GA1
        if ga1_col not in chunk.columns:
            print(f"⚠️ Column '{ga1_col}' not found. Skipping GA1 analysis.")
        else:
            chunk['has_ga1'] = chunk[ga1_col].notna() & (chunk[ga1_col] != 0)
            chunk_counts_ga1 = chunk.groupby(chunk['crsp_date'].dt.year).agg(
                total_firms_ga1=('gvkey', 'nunique'),
                non_zero_ga1_firms=('gvkey', lambda x: x[chunk['has_ga1']].nunique())
            )
            non_zero_counts_ga1.append(chunk_counts_ga1)

        # GA2
        if ga2_col not in chunk.columns:
            print(f"⚠️ Column '{ga2_col}' not found. Skipping GA2 analysis.")
        else:
            chunk['has_ga2'] = chunk[ga2_col].notna() & (chunk[ga2_col] != 0)
            chunk_counts_ga2 = chunk.groupby(chunk['crsp_date'].dt.year).agg(
                total_firms_ga2=('gvkey', 'nunique'),
                non_zero_ga2_firms=('gvkey', lambda x: x[chunk['has_ga2']].nunique())
            )
            non_zero_counts_ga2.append(chunk_counts_ga2)

    # Save GA1 non-zero counts
    if non_zero_counts_ga1:
        non_zero_df_ga1 = pd.concat(non_zero_counts_ga1).groupby(level=0).sum()
        non_zero_df_ga1['pct_non_zero'] = non_zero_df_ga1['non_zero_ga1_firms'] / non_zero_df_ga1['total_firms_ga1']
        non_zero_df_ga1.to_csv(os.path.join(file2_dir, "firms_unique_goodwill_to_sales_lagged_per_year.csv"))
        print(f"\nNon-zero GA1 (goodwill_to_sales_lagged) firms per year saved to {file2_dir}/firms_unique_goodwill_to_sales_lagged_per_year.csv")
        print(non_zero_df_ga1)

    # Save GA2 non-zero counts
    if non_zero_counts_ga2:
        non_zero_df_ga2 = pd.concat(non_zero_counts_ga2).groupby(level=0).sum()
        non_zero_df_ga2['pct_non_zero'] = non_zero_df_ga2['non_zero_ga2_firms'] / non_zero_df_ga2['total_firms_ga2']
        non_zero_df_ga2.to_csv(os.path.join(file2_dir, "firms_unique_goodwill_to_equity_lagged_per_year.csv"))
        print(f"\nNon-zero GA2 (goodwill_to_equity_lagged) firms per year saved to {file2_dir}/firms_unique_goodwill_to_equity_lagged_per_year.csv")
        print(non_zero_df_ga2)

    # Sample for detailed analysis
    sample_df = pd.read_csv(processed_path, nrows=100_000, parse_dates=['crsp_date'], dtype=dtypes, low_memory=False)

    # Plot return distribution
    plt.figure(figsize=(10, 6))
    sns.histplot(sample_df['ret'].dropna(), bins=50, kde=True)
    plt.title("Distribution of Monthly Returns (Sample)")
    plt.xlabel("Monthly Return")
    plt.ylabel("Frequency")
    plt.savefig(os.path.join(file2_dir, "returns_distribution.png"))
    plt.close()

    # Distributions of GA1 and GA2
    if ga1_col in sample_df.columns:
        plt.figure(figsize=(10, 6))
        sns.histplot(sample_df[sample_df[ga1_col].notna()][ga1_col], bins=50, kde=True, log_scale=True)
        plt.title("Distribution of GA1 (goodwill_to_sales_lagged, Log Scale)")
        plt.xlabel("GA1 (Log Scale)")
        plt.ylabel("Frequency")
        plt.savefig(os.path.join(file2_dir, "ga1_distribution.png"))
        plt.close()

    if ga2_col in sample_df.columns:
        plt.figure(figsize=(10, 6))
        sns.histplot(sample_df[sample_df[ga2_col].notna()][ga2_col], bins=50, kde=True, log_scale=True)
        plt.title("Distribution of GA2 (goodwill_to_equity_lagged, Log Scale)")
        plt.xlabel("GA2 (Log Scale)")
        plt.ylabel("Frequency")
        plt.savefig(os.path.join(file2_dir, "ga2_distribution.png"))
        plt.close()

    # Trends of GA1 and GA2 over time
    trends = []
    for chunk in pd.read_csv(processed_path, chunksize=chunk_size, parse_dates=['crsp_date'], dtype=dtypes, low_memory=False):
        chunk['year'] = chunk['crsp_date'].dt.year
        chunk_trends = chunk.groupby('year').agg(
            mean_ga1=(ga1_col, 'mean') if ga1_col in chunk.columns else None,
            mean_ga2=(ga2_col, 'mean') if ga2_col in chunk.columns else None
        )
        trends.append(chunk_trends)
    trends_df = pd.concat(trends).groupby(level=0).mean()
    trends_df.to_csv(os.path.join(file2_dir, "ga1_ga2_trends_over_time.csv"))
    print(f"\nGA1 and GA2 trends over time saved to {file2_dir}/ga1_ga2_trends_over_time.csv")

    plt.figure(figsize=(10, 6))
    if ga1_col in sample_df.columns:
        sns.lineplot(data=trends_df['mean_ga1'], label="Mean GA1 (goodwill_to_sales_lagged)")
    if ga2_col in sample_df.columns:
        sns.lineplot(data=trends_df['mean_ga2'], label="Mean GA2 (goodwill_to_equity_lagged)")
    plt.title("Average GA1 and GA2 Over Time")
    plt.xlabel("Year")
    plt.ylabel("Mean Value")
    plt.legend()
    plt.savefig(os.path.join(file2_dir, "ga1_ga2_trends_over_time.png"))
    plt.close()

    # Relationship between GA1, GA2, and returns
    if ga1_col in sample_df.columns and 'ret' in sample_df.columns:
        plt.figure(figsize=(10, 6))
        sns.scatterplot(x=sample_df[ga1_col], y=sample_df['ret'], alpha=0.5)
        plt.xscale('log')
        plt.title("GA1 (goodwill_to_sales_lagged) vs. Monthly Returns")
        plt.xlabel("GA1 (Log Scale)")
        plt.ylabel("Monthly Return")
        plt.savefig(os.path.join(file2_dir, "ga1_vs_returns_scatter.png"))
        plt.close()

    if ga2_col in sample_df.columns and 'ret' in sample_df.columns:
        plt.figure(figsize=(10, 6))
        sns.scatterplot(x=sample_df[ga2_col], y=sample_df['ret'], alpha=0.5)
        plt.xscale('log')
        plt.title("GA2 (goodwill_to_equity_lagged) vs. Monthly Returns")
        plt.xlabel("GA2 (Log Scale)")
        plt.ylabel("Monthly Return")
        plt.savefig(os.path.join(file2_dir, "ga2_vs_returns_scatter.png"))
        plt.close()

    # Box plots: Returns for high vs. low GA1 and GA2
    if ga1_col in sample_df.columns:
        sample_df['ga1_quartile'] = pd.qcut(sample_df[ga1_col], 4, labels=['Q1', 'Q2', 'Q3', 'Q4'], duplicates='drop')
        plt.figure(figsize=(10, 6))
        sns.boxplot(x='ga1_quartile', y='ret', data=sample_df)
        plt.title("Monthly Returns by GA1 Quartiles")
        plt.xlabel("GA1 Quartile")
        plt.ylabel("Monthly Return")
        plt.savefig(os.path.join(file2_dir, "returns_by_ga1_quartiles.png"))
        plt.close()

    if ga2_col in sample_df.columns:
        sample_df['ga2_quartile'] = pd.qcut(sample_df[ga2_col], 4, labels=['Q1', 'Q2', 'Q3', 'Q4'], duplicates='drop')
        plt.figure(figsize=(10, 6))
        sns.boxplot(x='ga2_quartile', y='ret', data=sample_df)
        plt.title("Monthly Returns by GA2 Quartiles")
        plt.xlabel("GA2 Quartile")
        plt.ylabel("Monthly Return")
        plt.savefig(os.path.join(file2_dir, "returns_by_ga2_quartiles.png"))
        plt.close()

    # Compare goodwill vs non-goodwill firms
    sample_df['has_goodwill'] = sample_df['gdwl'] > 0
    if 'market_cap' in sample_df.columns:
        goodwill_stats = sample_df.groupby('has_goodwill').agg(
            mean_ret=('ret', 'mean'),
            std_ret=('ret', 'std'),
            mean_market_cap=('market_cap', 'mean'),
            count=('gvkey', 'count')
        )
        goodwill_stats.to_csv(os.path.join(file2_dir, "goodwill_vs_non_goodwill_stats.csv"))
        print(f"\nZero vs. Non-Zero Goodwill Firms (Sample):\n{goodwill_stats}")

########################################
### Fama-French Factor Analysis (File 3) ###
########################################

def fama_french_analysis(fama_french):
    print("\n📈 Fama-French Factor Analysis (File 3 - Download Fama-French):")
    
    # Basic stats
    print(f"Rows: {len(fama_french)}")
    print(f"Date range: {fama_french['date'].min()} to {fama_french['date'].max()}")
    factor_cols = ['mkt_rf', 'smb', 'hml', 'rmw', 'cma', 'mom', 'rf']
    print(f"\nMissing values:\n{fama_french[factor_cols].isna().sum()}")
    
    # Summary statistics
    ff_stats = fama_french[factor_cols].describe()
    ff_stats.to_csv(os.path.join(file3_dir, "fama_french_stats.csv"))
    print(f"\nFama-French factors summary stats saved to {file3_dir}/fama_french_stats.csv")
    print(ff_stats)

    # Extreme values
    for col in factor_cols:
        print(f"\n{col.upper()} — Top 5:")
        print(fama_french[[col]].sort_values(by=col, ascending=False).head(5))
        print(f"{col.upper()} — Bottom 5:")
        print(fama_french[[col]].sort_values(by=col).head(5))

    # Time series plots (12-month rolling average)
    for col in factor_cols:
        fama_french[f'{col}_roll12'] = fama_french[col].rolling(window=12).mean()
        plt.figure(figsize=(10, 6))
        sns.lineplot(x='date', y=f'{col}_roll12', data=fama_french, label=f'12-mo Rolling Mean: {col.upper()}')
        plt.axhline(0, color='gray', linestyle='--')
        plt.title(f'Rolling 12-Month Average of {col.upper()}')
        plt.xlabel('Date')
        plt.ylabel(f'{col.upper()} (12-mo Avg)')
        plt.legend()
        plt.savefig(os.path.join(file3_dir, f"{col}_rolling_avg.png"))
        plt.close()

    # Distribution plots
    for col in factor_cols:
        plt.figure(figsize=(10, 6))
        sns.histplot(fama_french[col].dropna(), bins=50, kde=True)
        plt.title(f"Distribution of {col.upper()}")
        plt.xlabel(col.upper())
        plt.ylabel("Frequency")
        plt.savefig(os.path.join(file3_dir, f"{col}_distribution.png"))
        plt.close()

    # Correlation heatmap
    plt.figure(figsize=(10, 8))
    sns.heatmap(fama_french[factor_cols].corr(), annot=True, cmap='coolwarm', vmin=-1, vmax=1)
    plt.title("Correlation Heatmap of Fama-French Factors")
    plt.savefig(os.path.join(file3_dir, "fama_french_correlation_heatmap.png"))
    plt.close()

########################################
### Additional Explorative Elements ###
########################################

def explorative_analysis(compustat, processed_path):
    print("\n🔬 Additional Explorations:")
    
    # Goodwill prevalence over time
    goodwill_trend = compustat.groupby(compustat['date'].dt.year).agg(
        firms_with_gdwl=('gdwl', lambda x: (x > 0).sum()),
        total_firms=('gvkey', 'nunique')
    )
    goodwill_trend['pct_with_gdwl'] = goodwill_trend['firms_with_gdwl'] / goodwill_trend['total_firms']
    goodwill_trend.to_csv(os.path.join(file1_dir, "goodwill_prevalence.csv"))
    print(f"Goodwill prevalence saved to {file1_dir}/goodwill_prevalence.csv")

    # Exchange distribution
    dtypes = {'naicsh': 'str', 'sich': 'str', 'dlstcd': 'str', 'linktype': 'str'}
    sample_df = pd.read_csv(processed_path, nrows=100_000, parse_dates=['crsp_date'], dtype=dtypes, low_memory=False)
    exch_dist = sample_df.groupby('exchcd')['permno'].nunique().reset_index()
    exch_dist.columns = ['Exchange Code', 'Unique Permnos']
    exch_dist.to_csv(os.path.join(file2_dir, "exchange_distribution.csv"), index=False)
    print(f"Exchange distribution (sample):\n{exch_dist}")

    # Goodwill prevalence by industry (using naicsh)
    if 'naicsh' in sample_df.columns:
        industry_goodwill = sample_df.groupby('naicsh').agg(
            total_firms=('gvkey', 'nunique'),
            firms_with_gdwl=('gdwl', lambda x: (x > 0).sum())
        )
        industry_goodwill['pct_with_gdwl'] = industry_goodwill['firms_with_gdwl'] / industry_goodwill['total_firms']
        industry_goodwill = industry_goodwill.sort_values('pct_with_gdwl', ascending=False)
        industry_goodwill.to_csv(os.path.join(file2_dir, "goodwill_by_industry.csv"))
        print(f"Goodwill prevalence by industry saved to {file2_dir}/goodwill_by_industry.csv")

########################################
### Main Execution ###
########################################

def main():
    try:
        compustat, crsp_ret, crsp_delist, crsp_compustat, processed_path, fama_french = load_data()
        raw_data_overview(compustat, crsp_ret, crsp_delist, crsp_compustat)
        processed_data_exploration(processed_path)
        fama_french_analysis(fama_french)
        explorative_analysis(compustat, processed_path)
        print("✅ Data overview and exploration completed!")
    except Exception as e:
        print(f"❌ Failed: {e}")
        raise

if __name__ == "__main__":
    main()