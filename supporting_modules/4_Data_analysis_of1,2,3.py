import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
import glob
import logging
from scipy.interpolate import make_interp_spline

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(name)s - %(message)s',
    handlers=[
        logging.FileHandler('file4_analysis.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# Set output directory
output_dir = "/Users/carlaamodt/thesis_project/File_4_analysis"
os.makedirs(output_dir, exist_ok=True)
logger.info(f"Output directory set to {output_dir}")

# Set seaborn style
sns.set(style="white")

# Counter for numbering outputs
output_counter = 1

def get_output_filename(prefix):
    """Generate unique filename with counter and prefix."""
    global output_counter
    filename = f"file4_{output_counter:03d}_{prefix}"
    output_counter += 1
    logger.debug(f"Generated filename: {filename}")
    return filename

# Color palette for 10 industries
colors_10 = sns.color_palette("tab10", 10)

########################################
### Load Raw and Processed Data ###
########################################

def load_ff48_mapping_from_excel(filepath: str) -> dict:
    """Load Fama-French 48 industry classification mapping from Excel."""
    logger.info(f"Loading FF48 mapping from {filepath}")
    if not os.path.exists(filepath):
        logger.error(f"Excel file not found: {filepath}")
        raise FileNotFoundError(f"Excel file not found: {filepath}")
    
    df = pd.read_excel(filepath)
    required_cols = ['Industry number', 'Industry code', 'Industry description']
    if not all(col in df.columns for col in required_cols):
        logger.error(f"Missing required columns in {filepath}: {required_cols}")
        raise ValueError(f"Excel file must contain columns: {required_cols}")
    
    df['Industry number'] = df['Industry number'].ffill()
    ff48_mapping = {}
    for _, row in df.iterrows():
        code = row['Industry code']
        industry_number = row['Industry number']
        if isinstance(code, str) and '-' in code:
            try:
                start, end = map(int, code.strip().split('-'))
                ff48_mapping[range(start, end + 1)] = int(industry_number) if pd.notna(industry_number) else 48
            except ValueError as e:
                logger.warning(f"Skipping invalid SIC range '{code}': {e}")
    logger.info(f"Loaded {len(ff48_mapping)} SIC ranges into FF48 mapping")
    return ff48_mapping

def map_sic_to_ff48(sic, ff48_mapping):
    """Map a 4-digit SIC code to Fama-French 48 industry."""
    try:
        if pd.isna(sic):
            return 48
        sic_str = str(sic).split('.')[0]
        sic_int = int(sic_str) if sic_str.isdigit() else 0
        for sic_range, industry in ff48_mapping.items():
            if sic_int in sic_range:
                return industry
        return 48
    except (ValueError, TypeError) as e:
        logger.warning(f"Invalid SIC code '{sic}': {e}. Assigning to 'Other' (48)")
        return 48

def map_ff48_to_10_industries(ff48_id):
    """Map Fama-French 48 industries to 10 broader categories."""
    mapping = {
        1: "Consumer Goods", 2: "Consumer Goods", 3: "Consumer Goods", 4: "Consumer Goods",
        5: "Consumer Goods", 9: "Consumer Goods", 10: "Consumer Goods", 16: "Consumer Goods",
        11: "Healthcare", 12: "Healthcare", 13: "Healthcare",
        14: "Manufacturing", 15: "Manufacturing", 17: "Manufacturing", 18: "Manufacturing",
        19: "Manufacturing", 20: "Manufacturing", 21: "Manufacturing", 22: "Manufacturing",
        23: "Manufacturing", 24: "Manufacturing", 25: "Manufacturing", 37: "Manufacturing",
        38: "Manufacturing", 39: "Manufacturing",
        26: "Energy", 27: "Energy", 28: "Energy", 29: "Energy", 30: "Energy",
        32: "Technology", 34: "Technology", 35: "Technology", 36: "Technology", 48: "Technology",
        31: "Utilities",
        40: "Transportation",
        41: "Retail/Wholesale", 42: "Retail/Wholesale", 47: "Retail/Wholesale",
        6: "Services", 7: "Services", 8: "Services", 33: "Services", 43: "Services",
        44: "Finance", 45: "Finance", 46: "Finance"
    }
    return mapping.get(ff48_id, "Other")

def load_latest_file(pattern):
    """Load the most recent file matching the pattern."""
    files = sorted(glob.glob(pattern))
    if not files:
        logger.warning(f"No files found for pattern: {pattern}")
        return None
    logger.info(f"Loading latest file: {files[-1]}")
    return files[-1]

def load_data(directory="/Users/carlaamodt/thesis_project/data"):
    """Load all required data files."""
    logger.info("Loading data files...")
    files = {
        "compustat": load_latest_file(f"{directory}/compustat_*.csv"),
        "crsp_ret": load_latest_file(f"{directory}/crsp_ret_*.csv"),
        "crsp_delist": load_latest_file(f"{directory}/crsp_delist_*.csv"),
        "crsp_compustat": load_latest_file(f"{directory}/crsp_compustat_*.csv"),
        "processed": f"{directory}/processed_data.csv",
        "fama_french": f"{directory}/FamaFrench_factors_with_momentum.csv",
        "ff48_mapping": "/Users/carlaamodt/thesis_project/Excel own/Industry classificationFF.xlsx"
    }
    
    missing_files = [k for k, v in files.items() if v is None or not os.path.exists(v)]
    if missing_files:
        logger.error(f"Missing files: {', '.join(missing_files)}")
        raise FileNotFoundError(f"Missing files: {', '.join(missing_files)}")
    
    data = {
        "compustat": pd.read_csv(files["compustat"], parse_dates=['date']),
        "crsp_ret": pd.read_csv(files["crsp_ret"], parse_dates=['date']),
        "crsp_delist": pd.read_csv(files["crsp_delist"], parse_dates=['dlstdt']),
        "crsp_compustat": pd.read_csv(files["crsp_compustat"], parse_dates=['linkdt', 'linkenddt']),
        "processed_path": files["processed"],
        "fama_french": pd.read_csv(files["fama_french"], parse_dates=['date']),
        "ff48_mapping": load_ff48_mapping_from_excel(files["ff48_mapping"])
    }
    logger.info("All data files loaded successfully")
    return data

########################################
### Raw Data Overview ###
########################################

def raw_data_overview(data):
    """Generate overview of raw Compustat and CRSP data."""
    logger.info("Generating raw data overview...")
    compustat = data['compustat']
    
    stats = {
        "Rows": len(compustat),
        "Unique firms (gvkey)": compustat['gvkey'].nunique(),
        "Date range": f"{compustat['date'].min().date()} to {compustat['date'].max().date()}",
        "Firms with goodwill > 0": compustat.loc[compustat['gdwl'] > 0, 'gvkey'].nunique(),
        "Firms with goodwill NaN": compustat.loc[compustat['gdwl'].isna(), 'gvkey'].nunique()
    }
    stats_df = pd.DataFrame.from_dict(stats, orient='index', columns=['Value'])
    output_path = os.path.join(output_dir, get_output_filename("compustat_overview.csv"))
    stats_df.to_csv(output_path)
    logger.info(f"Compustat overview saved to {output_path}")
    
    key_vars = ['gdwl', 'at', 'ceq', 'csho']
    compustat_stats = compustat[key_vars].describe()
    output_path = os.path.join(output_dir, get_output_filename("compustat_key_vars_stats.csv"))
    compustat_stats.to_csv(output_path)
    logger.info(f"Compustat key variables stats saved to {output_path}")
    
    plt.figure(figsize=(10, 6))
    sns.histplot(compustat[compustat['gdwl'] > 0]['gdwl'], bins=50, kde=True, log_scale=True)
    plt.title("Distribution of Goodwill (Log Scale)")
    plt.xlabel("Goodwill (Log Scale)")
    plt.ylabel("Frequency")
    output_path = os.path.join(output_dir, get_output_filename("goodwill_distribution.png"))
    plt.savefig(output_path, bbox_inches='tight')
    plt.close()
    logger.info(f"Goodwill distribution plot saved to {output_path}")
    
    logger.info("Calculating cumulative impairments and goodwill balance...")
    compustat_clean = compustat.dropna(subset=['gdwl']).copy()
    compustat_clean['year'] = compustat_clean['date'].dt.year

    agg_df = (
        compustat_clean.groupby('year')
        .agg(total_goodwill=('gdwl', 'sum'), total_impairment=('gdwlip', 'sum'))
        .fillna(0)
        .sort_index()
        .reset_index()
    )

    agg_df['delta_goodwill'] = agg_df['total_goodwill'].diff()
    agg_df['cumulative_impairments'] = agg_df['total_impairment'].cumsum()

    excel_path = os.path.join(output_dir, get_output_filename("goodwill_impairments_and_balance.xlsx"))
    with pd.ExcelWriter(excel_path) as writer:
        agg_df.to_excel(writer, index=False, sheet_name='Goodwill_Impairments')
    logger.info(f"Goodwill impairments and balance saved to {excel_path}")

    plt.figure(figsize=(12, 6))
    sns.barplot(data=agg_df, x='year', y='total_impairment', color='lightblue', label='Annual Impairments')
    plt.plot(agg_df['year'], agg_df['cumulative_impairments'], label='Cumulative Impairments', linewidth=2)
    plt.title('Goodwill Impairments Over Time')
    plt.xlabel('Year')
    plt.ylabel('Impairment Amount')
    plt.legend()
    plt.tight_layout()
    output_path = os.path.join(output_dir, get_output_filename("impairments_over_time.png"))
    plt.savefig(output_path, bbox_inches='tight')
    plt.close()
    logger.info(f"Impairments over time plot saved to {output_path}")

    fig, ax1 = plt.subplots(figsize=(12, 6))
    sns.lineplot(data=agg_df, x='year', y='total_goodwill', ax=ax1, label='Total Goodwill', linewidth=2, color='navy')
    ax1.set_ylabel('Total Goodwill')
    ax1.set_xlabel('Year')
    ax1.tick_params(axis='y')
    ax2 = ax1.twinx()
    sns.barplot(data=agg_df, x='year', y='total_impairment', ax=ax2, color='lightcoral', alpha=0.5, label='Annual Impairments')
    ax2.set_ylabel('Annual Impairments')
    fig.suptitle('Total Goodwill vs Annual Impairments')
    fig.legend(loc='upper right')
    fig.tight_layout()
    output_path = os.path.join(output_dir, get_output_filename("goodwill_vs_impairments.png"))
    plt.savefig(output_path, bbox_inches='tight')
    plt.close()
    logger.info(f"Goodwill vs impairments plot saved to {output_path}")

    agg_df['total_impairment'] = pd.to_numeric(agg_df['total_impairment'], errors='coerce')
    peak_row = agg_df.loc[agg_df['total_impairment'].abs().idxmax()]
    logger.info(f"Peak impairment year: {peak_row['year']} with impairment of {peak_row['total_impairment']:,.2f}")
    highest_gw_row = agg_df.loc[agg_df['total_goodwill'].idxmax()]
    logger.info(f"Highest goodwill year: {highest_gw_row['year']} with goodwill of {highest_gw_row['total_goodwill']:,.2f}")
    net_shrink_years = agg_df.loc[agg_df['delta_goodwill'] < 0, 'year'].tolist()
    logger.info(f"Years with net goodwill shrinkage: {net_shrink_years}")

    goodwill_impairment_analysis(compustat, data['ff48_mapping'])

def goodwill_impairment_analysis(compustat, ff48_mapping):
    """Analyze goodwill impairment data (gdwlip)."""
    logger.info("Analyzing goodwill impairment data...")
    impairment_df = compustat.dropna(subset=['gdwlip']).copy()
    impairment_df = impairment_df[impairment_df['gdwlip'] > 0]

    if impairment_df.empty:
        logger.warning("No goodwill impairments found in the dataset")
        return

    impairment_df['ff48'] = impairment_df['sich'].apply(lambda x: map_sic_to_ff48(x, ff48_mapping))
    impairment_df['broad_category'] = impairment_df['ff48'].map(map_ff48_to_10_industries)

    max_rows = impairment_df.loc[impairment_df.groupby('gvkey')['gdwlip'].idxmax()].copy()
    max_rows['gdwl'] = compustat.set_index(['gvkey', 'date']).loc[list(zip(max_rows['gvkey'], max_rows['date'])), 'gdwl'].values
    max_rows['ceq'] = compustat.set_index(['gvkey', 'date']).loc[list(zip(max_rows['gvkey'], max_rows['date'])), 'ceq'].values
    max_rows['Impairment_to_Goodwill_%'] = (max_rows['gdwlip'] / max_rows['gdwl'].replace(0, np.nan)) * 100
    max_rows['Impairment_to_Equity_%'] = (max_rows['gdwlip'] / max_rows['ceq'].replace(0, np.nan)) * 100

    top10_impairments = (
        max_rows[['gvkey', 'broad_category', 'gdwlip', 'gdwl', 'ceq', 'Impairment_to_Goodwill_%', 'Impairment_to_Equity_%']]
        .rename(columns={'gdwlip': 'Max_Impairment', 'gdwl': 'Goodwill', 'ceq': 'Common_Equity'})
        .sort_values(by='Max_Impairment', ascending=False)
        .head(10)
    )

    insights = {
        'Total number of impairment events': len(impairment_df),
        'Unique firms with impairments': impairment_df['gvkey'].nunique(),
        'Median impairment size': impairment_df['gdwlip'].median(),
        'Average impairment size': impairment_df['gdwlip'].mean(),
        'Maximum impairment size': impairment_df['gdwlip'].max(),
    }

    output_path = os.path.join(output_dir, get_output_filename("top10_impairments.xlsx"))
    with pd.ExcelWriter(output_path) as writer:
        top10_impairments.to_excel(writer, index=False, sheet_name='Top10_Impairments')
        industry_impairments = (
            impairment_df.groupby('broad_category')['gdwlip']
            .agg(['count', 'sum', 'median', 'mean', 'max'])
            .reset_index()
            .sort_values(by='sum', ascending=False)
        )
        industry_impairments.to_excel(writer, index=False, sheet_name='Industry_Breakdown')
        pd.DataFrame(list(insights.items()), columns=['Metric', 'Value']).to_excel(writer, index=False, sheet_name='Insights')
    logger.info(f"Top 10 impairments and insights saved to {output_path}")

    logger.info("Goodwill Impairment Insights:")
    for k, v in insights.items():
        logger.info(f"- {k}: {v:,.2f}")

    impairment_df['year'] = pd.to_datetime(impairment_df['date']).dt.year
    yearly_impairments = impairment_df.groupby('year')['gdwlip'].sum().reset_index()
    plt.figure(figsize=(12, 6))
    sns.lineplot(data=yearly_impairments, x='year', y='gdwlip', marker='o')
    plt.title("Total Goodwill Impairments Over Time")
    plt.xlabel("Year")
    plt.ylabel("Total Impairments")
    plt.tight_layout()
    output_path = os.path.join(output_dir, get_output_filename("yearly_impairments_over_time.png"))
    plt.savefig(output_path, bbox_inches='tight')
    plt.close()
    logger.info(f"Yearly impairments over time plot saved to {output_path}")

########################################
### Processed Data Exploration ###
########################################

def covid19_shock_analysis(processed_path, chunk_size=500_000):
    """Analyze max drawdowns and recoveries during COVID-19 (March 2020 to May 2023)."""
    logger.info("Starting COVID-19 impact analysis...")
    covid_start = pd.Timestamp("2020-03-01")
    covid_end = pd.Timestamp("2023-05-31")
    dtypes = {'permno': 'int', 'ret': 'float'}
    covid_returns = []

    for chunk in pd.read_csv(processed_path, chunksize=chunk_size, parse_dates=['crsp_date'], dtype=dtypes, low_memory=False):
        chunk = chunk[(chunk['crsp_date'] >= covid_start) & (chunk['crsp_date'] <= covid_end)]
        if not chunk.empty:
            covid_returns.append(chunk[['permno', 'crsp_date', 'ret']])
    
    if not covid_returns:
        logger.warning("No data found in COVID-19 period")
        return

    df_covid = pd.concat(covid_returns)
    df_covid.sort_values(['permno', 'crsp_date'], inplace=True)
    df_covid['cum_return'] = df_covid.groupby('permno')['ret'].transform(lambda x: (1 + x).cumprod())
    
    summary = df_covid.groupby('permno').agg(
        min_cum_return=('cum_return', 'min'),
        max_cum_return=('cum_return', 'max'),
        start_cum_return=('cum_return', 'first'),
        end_cum_return=('cum_return', 'last')
    ).reset_index()
    
    summary['max_loss_pct'] = (summary['min_cum_return'] - summary['start_cum_return']) / summary['start_cum_return'] * 100
    summary['max_gain_pct'] = (summary['max_cum_return'] - summary['min_cum_return']) / summary['min_cum_return'] * 100

    logger.info(f"COVID-19 analysis on {len(summary)} stocks:")
    logger.info(f"Average max loss: {summary['max_loss_pct'].mean():.2f}%")
    logger.info(f"Average max gain: {summary['max_gain_pct'].mean():.2f}%")
    logger.info(f"Median max loss: {summary['max_loss_pct'].median():.2f}%")
    logger.info(f"Median max gain: {summary['max_gain_pct'].median():.2f}%")
    unrecovered = (summary['end_cum_return'] < summary['start_cum_return']).sum()
    logger.info(f"Stocks below pre-COVID level by May 2023: {unrecovered} / {len(summary)}")

def financial_crisis_shock_analysis(processed_path, chunk_size=500_000):
    """Analyze max drawdowns and recoveries during Financial Crisis (Oct 2007 to Dec 2012)."""
    logger.info("Starting Financial Crisis impact analysis...")
    gfc_start = pd.Timestamp("2007-10-01")
    gfc_end = pd.Timestamp("2012-12-31")
    dtypes = {'permno': 'int', 'ret': 'float'}
    gfc_returns = []

    for chunk in pd.read_csv(processed_path, chunksize=chunk_size, parse_dates=['crsp_date'], dtype=dtypes, low_memory=False):
        chunk = chunk[(chunk['crsp_date'] >= gfc_start) & (chunk['crsp_date'] <= gfc_end)]
        if not chunk.empty:
            gfc_returns.append(chunk[['permno', 'crsp_date', 'ret']])
    
    if not gfc_returns:
        logger.warning("No data found in Financial Crisis period")
        return

    df_gfc = pd.concat(gfc_returns)
    df_gfc.sort_values(['permno', 'crsp_date'], inplace=True)
    df_gfc['cum_return'] = df_gfc.groupby('permno')['ret'].transform(lambda x: (1 + x).cumprod())
    
    summary = df_gfc.groupby('permno').agg(
        min_cum_return=('cum_return', 'min'),
        max_cum_return=('cum_return', 'max'),
        start_cum_return=('cum_return', 'first'),
        end_cum_return=('cum_return', 'last')
    ).reset_index()
    
    summary['max_loss_pct'] = (summary['min_cum_return'] - summary['start_cum_return']) / summary['start_cum_return'] * 100
    summary['max_gain_pct'] = (summary['max_cum_return'] - summary['min_cum_return']) / summary['min_cum_return'] * 100

    logger.info(f"Financial Crisis analysis on {len(summary)} stocks:")
    logger.info(f"Average max loss: {summary['max_loss_pct'].mean():.2f}%")
    logger.info(f"Average max gain: {summary['max_gain_pct'].mean():.2f}%")
    logger.info(f"Median max loss: {summary['max_loss_pct'].median():.2f}%")
    logger.info(f"Median max gain: {summary['max_gain_pct'].median():.2f}%")
    unrecovered = (summary['end_cum_return'] < summary['start_cum_return']).sum()
    logger.info(f"Stocks below pre-crisis level by Dec 2012: {unrecovered} / {len(summary)}")

def processed_data_exploration(processed_path, ff48_mapping, chunk_size=500_000):
    """Explore processed data with industry distribution and key metrics."""
    logger.info("Exploring processed data...")
    
    dtypes = {'naicsh': 'str', 'sich': 'str', 'dlstcd': 'str', 'linktype': 'str'}
    total_rows, unique_gvkeys, unique_permnos, years = 0, set(), set(), set()

    for chunk in pd.read_csv(processed_path, chunksize=chunk_size, parse_dates=['crsp_date'], dtype=dtypes, low_memory=False):
        total_rows += len(chunk)
        unique_gvkeys.update(chunk['gvkey'].unique())
        unique_permnos.update(chunk['permno'].unique())
        years.update(chunk['crsp_date'].dt.year.unique())
    
    num_firms = len(unique_gvkeys)
    num_stocks = len(unique_permnos)
    logger.info(f"Processed data: {num_firms} firms, {num_stocks} stocks, {total_rows} rows")

    stats = {
        "Rows": total_rows,
        "Unique firms (gvkey)": num_firms,
        "Unique stocks (permno)": num_stocks,
        "Years covered": sorted(years)
    }
    stats_df = pd.DataFrame.from_dict(stats, orient='index', columns=['Value'])
    output_path = os.path.join(output_dir, get_output_filename("processed_data_stats.csv"))
    stats_df.to_csv(output_path)
    logger.info(f"Processed data stats saved to {output_path}")
    
    industry_data = plot_industry_distributions(processed_path, ff48_mapping, chunk_size, num_firms)
    
    coverage = []
    for chunk in pd.read_csv(processed_path, chunksize=chunk_size, parse_dates=['crsp_date'], dtype=dtypes, low_memory=False):
        chunk_coverage = chunk.groupby(chunk['crsp_date'].dt.year).agg(firms=('gvkey', 'nunique'))
        coverage.append(chunk_coverage)
    coverage_df = pd.concat(coverage).groupby(level=0).sum()
    coverage_df = coverage_df[coverage_df.index >= 2003]
    output_path = os.path.join(output_dir, get_output_filename("firms_by_year.csv"))
    coverage_df.to_csv(output_path)
    
    plt.figure(figsize=(10, 6))
    sns.lineplot(data=coverage_df['firms'], label="Unique Firms")
    plt.title("Number of Firms Over Time")
    plt.xlabel("Year")
    plt.ylabel("Number of Firms")
    plt.xticks(ticks=coverage_df.index, labels=coverage_df.index.astype(int), rotation=45)
    output_path = os.path.join(output_dir, get_output_filename("firms_over_time.png"))
    plt.savefig(output_path, bbox_inches='tight')
    plt.close()
    logger.info(f"Firms over time plot saved to {output_path}")
    
    plot_goodwill_trends(processed_path, ff48_mapping, industry_data['industry_df_10'], chunk_size)

    logger.info("Calculating median market cap for NYSE and All Exchanges...")

    market_cap_all = []
    for chunk in pd.read_csv(
        processed_path, chunksize=chunk_size, parse_dates=['crsp_date'],
        dtype={'exchcd': 'float', 'market_cap': 'float'}, low_memory=False
    ):
        chunk = chunk.dropna(subset=['exchcd', 'market_cap'])
        chunk = chunk[chunk['exchcd'].isin([1, 2, 3])]
        chunk['year'] = chunk['crsp_date'].dt.year
        market_cap_all.append(chunk[['year', 'exchcd', 'market_cap']])

    if market_cap_all:
        df_mc = pd.concat(market_cap_all)
        exchange_map = {1: 'NYSE', 2: 'AMEX', 3: 'NASDAQ'}
        df_mc['Exchange'] = df_mc['exchcd'].map(exchange_map)
        df_mc = df_mc[df_mc['year'] >= 2004]

        # Compute medians
        nyse_df = df_mc[df_mc['Exchange'] == 'NYSE']
        nyse_median = nyse_df.groupby('year')['market_cap'].median().reset_index()
        nyse_median['Exchange'] = 'NYSE'

        all_median = df_mc.groupby('year')['market_cap'].median().reset_index()
        all_median['Exchange'] = 'All'

        plot_df = pd.concat([nyse_median, all_median], ignore_index=True)
        plot_df['Market_Cap_Million'] = plot_df['market_cap'] / 1e6

        plt.figure(figsize=(12, 6))
        color_map = {'NYSE': '#08306b', 'All': '#6baed6'}

        for label, group in plot_df.groupby('Exchange'):
            years = group['year'].values
            mcaps = group['Market_Cap_Million'].values
            if len(years) >= 3:
                xnew = np.linspace(years.min(), years.max(), 300)
                spl = make_interp_spline(years, mcaps, k=3)
                y_smooth = spl(xnew)
                plt.plot(xnew, y_smooth, label=label, color=color_map[label], linewidth=2.5)

        ax = plt.gca()
        ax.spines[['top', 'right']].set_visible(False)
        ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f'{x:.0f}'))

        xticks = sorted(plot_df['year'].unique())
        ax.set_xticks(xticks[::2])  # every second year
        ax.set_xticklabels([str(y) for y in xticks[::2]])

        plt.ylabel("Median Market Cap (Millions)")
        plt.legend(title=None)
        plt.tight_layout()

        output_path = os.path.join(output_dir, get_output_filename("market_cap_median_trend_smooth_nyse_all.png"))
        plt.savefig(output_path, bbox_inches='tight')
        plt.close()
        logger.info(f"Median market cap trend plot saved to {output_path}")
    else:
        logger.warning("No data found for exchanges 1, 2, 3 when calculating market cap.")

def plot_industry_distributions(processed_path, ff48_mapping, chunk_size=500_000, total_firms=None):
    """Create visualizations for FF48 and 10-industry distributions."""
    logger.info("Generating industry distribution visualizations...")
    
    firm_industries = {}
    unique_firms_with_sic = set()
    dtypes = {'sich': 'str'}
    for chunk in pd.read_csv(processed_path, chunksize=chunk_size, dtype=dtypes, low_memory=False):
        chunk_firms = chunk[['gvkey', 'sich']].drop_duplicates()
        chunk_with_sic = chunk_firms.dropna(subset=['sich']).copy()
        chunk_without_sic = chunk_firms[chunk_firms['sich'].isna()]
        unique_firms_with_sic.update(chunk_with_sic['gvkey'].unique())
        chunk_with_sic['ff48'] = chunk_with_sic['sich'].apply(lambda x: map_sic_to_ff48(x, ff48_mapping))
        industry_counts = chunk_with_sic.groupby('ff48')['gvkey'].nunique()
        for ff48, count in industry_counts.items():
            firm_industries[ff48] = firm_industries.get(ff48, 0) + count
    
    firms_without_sic = total_firms - len(unique_firms_with_sic)
    logger.info(f"Firms with SIC: {len(unique_firms_with_sic)}, Firms without SIC: {firms_without_sic}")
    
    all_industries = {i: 0 for i in range(1, 49)}
    all_industries.update(firm_industries)
    all_industries[48] = all_industries.get(48, 0) + firms_without_sic
    
    industry_df_48 = pd.DataFrame.from_dict(all_industries, orient='index', columns=['num_firms']).sort_index()
    ff48_excel = pd.read_excel("/Users/carlaamodt/thesis_project/Excel own/Industry classificationFF.xlsx")
    ff48_excel['Industry number'] = pd.to_numeric(ff48_excel['Industry number'], errors='coerce').fillna(48).astype(int)
    ff48_names = dict(zip(ff48_excel['Industry number'], ff48_excel['Industry description'].fillna('Other')))
    industry_df_48['industry_name'] = industry_df_48.index.map(ff48_names).fillna('Other')
    industry_df_48['percentage'] = (industry_df_48['num_firms'] / total_firms * 100).round(2)
    
    output_path = os.path.join(output_dir, get_output_filename("ff48_industry_distribution.csv"))
    industry_df_48[['industry_name', 'num_firms', 'percentage']].to_csv(output_path)
    logger.info(f"FF48 industry distribution saved to {output_path}")
    
    industry_df_48_pie = industry_df_48[industry_df_48['num_firms'] > 0].copy()
    industry_df_48_pie['percent'] = industry_df_48_pie['num_firms'] / total_firms * 100

    # Separate small industries and create 'Assigned other'
    large_industries = industry_df_48_pie[industry_df_48_pie['percent'] >= 1.5].copy()
    small_industries = industry_df_48_pie[industry_df_48_pie['percent'] < 1.5]
    if not small_industries.empty:
        other_row = pd.DataFrame({
            'industry_name': ['Assigned other'],
            'num_firms': [small_industries['num_firms'].sum()],
            'percent': [small_industries['percent'].sum()]
        })
        large_industries = large_industries.sort_values('num_firms', ascending=True)
        industry_df_48_pie = pd.concat([other_row, large_industries], ignore_index=True)
    else:
        industry_df_48_pie = industry_df_48_pie.sort_values('num_firms', ascending=True)

    # Color palette: light → dark (Assigned other first)
    colors_48 = sns.color_palette("Blues", len(industry_df_48_pie))

    # Plot pie chart
    plt.figure(figsize=(12, 12))
    wedges, texts, autotexts = plt.pie(
        industry_df_48_pie['num_firms'],
        labels=industry_df_48_pie['industry_name'],
        autopct='%1.1f%%',
        startangle=90,
        counterclock=False,
        colors=colors_48,
        textprops={'fontsize': 12, 'color': 'black'},
        labeldistance=1.1,
        pctdistance=0.85
    )

    # White number (autotext) for darkest 2 slices only
    for i, autotext in enumerate(autotexts):
        if i >= len(industry_df_48_pie) - 2:  # last two = darkest
            autotext.set_color('white')
        autotext.set_fontsize(10)

    # Center donut label
    plt.text(0, 0, f'Total Firms\n{total_firms}', fontsize=25, ha='center', va='center', weight='bold')

    # Create donut hole
    centre_circle = plt.Circle((0, 0), 0.70, fc='white')
    plt.gca().add_artist(centre_circle)

    plt.axis('equal')
    plt.title('Fama-French 48 Industry Distribution of Firms', fontsize=14, pad=20)

    # Save
    output_path = os.path.join(output_dir, get_output_filename("ff48_industry_distribution_donut.png"))
    plt.savefig(output_path, bbox_inches='tight')
    plt.close()
    logger.info(f"FF48 industry distribution donut saved to {output_path}")
    
    top_10_df = industry_df_48_pie.sort_values('num_firms', ascending=False).head(10)
    plt.figure(figsize=(12, 6))
    blue_palette = sns.color_palette("Blues", len(top_10_df))[::-1]
    sns.barplot(x='num_firms', y='industry_name', data=top_10_df, palette=blue_palette)
    plt.title('Top 10 Fama-French 48 Industries by Firm Count')
    plt.xlabel('Number of Firms')
    plt.ylabel('Industry')
    output_path = os.path.join(output_dir, get_output_filename("ff48_top_10_industries_bar.png"))
    plt.savefig(output_path, bbox_inches='tight')
    plt.close()
    logger.info(f"Top 10 FF48 industries bar chart saved to {output_path}")
    
    industry_df_48['broad_category'] = industry_df_48.index.map(map_ff48_to_10_industries)
    industry_df_10 = industry_df_48.groupby('broad_category').agg({'num_firms': 'sum'}).reset_index()
    industry_df_10['percentage'] = (industry_df_10['num_firms'] / total_firms * 100).round(2)
    industry_df_10 = industry_df_10.sort_values('num_firms', ascending=False)
    
    colors_10_donut = sns.color_palette("Blues", len(industry_df_10))[::-1]
    plt.figure(figsize=(10, 10))
    wedges, _, autotexts = plt.pie(
        industry_df_10['num_firms'], labels=None, autopct='%1.1f%%', startangle=90,
        colors=colors_10_donut, textprops={'fontsize': 12}, labeldistance=1.05, pctdistance=0.85
    )
    for autotext in autotexts:
        autotext.set_fontsize(10)
    plt.text(0, 0, f'Total Firms\n{total_firms}', fontsize=14, ha='center', va='center', weight='bold')
    centre_circle = plt.Circle((0, 0), 0.52, fc='white')
    fig = plt.gcf()
    fig.gca().add_artist(centre_circle)
    plt.axis('equal')
    plt.title('Combined Industry Distribution of Firms', fontsize=14, pad=20)
    plt.legend(wedges, industry_df_10['broad_category'], title="Industries",
               loc="lower center", bbox_to_anchor=(0.5, -0.1), ncol=3, fontsize=10, title_fontsize=12)
    output_path = os.path.join(output_dir, get_output_filename("10_industry_distribution_donut.png"))
    plt.savefig(output_path, bbox_inches='tight')
    plt.close()
    logger.info(f"10-industry distribution donut saved to {output_path}")

    output_path = os.path.join(output_dir, get_output_filename("10_industry_distribution.csv"))
    industry_df_10[['broad_category', 'num_firms', 'percentage']].to_csv(output_path)
    logger.info(f"10-industry distribution saved to {output_path}")

    return {'industry_df_48': industry_df_48, 'industry_df_10': industry_df_10}

def plot_goodwill_trends(processed_path, ff48_mapping, industry_df_10, chunk_size=500_000):
    """Plot goodwill trends over time by major industry."""
    logger.info("Generating goodwill trends by major industry...")
    
    dtypes = {'sich': 'str', 'gdwl': 'float'}
    goodwill_data = []
    for chunk in pd.read_csv(processed_path, chunksize=chunk_size, parse_dates=['crsp_date'], dtype=dtypes, low_memory=False):
        chunk = chunk.dropna(subset=['sich', 'gdwl', 'crsp_date'])
        chunk['ff48'] = chunk['sich'].apply(lambda x: map_sic_to_ff48(x, ff48_mapping))
        chunk['broad_category'] = chunk['ff48'].map(map_ff48_to_10_industries)
        chunk['year'] = chunk['crsp_date'].dt.year.astype(int)
        goodwill_data.append(chunk[['year', 'broad_category', 'gdwl']])
    
    goodwill_df = pd.concat(goodwill_data)
    goodwill_trends = goodwill_df.groupby(['year', 'broad_category'])['gdwl'].mean().reset_index()
    goodwill_trends = goodwill_trends[goodwill_trends['year'] >= 2003]
    goodwill_pivot = goodwill_trends.pivot(index='year', columns='broad_category', values='gdwl').fillna(0)
    
    custom_colors = [
        '#1b263b', '#2a623d', '#3c2f2f', '#3d5a80', '#4a7043',
        '#5c6f7b', '#829440', '#606c88', '#778beb', '#93a8ac'
    ]
    
    plt.figure(figsize=(12, 8))
    goodwill_pivot.plot(kind='bar', stacked=True, color=custom_colors, ax=plt.gca())
    plt.title('Average Goodwill Over Time by Major Industry')
    plt.xlabel('Year')
    plt.ylabel('Average Goodwill (Millions)')
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()
    output_path = os.path.join(output_dir, get_output_filename("goodwill_trends_by_industry_stacked.png"))
    plt.savefig(output_path, bbox_inches='tight')
    plt.close()
    logger.info(f"Goodwill trends by industry saved to {output_path}")

    plt.figure(figsize=(12, 8))
    sns.barplot(data=goodwill_trends, x='year', y='gdwl', hue='broad_category', palette=colors_10)
    plt.title('Average Goodwill Over Time by Major Industry (Grouped)')
    plt.xlabel('Year')
    plt.ylabel('Average Goodwill (Millions)')
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.xticks(rotation=45)
    plt.tight_layout()
    output_path = os.path.join(output_dir, get_output_filename("goodwill_trends_by_industry_grouped.png"))
    plt.savefig(output_path, bbox_inches='tight')
    plt.close()
    logger.info(f"Grouped goodwill trends by industry saved to {output_path}")

########################################
### Fama-French Factor Analysis ###
########################################

def fama_french_analysis(fama_french):
    """Analyze Fama-French factors."""
    logger.info("Analyzing Fama-French factors...")
    
    stats = {
        "Rows": len(fama_french),
        "Date range": f"{fama_french['date'].min().date()} to {fama_french['date'].max().date()}"
    }
    stats_df = pd.DataFrame.from_dict(stats, orient='index', columns=['Value'])
    output_path = os.path.join(output_dir, get_output_filename("fama_french_overview.csv"))
    stats_df.to_csv(output_path)
    logger.info(f"Fama-French overview saved to {output_path}")
    
    factor_cols = ['mkt_rf', 'smb', 'hml', 'rmw', 'cma', 'mom', 'rf']
    ff_stats = fama_french[factor_cols].describe()
    output_path = os.path.join(output_dir, get_output_filename("fama_french_stats.csv"))
    ff_stats.to_csv(output_path)
    logger.info(f"Fama-French stats saved to {output_path}")
    
    plt.figure(figsize=(10, 8))
    sns.heatmap(fama_french[factor_cols].corr(), annot=True, cmap='coolwarm', vmin=-1, vmax=1)
    plt.title("Correlation of Fama-French Factors")
    output_path = os.path.join(output_dir, get_output_filename("fama_french_correlation_heatmap.png"))
    plt.savefig(output_path, bbox_inches='tight')
    plt.close()
    logger.info(f"Fama-French correlation heatmap saved to {output_path}")

########################################
### Main Execution ###
########################################

def main():
    """Main function to run data exploration."""
    logger.info("Starting data exploration...")
    try:
        data = load_data()
        raw_data_overview(data)
        covid19_shock_analysis(data['processed_path'])
        financial_crisis_shock_analysis(data['processed_path'])
        processed_data_exploration(data['processed_path'], data['ff48_mapping'])
        fama_french_analysis(data['fama_french'])
        logger.info("Data exploration completed successfully")
    except Exception as e:
        logger.error(f"Data exploration failed: {e}")
        raise

if __name__ == "__main__":
    main()