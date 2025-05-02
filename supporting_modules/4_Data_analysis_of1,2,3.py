import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
import glob
import logging
from scipy.interpolate import make_interp_spline

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Set output directory
output_dir = "/Users/carlaamodt/thesis_project/File_4_analysis"
os.makedirs(output_dir, exist_ok=True)

# Set seaborn style (no gridlines)
sns.set(style="white")

# Counter for numbering outputs
output_counter = 1
def get_output_filename(prefix):
    global output_counter
    filename = f"file4_{output_counter:03d}_{prefix}"
    output_counter += 1
    return filename

# Define a consistent color palette for the 10 industries (used in goodwill trends)
colors_10 = sns.color_palette("tab10", 10)

########################################
### Load Raw and Processed Data ###
########################################

def load_ff48_mapping_from_excel(filepath: str) -> dict:
    """Load Fama-French 48 industry classification mapping from an Excel file."""
    logger.info(f"Loading FF48 industry classification from {filepath}")
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"Excel file not found at {filepath}")
    
    df = pd.read_excel(filepath)
    if not all(col in df.columns for col in ['Industry number', 'Industry code', 'Industry description']):
        raise ValueError("Excel file must contain columns: 'Industry number', 'Industry code', 'Industry description'")
    
    df['Industry number'] = df['Industry number'].ffill()
    ff48_mapping = {}
    for _, row in df.iterrows():
        code = row['Industry code']
        industry_number = row['Industry number']
        if isinstance(code, str) and '-' in code:
            try:
                start, end = map(int, code.strip().split('-'))
                ff48_mapping[range(start, end + 1)] = int(industry_number) if pd.notna(industry_number) else 48
            except ValueError:
                logger.warning(f"Skipping invalid SIC range: {code}")
    logger.info(f"Loaded {len(ff48_mapping)} SIC ranges into FF48 mapping.")
    return ff48_mapping

def map_sic_to_ff48(sic, ff48_mapping):
    """Map a 4-digit SIC code to Fama-French 48 industry."""
    try:
        if pd.isna(sic):
            return 48  # Assign to "Other" if SIC is missing
        sic_str = str(sic).split('.')[0]  # Remove decimal part
        sic_int = int(sic_str) if sic_str.isdigit() else 0
        for sic_range, industry in ff48_mapping.items():
            if sic_int in sic_range:
                return industry
        return 48  # Default to "Other"
    except (ValueError, TypeError) as e:
        logger.warning(f"Invalid SIC code '{sic}': {e}. Defaulting to 'Other'.")
        return 48

def map_ff48_to_10_industries(ff48_id):
    """Map Fama-French 48 industries to 10 broader categories."""
    mapping = {
        # Consumer Goods
        1: "Consumer Goods",  # Agriculture
        2: "Consumer Goods",  # Food Products
        3: "Consumer Goods",  # Candy & Soda
        4: "Consumer Goods",  # Beer & Liquor
        5: "Consumer Goods",  # Tobacco Products
        9: "Consumer Goods",  # Consumer Goods
        10: "Consumer Goods",  # Apparel
        16: "Consumer Goods",  # Textiles
        # Healthcare
        11: "Healthcare",  # Healthcare
        12: "Healthcare",  # Medical Equipment
        13: "Healthcare",  # Pharmaceutical Products
        # Manufacturing
        14: "Manufacturing",  # Chemicals
        15: "Manufacturing",  # Rubber and Plastic Products
        17: "Manufacturing",  # Construction Materials
        18: "Manufacturing",  # Construction
        19: "Manufacturing",  # Steel Works Etc
        20: "Manufacturing",  # Fabricated Products
        21: "Manufacturing",  # Machinery
        22: "Manufacturing",  # Electrical Equipment
        23: "Manufacturing",  # Automobiles and Trucks
        24: "Manufacturing",  # Aircraft
        25: "Manufacturing",  # Shipbuilding, Railroad Equipment
        37: "Manufacturing",  # Measuring and Control Equipment
        38: "Manufacturing",  # Business Supplies
        39: "Manufacturing",  # Shipping Containers
        # Energy
        26: "Energy",  # Defense
        27: "Energy",  # Precious Metals
        28: "Energy",  # Non-Metallic and Industrial Metal Mining
        29: "Energy",  # Coal
        30: "Energy",  # Petroleum and Natural Gas
        # Technology
        32: "Technology",  # Communication
        34: "Technology",  # Business Services
        35: "Technology",  # Computers
        36: "Technology",  # Electronic Equipment
        48: "Technology",  # Cogeneration - SM power producer (also "Other")
        # Utilities
        31: "Utilities",  # Utilities
        # Transportation
        40: "Transportation",  # Transportation
        # Retail/Wholesale
        41: "Retail/Wholesale",  # Wholesale
        42: "Retail/Wholesale",  # Retail
        47: "Retail/Wholesale",  # Trading
        # Services
        6: "Services",  # Recreation
        7: "Services",  # Entertainment
        8: "Services",  # Printing and Publishing
        33: "Services",  # Personal Services
        43: "Services",  # Restaurants, Hotels, Motels
        # Finance
        44: "Finance",  # Banking
        45: "Finance",  # Insurance
        46: "Finance",  # Real Estate
    }
    return mapping.get(ff48_id, "Other")

def load_data(directory="/Users/carlaamodt/thesis_project/data"):
    """Load all required data files."""
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
    
    logger.info("Loading data...")
    data = {
        "compustat": pd.read_csv(files["compustat"], parse_dates=['date']),
        "crsp_ret": pd.read_csv(files["crsp_ret"], parse_dates=['date']),
        "crsp_delist": pd.read_csv(files["crsp_delist"], parse_dates=['dlstdt']),
        "crsp_compustat": pd.read_csv(files["crsp_compustat"], parse_dates=['linkdt', 'linkenddt']),
        "processed_path": files["processed"],
        "fama_french": pd.read_csv(files["fama_french"], parse_dates=['date']),
        "ff48_mapping": load_ff48_mapping_from_excel(files["ff48_mapping"])
    }
    return data

def load_latest_file(pattern):
    """Load the most recent file matching the pattern."""
    files = sorted(glob.glob(pattern))
    return files[-1] if files else None

########################################
### Raw Data Overview ###
########################################

def raw_data_overview(data):
    """Generate overview of raw Compustat and CRSP data."""
    logger.info("Generating raw data overview...")
    compustat = data['compustat']
    
    # Compustat Overview
    stats = {
        "Rows": len(compustat),
        "Unique firms (gvkey)": compustat['gvkey'].nunique(),
        "Date range": f"{compustat['date'].min().date()} to {compustat['date'].max().date()}",
        "Firms with goodwill > 0": compustat.loc[compustat['gdwl'] > 0, 'gvkey'].nunique(),
        "Firms with goodwill NaN": compustat.loc[compustat['gdwl'].isna(), 'gvkey'].nunique()
    }
    stats_df = pd.DataFrame.from_dict(stats, orient='index', columns=['Value'])
    stats_df.to_csv(os.path.join(output_dir, get_output_filename("compustat_overview.csv")))
    logger.info(f"Compustat overview saved as file4_{output_counter-1:03d}")
    
    # Key variables
    key_vars = ['gdwl', 'at', 'ceq', 'csho']
    compustat_stats = compustat[key_vars].describe()
    compustat_stats.to_csv(os.path.join(output_dir, get_output_filename("compustat_key_vars_stats.csv")))
    logger.info(f"Compustat key variables stats saved as file4_{output_counter-1:03d}")
    
    # Goodwill distribution
    plt.figure(figsize=(10, 6))
    sns.histplot(compustat[compustat['gdwl'] > 0]['gdwl'], bins=50, kde=True, log_scale=True)
    plt.title("Distribution of Goodwill (Log Scale)")
    plt.xlabel("Goodwill (Log Scale)")
    plt.ylabel("Frequency")
    plt.savefig(os.path.join(output_dir, get_output_filename("goodwill_distribution.png")), bbox_inches='tight')
    plt.close()
    logger.info(f"Goodwill distribution plot saved as file4_{output_counter-1:03d}")
        # NEW: Accumulated Impairments and Goodwill Balance Over Time
    logger.info("Calculating cumulative impairments and goodwill balance over time...")
    
    compustat_clean = compustat.dropna(subset=['gdwl']).copy()
    compustat_clean['year'] = compustat_clean['date'].dt.year

    agg_df = (
        compustat_clean.groupby('year')
        .agg(
            total_goodwill=('gdwl', 'sum'),
            total_impairment=('gdwlip', 'sum')
        )
        .fillna(0)
        .sort_index()
        .reset_index()
    )

    # Calculate delta and cumulative
    agg_df['delta_goodwill'] = agg_df['total_goodwill'].diff()
    agg_df['cumulative_impairments'] = agg_df['total_impairment'].cumsum()

    # Save Excel
    excel_path = os.path.join(output_dir, get_output_filename("goodwill_impairments_and_balance.xlsx"))
    with pd.ExcelWriter(excel_path) as writer:
        agg_df.to_excel(writer, index=False, sheet_name='Goodwill_Impairments')

    # Plot 1: Annual impairments + cumulative impairments
    plt.figure(figsize=(12, 6))
    sns.barplot(data=agg_df, x='year', y='total_impairment', color='lightblue', label='Annual Impairments')
    plt.plot(agg_df['year'], agg_df['cumulative_impairments'], label='Cumulative Impairments', linewidth=2)
    plt.title('Goodwill Impairments Over Time')
    plt.xlabel('Year')
    plt.ylabel('Impairment Amount')
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, get_output_filename("impairments_over_time.png")), bbox_inches='tight')
    plt.close()

    # Plot 2: Goodwill balance vs annual impairments (dual axis)
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
    plt.savefig(os.path.join(output_dir, get_output_filename("goodwill_vs_impairments.png")), bbox_inches='tight')
    plt.close()

    # Log key stats
    # Ensure numeric type for total_impairment to avoid any dtype issues
    agg_df['total_impairment'] = pd.to_numeric(agg_df['total_impairment'], errors='coerce')

    # Find peak impairment year based on absolute impairment value
    peak_row = agg_df.loc[agg_df['total_impairment'].abs().idxmax()]
    peak_impairment_year = peak_row['year']
    peak_impairment_value = peak_row['total_impairment']

    logger.info(f"Peak impairment year (by total impairment amount): {peak_impairment_year} "
                f"with impairment of {peak_impairment_value:,.2f}")

    # Find year with highest goodwill balance
    highest_gw_row = agg_df.loc[agg_df['total_goodwill'].idxmax()]
    highest_goodwill_year = highest_gw_row['year']
    highest_goodwill_value = highest_gw_row['total_goodwill']

    logger.info(f"Highest goodwill year: {highest_goodwill_year} "
                f"with goodwill of {highest_goodwill_value:,.2f}")

    # Identify years with net goodwill shrinkage
    net_shrink_years = agg_df.loc[agg_df['delta_goodwill'] < 0, 'year'].tolist()
    logger.info(f"Years with net goodwill shrinkage: {net_shrink_years}")
    # NEW: Goodwill Impairment Analysis
    goodwill_impairment_analysis(compustat, data['ff48_mapping'])

def goodwill_impairment_analysis(compustat, ff48_mapping):
    logger.info("Analyzing goodwill impairment data (gdwlip)...")
    impairment_df = compustat.dropna(subset=['gdwlip']).copy()
    impairment_df = impairment_df[impairment_df['gdwlip'] > 0]

    if impairment_df.empty:
        logger.warning("No goodwill impairments found in the dataset.")
        return

    impairment_df['ff48'] = impairment_df['sich'].apply(lambda x: map_sic_to_ff48(x, ff48_mapping))
    impairment_df['broad_category'] = impairment_df['ff48'].map(map_ff48_to_10_industries)

    # Get the row with max impairment per firm
    max_rows = impairment_df.loc[
        impairment_df.groupby('gvkey')['gdwlip'].idxmax()
    ].copy()

    # Add ratios to goodwill and common equity
    max_rows['gdwl'] = compustat.set_index(['gvkey', 'date']).loc[
        list(zip(max_rows['gvkey'], max_rows['date'])), 'gdwl'
    ].values
    max_rows['ceq'] = compustat.set_index(['gvkey', 'date']).loc[
        list(zip(max_rows['gvkey'], max_rows['date'])), 'ceq'
    ].values

    # Avoid division by zero
    max_rows['Impairment_to_Goodwill_%'] = (max_rows['gdwlip'] / max_rows['gdwl'].replace(0, np.nan)) * 100
    max_rows['Impairment_to_Equity_%'] = (max_rows['gdwlip'] / max_rows['ceq'].replace(0, np.nan)) * 100

    top10_impairments = (
        max_rows[['gvkey', 'broad_category', 'gdwlip', 'gdwl', 'ceq', 'Impairment_to_Goodwill_%', 'Impairment_to_Equity_%']]
        .rename(columns={
            'gdwlip': 'Max_Impairment',
            'gdwl': 'Goodwill',
            'ceq': 'Common_Equity'
        })
        .sort_values(by='Max_Impairment', ascending=False)
        .head(10)
    )

    median_impairment = impairment_df['gdwlip'].median()
    insights = {
        'Total number of impairment events': len(impairment_df),
        'Unique firms with impairments': impairment_df['gvkey'].nunique(),
        'Median impairment size': median_impairment,
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

    # Log key insights
    logger.info("Goodwill Impairment Insights:")
    for k, v in insights.items():
        logger.info(f"- {k}: {v:,.2f}")

    # Plot impairment levels over time
    impairment_df['year'] = pd.to_datetime(impairment_df['date']).dt.year
    yearly_impairments = impairment_df.groupby('year')['gdwlip'].sum().reset_index()

    plt.figure(figsize=(12, 6))
    sns.lineplot(data=yearly_impairments, x='year', y='gdwlip', marker='o')
    plt.title("Total Goodwill Impairments Over Time")
    plt.xlabel("Year")
    plt.ylabel("Total Impairments")
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, get_output_filename("impairments_over_time.png")), bbox_inches='tight')
    plt.close()
    logger.info("Impairments over time plot saved.")


    # Log key insights for quick view
    logger.info("Goodwill Impairment Insights:")
    for k, v in insights.items():
        logger.info(f"- {k}: {v:,.2f}")   
########################################
### Processed Data Exploration ###
########################################
def covid19_shock_analysis(processed_path, chunk_size=500_000):
    """Analyze max drawdowns and recoveries during COVID-19 (March 2020 to May 2023)."""
    logger.info("Starting COVID-19 impact analysis (March 2020 to May 2023)...")
    
    covid_start = pd.Timestamp("2020-03-01")
    covid_end = pd.Timestamp("2023-05-31")

    dtypes = {'permno': 'int', 'ret': 'float'}
    covid_returns = []

    for chunk in pd.read_csv(processed_path, chunksize=chunk_size, parse_dates=['crsp_date'], dtype=dtypes, low_memory=False):
        chunk = chunk[(chunk['crsp_date'] >= covid_start) & (chunk['crsp_date'] <= covid_end)]
        covid_returns.append(chunk[['permno', 'crsp_date', 'ret']])
    
    if not covid_returns:
        logger.warning("No data found in the specified COVID-19 period.")
        return

    df_covid = pd.concat(covid_returns)
    
    # Calculate cumulative return per stock
    df_covid.sort_values(['permno', 'crsp_date'], inplace=True)
    df_covid['cum_return'] = df_covid.groupby('permno')['ret'].transform(lambda x: (1 + x).cumprod())    
    # Get min and max cumulative returns for each stock
    summary = df_covid.groupby('permno').agg(
        min_cum_return=('cum_return', 'min'),
        max_cum_return=('cum_return', 'max'),
        start_cum_return=('cum_return', 'first'),
        end_cum_return=('cum_return', 'last')
    ).reset_index()
    
    # Calculate % loss and % gain from peak to trough and recovery
    summary['max_loss_pct'] = (summary['min_cum_return'] - summary['start_cum_return']) / summary['start_cum_return'] * 100
    summary['max_gain_pct'] = (summary['max_cum_return'] - summary['min_cum_return']) / summary['min_cum_return'] * 100

    # Log high-level insights
    logger.info(f"COVID-19 analysis on {len(summary)} stocks:")
    logger.info(f"Average max loss: {summary['max_loss_pct'].mean():.2f}%")
    logger.info(f"Average max gain: {summary['max_gain_pct'].mean():.2f}%")
    logger.info(f"Median max loss: {summary['max_loss_pct'].median():.2f}%")
    logger.info(f"Median max gain: {summary['max_gain_pct'].median():.2f}%")

    # Count how many stocks never recovered to pre-COVID level
    unrecovered = (summary['end_cum_return'] < summary['start_cum_return']).sum()
    logger.info(f"Stocks that ended below pre-COVID level by May 2023: {unrecovered} / {len(summary)}")
    ### FINANCIAL CRISIS SHOCK ANALYSIS ### 
def financial_crisis_shock_analysis(processed_path, chunk_size=500_000):
    """Analyze max drawdowns and recoveries during the Global Financial Crisis (Oct 2007 to Dec 2012)."""
    logger.info("Starting Financial Crisis impact analysis (Oct 2007 to Dec 2012)...")
    
    gfc_start = pd.Timestamp("2007-10-01")
    gfc_end = pd.Timestamp("2012-12-31")

    dtypes = {'permno': 'int', 'ret': 'float'}
    gfc_returns = []

    for chunk in pd.read_csv(processed_path, chunksize=chunk_size, parse_dates=['crsp_date'], dtype=dtypes, low_memory=False):
        chunk = chunk[(chunk['crsp_date'] >= gfc_start) & (chunk['crsp_date'] <= gfc_end)]
        gfc_returns.append(chunk[['permno', 'crsp_date', 'ret']])
    
    if not gfc_returns:
        logger.warning("No data found in the specified Financial Crisis period.")
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
    logger.info(f"Stocks that ended below pre-crisis level by Dec 2012: {unrecovered} / {len(summary)}")

def processed_data_exploration(processed_path, ff48_mapping, chunk_size=500_000):
    """Explore processed data with industry distribution and key metrics."""
    logger.info("Exploring processed data...")
    
    dtypes = {'naicsh': 'str', 'sich': 'str', 'dlstcd': 'str', 'linktype': 'str'}
    
    # Basic stats
    total_rows, unique_gvkeys, unique_permnos, years = 0, set(), set(), set()
    for chunk in pd.read_csv(processed_path, chunksize=chunk_size, parse_dates=['crsp_date'], dtype=dtypes, low_memory=False):
        total_rows += len(chunk)
        unique_gvkeys.update(chunk['gvkey'].unique())
        unique_permnos.update(chunk['permno'].unique())
        years.update(chunk['crsp_date'].dt.year.unique())
    
    num_firms = len(unique_gvkeys)
    num_stocks = len(unique_permnos)
    print(f"Number of Firms (unique gvkey): {num_firms}")
    print(f"Number of Stocks (unique permno): {num_stocks}")
    
    stats = {
        "Rows": total_rows,
        "Unique firms (gvkey)": num_firms,
        "Unique stocks (permno)": num_stocks,
        "Years covered": sorted(years)
    }
    stats_df = pd.DataFrame.from_dict(stats, orient='index', columns=['Value'])
    stats_df.to_csv(os.path.join(output_dir, get_output_filename("processed_data_stats.csv")))
    logger.info(f"Processed data stats saved as file4_{output_counter-1:03d}")
    
    # Industry distribution (FF48 and 10 categories)
    industry_data = plot_industry_distributions(processed_path, ff48_mapping, chunk_size, num_firms)
    
    # Firm count over time
    coverage = []
    for chunk in pd.read_csv(processed_path, chunksize=chunk_size, parse_dates=['crsp_date'], dtype=dtypes, low_memory=False):
        chunk_coverage = chunk.groupby(chunk['crsp_date'].dt.year).agg(firms=('gvkey', 'nunique'))
        coverage.append(chunk_coverage)
    coverage_df = pd.concat(coverage).groupby(level=0).sum()
    coverage_df = coverage_df[coverage_df.index >= 2003]  # Start from 2003
    coverage_df.to_csv(os.path.join(output_dir, get_output_filename("firms_by_year.csv")))
    
    plt.figure(figsize=(10, 6))
    sns.lineplot(data=coverage_df['firms'], label="Unique Firms")
    plt.title("Number of Firms Over Time")
    plt.xlabel("Year")
    plt.ylabel("Number of Firms")
    plt.xticks(ticks=coverage_df.index, labels=coverage_df.index.astype(int), rotation=45)  # Full years
    plt.savefig(os.path.join(output_dir, get_output_filename("firms_over_time.png")), bbox_inches='tight')
    plt.close()
    logger.info(f"Firms over time plot saved as file4_{output_counter-1:03d}")
    
    # Additional analysis: Goodwill trends by major industry
    plot_goodwill_trends(processed_path, ff48_mapping, industry_data['industry_df_10'], chunk_size)

        # ------------------------
        # ------------------------
    # NEW: Average Market Cap (NYSE, AMEX, NASDAQ)
    # ------------------------
    logger.info("Calculating average market cap for NYSE, AMEX, NASDAQ exchanges...")

    market_cap_all = []
    for chunk in pd.read_csv(
        processed_path,
        chunksize=chunk_size,
        parse_dates=['crsp_date'],
        dtype={'exchcd': 'float', 'market_cap': 'float'},
        low_memory=False
    ):
        chunk = chunk.dropna(subset=['exchcd', 'market_cap'])
        chunk = chunk[chunk['exchcd'].isin([1, 2, 3])]
        chunk['year'] = chunk['crsp_date'].dt.year
        market_cap_all.append(chunk[['year', 'exchcd', 'market_cap']])

    if market_cap_all:
        df_mc = pd.concat(market_cap_all)

        # Map exchange codes to names
        exchange_map = {1: 'NYSE', 2: 'AMEX', 3: 'NASDAQ'}
        df_mc['Exchange'] = df_mc['exchcd'].map(exchange_map)

        # Overall average market cap (whole period)
        overall_avg = df_mc.groupby('Exchange')['market_cap'].mean().reset_index()
        overall_avg['year'] = 'Overall'
        overall_avg = overall_avg[['year', 'Exchange', 'market_cap']]
        overall_avg.columns = ['Year', 'Exchange', 'Avg_Market_Cap']

        # Year-over-year average market cap
        yoy_avg = df_mc.groupby(['year', 'Exchange'])['market_cap'].mean().reset_index()
        yoy_avg.columns = ['Year', 'Exchange', 'Avg_Market_Cap']

        # Combine overall + yearly into one DataFrame
        summary_df = pd.concat([yoy_avg, overall_avg], ignore_index=True)

        # Save to CSV
        summary_df.to_csv(os.path.join(output_dir, get_output_filename("market_cap.csv")), index=False)
        logger.info(f"Market cap summary saved as file4_{output_counter-1:03d}")

                 # New Plot: Year-over-year average market cap (in millions) with smoothed blue shades
        logger.info("Plotting smooth average market cap over time (in millions) with blue shades...")

        # Prepare data: convert to millions & exclude 'Overall'
        yoy_avg_million = yoy_avg.copy()
        yoy_avg_million = yoy_avg_million[yoy_avg_million['Year'] != 'Overall'].copy()
        yoy_avg_million['Year'] = yoy_avg_million['Year'].astype(int)
        yoy_avg_million = yoy_avg_million[yoy_avg_million['Year'] >= 2004]  # Start from 2004
        yoy_avg_million['Avg_Market_Cap_Million'] = yoy_avg_million['Avg_Market_Cap'] / 1e6

        plt.figure(figsize=(12, 6))
        blue_palette = {
            'NYSE': '#08306b',    # Dark Blue
            'NASDAQ': '#2171b5',    # Medium Blue
            'AMEX': '#6baed6'   # Light Blue
        }

        # For each exchange: smooth line + scatter points
        for exchange in yoy_avg_million['Exchange'].unique():
            data = yoy_avg_million[yoy_avg_million['Exchange'] == exchange].sort_values('Year')
            years = data['Year']
            mcaps = data['Avg_Market_Cap_Million']

            if len(years) >= 3:
                # Use spline for smooth curve
                from scipy.interpolate import make_interp_spline

                xnew = np.linspace(years.min(), years.max(), 300)
                spl = make_interp_spline(years, mcaps, k=3)
                y_smooth = spl(xnew)
                plt.plot(xnew, y_smooth, label=exchange, color=blue_palette.get(exchange), linewidth=2.5)
                # Add scatter points at original data
                plt.scatter(years, mcaps, color=blue_palette.get(exchange), s=40)
            else:
                # Not enough points for spline, just plot line
                plt.plot(years, mcaps, label=exchange, color=blue_palette.get(exchange), linewidth=2.5, marker='o')

        plt.title("Average Market Cap Over Time by Exchange (Since 2004)")
        plt.xlabel("Year")
        plt.ylabel("Average Market Cap (Millions)")

        # Format y-axis as integer (no decimals)
        ax = plt.gca()
        ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f'{x:.0f}'))

        # Set x-axis ticks to full years only (no decimals)
        years_to_show = sorted(yoy_avg_million['Year'].unique())
        ax.set_xticks(years_to_show)
        ax.set_xticklabels([str(y) for y in years_to_show])

        plt.legend(title="Exchange")
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, get_output_filename("market_cap_trend_smooth_millions_2004on.png")), bbox_inches='tight')
        plt.close()
        logger.info(f"Smooth average market cap (millions, from 2004) trend plot saved as file4_{output_counter-1:03d}")

    else:
        logger.warning("No data found for exchanges 1, 2, 3 when calculating market cap.")
    # ------------------------
def plot_industry_distributions(processed_path, ff48_mapping, chunk_size=500_000, total_firms=None):
    """Create donut charts for FF48 and 10-industry distributions, plus a bar chart for top 10 FF48 industries."""
    logger.info("Generating industry distribution visualizations...")
    
    firm_industries = {}
    unique_firms_with_sic = set()
    
    # Read processed data in chunks
    dtypes = {'sich': 'str'}
    for chunk in pd.read_csv(processed_path, chunksize=chunk_size, dtype=dtypes, low_memory=False):
        chunk_firms = chunk[['gvkey', 'sich']].drop_duplicates()
        
        # Split into firms with and without SIC codes
        chunk_with_sic = chunk_firms.dropna(subset=['sich']).copy()  # Add .copy() to avoid SettingWithCopyWarning
        chunk_without_sic = chunk_firms[chunk_firms['sich'].isna()]
        
        # Add firms with SIC codes to the set for counting
        unique_firms_with_sic.update(chunk_with_sic['gvkey'].unique())
        
        # Map firms with SIC codes to FF48 industries
        chunk_with_sic['ff48'] = chunk_with_sic['sich'].apply(lambda x: map_sic_to_ff48(x, ff48_mapping))
        industry_counts = chunk_with_sic.groupby('ff48')['gvkey'].nunique()
        
        for ff48, count in industry_counts.items():
            firm_industries[ff48] = firm_industries.get(ff48, 0) + count
    
    # Count firms without SIC codes (they will be assigned to "Other")
    firms_without_sic = total_firms - len(unique_firms_with_sic)
    logger.info(f"Firms with SIC: {len(unique_firms_with_sic)}, Firms without SIC: {firms_without_sic}")
    
    # Ensure all 48 industries are represented, even with zero firms
    all_industries = {i: 0 for i in range(1, 49)}
    all_industries.update(firm_industries)
    
    # Add firms without SIC codes to the "Other" category (FF48 industry 48)
    all_industries[48] = all_industries.get(48, 0) + firms_without_sic
    
    # Convert to DataFrame for FF48
    industry_df_48 = pd.DataFrame.from_dict(all_industries, orient='index', columns=['num_firms'])
    industry_df_48 = industry_df_48.sort_index()
    
    # Load FF48 industry names
    ff48_excel = pd.read_excel("/Users/carlaamodt/thesis_project/Excel own/Industry classificationFF.xlsx")
    ff48_excel['Industry number'] = pd.to_numeric(ff48_excel['Industry number'], errors='coerce').fillna(48).astype(int)
    ff48_names = dict(zip(ff48_excel['Industry number'], ff48_excel['Industry description'].fillna('Other')))
    industry_df_48['industry_name'] = industry_df_48.index.map(ff48_names).fillna('Other')
    
    # Calculate percentages
    industry_df_48['percentage'] = (industry_df_48['num_firms'] / total_firms * 100).round(2)
    
    # Save full FF48 data (all 48 industries)
    industry_df_48[['industry_name', 'num_firms', 'percentage']].to_csv(
        os.path.join(output_dir, get_output_filename("ff48_industry_distribution.csv"))
    )
    logger.info(f"FF48 industry distribution saved as file4_{output_counter-1:03d}")
    
    # Group small industries (<1%) into "Other" for the donut chart
    industry_df_48_pie = industry_df_48[industry_df_48['num_firms'] > 0].copy()
    small_industries = industry_df_48_pie[industry_df_48_pie['percentage'] < 0.0]
    if not small_industries.empty:
        other_count = small_industries['num_firms'].sum()
        industry_df_48_pie = industry_df_48_pie[industry_df_48_pie['percentage'] >= 0.0]
        industry_df_48_pie.loc[48, 'num_firms'] = other_count
        industry_df_48_pie.loc[48, 'industry_name'] = 'Other'
        industry_df_48_pie.loc[48, 'percentage'] = (other_count / total_firms * 100).round(2)
    
    # Sort by firm count for FF48
    industry_df_48_pie = industry_df_48_pie.sort_values('num_firms', ascending=False)
    
    # Group industries under 1.5% into 'Other'
    industry_df_48_pie = industry_df_48_pie.copy()
    industry_df_48_pie['percent'] = industry_df_48_pie['num_firms'] / total_firms * 100

    # Separate large and small industries
    large_industries = industry_df_48_pie[industry_df_48_pie['percent'] >= 1.5].copy()
    small_industries = industry_df_48_pie[industry_df_48_pie['percent'] < 1.5]

    # Sum small industries into 'Other'
    if not small_industries.empty:
        other_row = pd.DataFrame({
            'industry_name': ['Assigned other'],
            'num_firms': [small_industries['num_firms'].sum()],
            'percent': [small_industries['percent'].sum()]
        })
        industry_df_48_pie = pd.concat([large_industries, other_row], ignore_index=True)
    # FF48 Donut Chart with dark-to-light blue coloring
    colors_48 = sns.color_palette("Blues", len(industry_df_48_pie))[::-1]  # Dark to light Blues
    plt.figure(figsize=(12, 12))
    wedges, texts, autotexts = plt.pie(
        industry_df_48_pie['num_firms'],
        labels=industry_df_48_pie['industry_name'],
        autopct='%1.1f%%',
        startangle=90,
        colors=colors_48,
        textprops={'fontsize': 12},
        labeldistance=1.1,
        pctdistance=0.85
    )
    for text in texts:
        text.set_fontsize(12)
    for autotext in autotexts:
        autotext.set_fontsize(10)
    plt.text(0, 0, f'Total Firms\n{total_firms}', fontsize=14, ha='center', va='center', weight='bold')
    centre_circle = plt.Circle((0, 0), 0.70, fc='white')
    fig = plt.gcf()
    fig.gca().add_artist(centre_circle)
    plt.axis('equal')
    plt.title('Fama-French 48 Industry Distribution of Firms', fontsize=14, pad=20)
    output_path = os.path.join(output_dir, get_output_filename("ff48_industry_distribution_donut.png"))
    plt.savefig(output_path, bbox_inches='tight')
    plt.close()
    
    # Bar chart for top 10 FF48 industries (sorted by num_firms, descending)
    top_10_df = industry_df_48_pie.sort_values('num_firms', ascending=False).head(10)

    plt.figure(figsize=(12, 6))

    # Dark-to-light blue palette
    blue_palette = sns.color_palette("Blues", len(top_10_df))[::-1]  # reverse: darkest first

    sns.barplot(
        x='num_firms',
        y='industry_name',
        data=top_10_df,
        palette=blue_palette,
        legend=False
    )
    plt.title('Top 10 Fama-French 48 Industries by Firm Count')
    plt.xlabel('Number of Firms')
    plt.ylabel('Industry')

    plt.savefig(
        os.path.join(output_dir, get_output_filename("ff48_top_10_industries_bar.png")),
        bbox_inches='tight'
    )
    plt.close()
    logger.info(f"Top 10 FF48 industries bar chart saved as file4_{output_counter-1:03d}")
        
    # Map to 10 industries
    industry_df_48['broad_category'] = industry_df_48.index.map(map_ff48_to_10_industries)
    industry_df_10 = industry_df_48.groupby('broad_category').agg({'num_firms': 'sum'}).reset_index()
    industry_df_10['percentage'] = (industry_df_10['num_firms'] / total_firms * 100).round(2)
    industry_df_10 = industry_df_10.sort_values('num_firms', ascending=False)
    
    # 10-Industry Donut Chart with dark-to-light blue coloring
    colors_10_donut = sns.color_palette("Blues", len(industry_df_10))[::-1]  # Dark to light Blues
    plt.figure(figsize=(10, 10))  # Increased height to accommodate legend
    wedges, texts, autotexts = plt.pie(
        industry_df_10['num_firms'],
        labels=None,  # Remove labels from the pie chart to avoid overlap
        autopct='%1.1f%%',
        startangle=90,
        colors=colors_10_donut,
        textprops={'fontsize': 12},
        labeldistance=1.05,
        pctdistance=0.85
    )
    for autotext in autotexts:
        autotext.set_fontsize(10)
    plt.text(0, 0, f'Total Firms\n{total_firms}', fontsize=14, ha='center', va='center', weight='bold')
    centre_circle = plt.Circle((0, 0), 0.52, fc='white')
    fig = plt.gcf()
    fig.gca().add_artist(centre_circle)
    plt.axis('equal')
    plt.title('Combined Industry Distribution of Firms', fontsize=14, pad=20)

    # Add legend at the bottom with industry names and colors
    plt.legend(
        wedges, 
        industry_df_10['broad_category'], 
        title="Industries",
        loc="lower center", 
        bbox_to_anchor=(0.5, -0.1),  # Position below the chart
        ncol=3,  # Number of columns in the legend
        fontsize=10,
        title_fontsize=12
    )

    output_path = os.path.join(output_dir, get_output_filename("10_industry_distribution_donut.png"))
    plt.savefig(output_path, bbox_inches='tight')
    plt.close()

    # Save 10-industry data
    industry_df_10[['broad_category', 'num_firms', 'percentage']].to_csv(
        os.path.join(output_dir, get_output_filename("10_industry_distribution.csv"))
    )
    logger.info(f"10-industry distribution saved as file4_{output_counter-1:03d}")

    return {'industry_df_48': industry_df_48, 'industry_df_10': industry_df_10}

def plot_goodwill_trends(processed_path, ff48_mapping, industry_df_10, chunk_size=500_000):
    """Plot goodwill trends over time by major industry as a stacked column chart."""
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
    
    # Filter to start from 2003
    goodwill_trends = goodwill_trends[goodwill_trends['year'] >= 2003]
    
    # Pivot the data for a stacked column chart
    goodwill_pivot = goodwill_trends.pivot(index='year', columns='broad_category', values='gdwl').fillna(0)
    
    # Define a custom palette with darker, neutral colors (blues, greens, greys)
    custom_colors = [
        '#1f4e79',  # Dark blue (Technology)
        '#2e7031',  # Dark green (Manufacturing)
        '#4a4a4a',  # Dark grey (Finance)
        '#355c7d',  # Medium blue (Retail/Wholesale)
        '#5c7a5a',  # Medium green (Healthcare)
        '#6d8299',  # Light blue-grey (Services)
        '#8a9a5b',  # Light green (Consumer Goods)
        '#6b7280',  # Medium grey (Energy)
        '#a3bffa',  # Light blue (Transportation)
        '#a9bdbd',  # Light grey (Utilities)
    ]
    
    # Plot as a stacked column chart with custom colors
    plt.figure(figsize=(12, 8))
    goodwill_pivot.plot(kind='bar', stacked=True, color=custom_colors, ax=plt.gca())
    plt.title('Average Goodwill Over Time by Major Industry')
    plt.xlabel('Year')
    plt.ylabel('Average Goodwill (Millions)')
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, get_output_filename("goodwill_trends_by_industry_stacked.png")), bbox_inches='tight')
    plt.close()
    logger.info(f"Goodwill trends by industry saved as file4_{output_counter-1:03d}")

    # Alternative: Plot as a grouped bar chart for better comparison
    plt.figure(figsize=(12, 8))
    sns.barplot(data=goodwill_trends, x='year', y='gdwl', hue='broad_category', palette=colors_10)
    plt.title('Average Goodwill Over Time by Major Industry (Grouped)')
    plt.xlabel('Year')
    plt.ylabel('Average Goodwill (Millions)')
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, get_output_filename("goodwill_trends_by_industry_grouped.png")), bbox_inches='tight')
    plt.close()
    logger.info(f"Grouped goodwill trends by industry saved as file4_{output_counter-1:03d}")

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
    stats_df.to_csv(os.path.join(output_dir, get_output_filename("fama_french_overview.csv")))
    
    factor_cols = ['mkt_rf', 'smb', 'hml', 'rmw', 'cma', 'mom', 'rf']
    ff_stats = fama_french[factor_cols].describe()
    ff_stats.to_csv(os.path.join(output_dir, get_output_filename("fama_french_stats.csv")))
    logger.info(f"Fama-French stats saved as file4_{output_counter-1:03d}")
    
    # Correlation heatmap
    plt.figure(figsize=(10, 8))
    sns.heatmap(fama_french[factor_cols].corr(), annot=True, cmap='coolwarm', vmin=-1, vmax=1)
    plt.title("Correlation of Fama-French Factors")
    plt.savefig(os.path.join(output_dir, get_output_filename("fama_french_correlation_heatmap.png")), bbox_inches='tight')
    plt.close()
    logger.info(f"Fama-French correlation heatmap saved as file4_{output_counter-1:03d}")

########################################
### Main Execution ###
########################################

def main():
    """Main function to run data exploration."""
    try:
        data = load_data()
        raw_data_overview(data)
        covid19_shock_analysis(data['processed_path'])
        financial_crisis_shock_analysis(data['processed_path'])
        processed_data_exploration(data['processed_path'], data['ff48_mapping'])
        fama_french_analysis(data['fama_french'])
        logger.info("Data exploration completed!")
    except Exception as e:
        logger.error(f"Failed: {e}")
        raise

if __name__ == "__main__":
    main()