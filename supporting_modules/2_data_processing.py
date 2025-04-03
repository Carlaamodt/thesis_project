import pandas as pd
import numpy as np
import os

##################################
### 1. Load Data
##################################

def load_data(directory="data/"):
    files = {"compustat": None, "crsp_ret": None, "crsp_delist": None, "crsp_compustat": None}
    for file in os.listdir(directory):
        if file.startswith("compustat_") and files["compustat"] is None:
            files["compustat"] = os.path.join(directory, file)
        elif file.startswith("crsp_ret_") and files["crsp_ret"] is None:
            files["crsp_ret"] = os.path.join(directory, file)
        elif file.startswith("crsp_delist_") and files["crsp_delist"] is None:
            files["crsp_delist"] = os.path.join(directory, file)
        elif file.startswith("crsp_compustat_") and files["crsp_compustat"] is None:
            files["crsp_compustat"] = os.path.join(directory, file)
    if None in files.values():
        raise FileNotFoundError(f"Missing files: {', '.join(k for k, v in files.items() if v is None)}")
    print("📥 Loading data...")
    print(f"Compustat file: {files['compustat']}")
    print(f"CRSP returns file: {files['crsp_ret']}")
    print(f"CRSP delist file: {files['crsp_delist']}")
    print(f"CRSP-Compustat link file: {files['crsp_compustat']}")
    
    compustat = pd.read_csv(files["compustat"])
    print(f"Raw Compustat rows: {compustat.shape[0]}, columns: {compustat.columns.tolist()}")
    if 'date' in compustat.columns:
        compustat['date'] = pd.to_datetime(compustat['date'])
    elif 'datadate' in compustat.columns:
        compustat['datadate'] = pd.to_datetime(compustat['datadate'])
    else:
        raise ValueError("No date column ('date' or 'datadate') found in Compustat CSV")
    
    crsp_ret = pd.read_csv(files["crsp_ret"], parse_dates=['date'])
    print(f"Raw CRSP returns rows: {crsp_ret.shape[0]}")
    crsp_delist = pd.read_csv(files["crsp_delist"], parse_dates=['dlstdt'])
    print(f"Raw CRSP delist rows: {crsp_delist.shape[0]}")
    crsp_compustat = pd.read_csv(files["crsp_compustat"], parse_dates=['linkdt', 'linkenddt'])
    print(f"Raw CRSP-Compustat link rows: {crsp_compustat.shape[0]}, unique gvkey: {crsp_compustat['gvkey'].nunique()}")
    return compustat, crsp_ret, crsp_delist, crsp_compustat

##################################
### 2. Clean and Merge COMPUSTAT & CRSP
##################################

from pandas.tseries.offsets import YearEnd, MonthEnd

def merge_compustat_crsp(compustat, crsp_ret, crsp_compustat):
    print("🔄 Merging datasets for 2002–2023 with FF 6-month lag...")
    crsp_compustat = crsp_compustat[
        crsp_compustat['linktype'].isin(['LU', 'LC', 'LN']) &
        crsp_compustat['linkprim'].isin(['P', 'C']) &
        (crsp_compustat['linkdt'] <= '2023-12-31')
    ].copy()
    crsp_compustat['linkenddt'] = crsp_compustat['linkenddt'].fillna(pd.Timestamp('2023-12-31'))
    print(f"Filtered link table rows: {crsp_compustat.shape[0]}, unique gvkey: {crsp_compustat['gvkey'].nunique()}")
    
    compustat = pd.merge(compustat, crsp_compustat, on='gvkey', how='inner')
    print(f"Compustat after link merge: {compustat.shape[0]} rows")
    date_col = 'date' if 'date' in compustat.columns else 'datadate'
    compustat = compustat[
        (compustat[date_col] >= compustat['linkdt']) & 
        (compustat[date_col] <= compustat['linkenddt']) &
        (compustat[date_col] >= '2002-01-01') & 
        (compustat[date_col] <= '2023-12-31')
    ].drop_duplicates(subset=['gvkey', date_col]).rename(columns={date_col: 'compustat_date'})
    print(f"Compustat after link date filter and dedupe: {compustat.shape[0]} rows")
    
    # CRSP FF_year: July t to June t+1 = t+1
    crsp_ret['FF_year'] = crsp_ret['date'].dt.year + np.where(crsp_ret['date'].dt.month >= 7, 1, 0)
    crsp_ret = crsp_ret.rename(columns={'date': 'crsp_date'})
    
    # Compustat FF_year: June scheme alignment
    def JuneScheme(row):
        date = row['compustat_date']
        if date.month < 4:  # Jan-Mar t -> June t
            month_end = date + YearEnd(0) + MonthEnd(-6)
        else:  # Apr-Dec t -> June t+1
            month_end = date + YearEnd(0) + MonthEnd(6)
        return pd.Series({'month_end': month_end})
    
    compustat[['month_end']] = compustat.apply(JuneScheme, axis=1)
    compustat['FF_year'] = compustat['month_end'].dt.year + 1  # June t -> July t to June t+1
    print("Sample FF_year assignments:")
    print(compustat[['compustat_date', 'month_end', 'FF_year']].head(5))
    print(f"Compustat after FF date assignment: {compustat.shape[0]} rows")
    
    compustat = compustat.sort_values(['gvkey', 'compustat_date']).groupby(['gvkey', 'FF_year']).tail(1)
    print(f"Compustat after keeping latest per gvkey, FF_year: {compustat.shape[0]} rows")
    
    dupes = crsp_ret.duplicated(subset=['permno', 'crsp_date'], keep=False)
    if dupes.sum() > 0:
        print(f"⚠️ Warning: Found {dupes.sum()} duplicate rows in CRSP returns.")
        print(crsp_ret.loc[dupes].sort_values(['permno', 'crsp_date']).head(5))
    crsp_ret = crsp_ret.drop_duplicates(subset=['permno', 'crsp_date'])
    print(f"CRSP returns after dedupe: {crsp_ret.shape[0]} rows")
    
    merged = pd.merge(crsp_ret, compustat, on=['permno', 'FF_year'], how='inner')
    print(f"✅ Merged dataset shape: {merged.shape}")
    return merged

##################################
### 3. Apply Filtering Rules
##################################

def apply_filters(df):
    print("📊 Applying filtering rules...")
    df = df[df['exchcd'].isin([1, 2, 3])]
    print(f"After exchange filter: {df.shape[0]} rows")
    
    df = df.groupby('permno').filter(lambda x: len(x) >= 12)
    print(f"After min obs filter: {df.shape[0]} rows")
    
    initial_permnos = df['permno'].nunique()
    print(f"Starting unique permnos: {initial_permnos}")
    
    df_at = df[df['at'] > 0]
    at_dropped = initial_permnos - df_at['permno'].nunique()
    print(f"After positive assets filter: {df_at.shape[0]} rows, dropped {at_dropped} companies")
    
    df_ceq = df_at[df_at['ceq'] > 0]
    ceq_dropped = df_at['permno'].nunique() - df_ceq['permno'].nunique()
    print(f"After positive equity filter: {df_ceq.shape[0]} rows, dropped {ceq_dropped} companies")
    
    df_gdwl = df_ceq[df_ceq['gdwl'] > 0]
    gdwl_dropped = df_ceq['permno'].nunique() - df_gdwl['permno'].nunique()
    print(f"After positive goodwill filter: {df_gdwl.shape[0]} rows, dropped {gdwl_dropped} companies")
    
    df = df_gdwl.copy()
    
    df.loc[:, 'zero_streak'] = df.groupby('permno')['ret'].apply(
        lambda x: (x == 0).astype(int).groupby((x != 0).cumsum()).cumsum()
    ).reset_index(drop=True)
    df.loc[:, 'prc'] = df.groupby('permno')['prc'].ffill()
    df.loc[:, 'market_cap'] = df['shrout'] * df['prc'].abs()
    df.loc[:, 'is_nyse'] = (df['exchcd'] == 1).astype(int)
    
    df = df[df['zero_streak'] < 6]
    print(f"After zero return filter: {df.shape[0]} rows")
    
    df = df[df['market_cap'].notna()]
    print(f"After non-missing market_cap filter: {df.shape[0]} rows")
    
    df['market_cap_roll'] = df.groupby('permno')['market_cap'].rolling(window=36, min_periods=1).mean().reset_index(level=0, drop=True)
    df['market_cap_percentile'] = df.groupby('crsp_date')['market_cap_roll'].rank(pct=True)
    df = df[df['market_cap_percentile'] > 0.005]
    print(f"After market_cap filter: {df.shape[0]} rows")
    
    df['avg_price'] = df.groupby('permno')['prc'].rolling(window=12, min_periods=1).mean().reset_index(level=0, drop=True)
    df = df[df['avg_price'].abs() >= 1]
    print(f"After penny stock filter: {df.shape[0]} rows")
    
    df['avg_vol'] = df.groupby('permno')['vol'].rolling(window=12, min_periods=1).mean().reset_index(level=0, drop=True)
    df = df[df['vol'] >= 0.05 * df['avg_vol']]
    print(f"After volume filter: {df.shape[0]} rows")
    
    print(f"✅ Filtering completed. Final shape: {df.shape}")
    return df

##################################
### 4. Compute Goodwill Factors
##################################

def compute_goodwill_factors(df):
    print("📊 Computing Goodwill factors...")
    df['goodwill_to_sales'] = df['gdwl'] / df['revt']
    df['goodwill_to_equity'] = df['gdwl'] / df['ceq']
    df['goodwill_to_market_cap'] = df['gdwl'] / df['market_cap']
    df['goodwill_to_sales_lagged'] = df.groupby('gvkey')['goodwill_to_sales'].shift(0)
    df['goodwill_to_equity_lagged'] = df.groupby('gvkey')['goodwill_to_equity'].shift(0)
    df['goodwill_to_market_cap_lagged'] = df.groupby('gvkey')['goodwill_to_market_cap'].shift(0)
    
    for col in ['goodwill_to_sales', 'goodwill_to_equity', 'goodwill_to_market_cap',
                'goodwill_to_sales_lagged', 'goodwill_to_equity_lagged', 'goodwill_to_market_cap_lagged']:
        df[col] = df[col].replace([np.inf, -np.inf], np.nan)
    
    for col in ['goodwill_to_sales', 'goodwill_to_equity', 'goodwill_to_market_cap']:
        lower, upper = df[col].quantile([0.01, 0.99])
        df[col] = df[col].clip(lower, upper)
    
    print("📊 Goodwill Factor Summary:")
    print(df[['goodwill_to_sales', 'goodwill_to_equity', 'goodwill_to_market_cap']].describe())
    return df

##################################
### 5. Adjust for Delistings
##################################

def adjust_for_delistings(df, crsp_delist):
    print("⚠️ Adjusting for delisting returns...")
    df['crsp_date_month'] = df['crsp_date'].dt.to_period('M')
    crsp_delist['dlstdt_month'] = crsp_delist['dlstdt'].dt.to_period('M')
    df = pd.merge(df, crsp_delist, left_on=['permno', 'crsp_date_month'], 
                  right_on=['permno', 'dlstdt_month'], how='left')
    df['ret'] = np.where(df['dlret'].notna(), 
                         (1 + df['ret'].fillna(0)) * (1 + df['dlret'].fillna(0)) - 1, 
                         df['ret'])
    return df

##################################
### 6. Winsorize Returns
##################################

def winsorize_and_filter(df):
    print("📊 Winsorizing returns (1%-99%)...")
    lower, upper = df['ret'].quantile([0.01, 0.99])
    df['ret'] = df['ret'].clip(lower, upper)
    return df

##################################
### 7. Save Final Data
##################################

def save_processed_data(df, filename="processed_data", directory="data/"):
    print("💾 Saving processed data...")
    os.makedirs(directory, exist_ok=True)
    filepath = os.path.join(directory, f"{filename}.csv")
    df.to_csv(filepath, index=False)
    print(f"✅ Data saved to: {filepath}. Rows: {df.shape[0]}")
    return df

##################################
### Main Pipeline Execution
##################################

def main():
    try:
        compustat, crsp_ret, crsp_delist, crsp_compustat = load_data()
        merged = merge_compustat_crsp(compustat, crsp_ret, crsp_compustat)
        filtered = apply_filters(merged)
        goodwill_factors = compute_goodwill_factors(filtered)
        delist_adjusted = adjust_for_delistings(goodwill_factors, crsp_delist)
        winsorized_data = winsorize_and_filter(delist_adjusted)
        save_processed_data(winsorized_data)
        print("✅ Full pipeline completed successfully!")
    except Exception as e:
        print(f"❌ Pipeline failed: {e}")
        raise

if __name__ == "__main__":
    main()