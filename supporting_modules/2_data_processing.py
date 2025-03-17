import pandas as pd
import numpy as np
import os

##################################
### 1. Load Data
##################################

def load_data(directory="data/"):
    files = {"compustat": None, "crsp_ret": None, "crsp_delist": None, "crsp_compustat": None}
    for file in os.listdir(directory):
        for key in files.keys():
            if key in file:
                files[key] = os.path.join(directory, file)
    if None in files.values():
        raise FileNotFoundError(f"Missing files: {', '.join(k for k, v in files.items() if v is None)}")
    print("📥 Loading data...")
    compustat = pd.read_csv(files["compustat"], parse_dates=['date'])
    crsp_ret = pd.read_csv(files["crsp_ret"], parse_dates=['date'])
    crsp_delist = pd.read_csv(files["crsp_delist"], parse_dates=['dlstdt'])
    crsp_compustat = pd.read_csv(files["crsp_compustat"], parse_dates=['linkdt', 'linkenddt'])
    return compustat, crsp_ret, crsp_delist, crsp_compustat

##################################
### 2. Clean and Merge COMPUSTAT & CRSP
##################################

def merge_compustat_crsp(compustat, crsp_ret, crsp_compustat):
    print("🔄 Merging datasets for 2002–2023...")
    # Filter link table for strong matches
    crsp_compustat = crsp_compustat[
        crsp_compustat['linktype'].isin(['LU', 'LC', 'LN']) &
        crsp_compustat['linkprim'].isin(['P', 'C'])
    ].copy()
    crsp_compustat['linkenddt'] = crsp_compustat['linkenddt'].fillna(pd.Timestamp('2023-12-31'))

    # Compustat link time check
    compustat = pd.merge(compustat, crsp_compustat, on='gvkey', how='inner')
    compustat = compustat[
        (compustat['date'] >= compustat['linkdt']) & 
        (compustat['date'] <= compustat['linkenddt'])
    ].drop_duplicates(subset=['gvkey', 'date']).rename(columns={'date': 'compustat_date'})

    # Prepare FF_year
    compustat['FF_year'] = compustat['compustat_date'].dt.year + 1
    crsp_ret['FF_year'] = crsp_ret['date'].dt.year + np.where(crsp_ret['date'].dt.month <= 6, 0, 1)
    crsp_ret = crsp_ret.rename(columns={'date': 'crsp_date'})

    # Merge on permno + FF_year
    merged = pd.merge(crsp_ret, compustat, on=['permno', 'FF_year'], how='inner')
    print(f"✅ Final merged dataset shape: {merged.shape}")
    return merged

##################################
### 3. Compute GA (Δgdwl/gdwl)
##################################

def compute_ga(merged):
    print("📊 Calculating GA (Δgdwl/gdwl) for firms with goodwill...")

    # Step 1: Identify firms that ever have goodwill
    merged['has_goodwill_firm'] = np.where(merged['gdwl'] > 0, 1, 0)
    goodwill_firms = merged.loc[merged['gdwl'] > 0, 'gvkey'].unique()
    print(f"💰 Number of unique firms with goodwill: {len(goodwill_firms)}")

    # Step 2: Prepare for GA calculation
    annual_ga = merged[['gvkey', 'compustat_date', 'gdwl']].drop_duplicates().sort_values(['gvkey', 'compustat_date'])

    # Step 3: Lagged goodwill
    annual_ga['gdwl_lagged'] = annual_ga.groupby('gvkey')['gdwl'].shift(1)

    # Step 4: GA calculation - **Leave as NaN when both current and lagged gdwl are zero**
    annual_ga['ga'] = np.where(
        (annual_ga['gdwl_lagged'] != 0) & (annual_ga['gdwl'] != 0),
        (annual_ga['gdwl'] - annual_ga['gdwl_lagged']) / annual_ga['gdwl_lagged'],
        np.nan
    )
    # Optional: Keep lagged GA for future use
    annual_ga['ga_lagged'] = annual_ga.groupby('gvkey')['ga'].shift(1)

    # Step 5: Add FF_year for correct merging
    annual_ga['FF_year'] = annual_ga['compustat_date'].dt.year + 1

    # Diagnostics
    print(f"🧹 Rows before dropping NA GA: {annual_ga.shape[0]}")
    print(f"❌ Rows where GA is NaN: {(annual_ga['ga'].isna()).sum()} (indicating no goodwill activity or both zeros)")

    # Step 6: Sample for gvkey 1004
    sample_1004 = annual_ga[annual_ga['gvkey'] == 1004].head(10)
    print("\n📊 Sample for gvkey 1004:\n", sample_1004[['gvkey', 'compustat_date', 'gdwl', 'gdwl_lagged', 'ga', 'ga_lagged']])

    # Step 7: Summary stats for non-NaN GA
    meaningful_ga = annual_ga.dropna(subset=['ga'])
    print("\n📊 GA Summary Statistics (non-NaN only):\n", meaningful_ga['ga'].describe())

    # Step 8: Merge GA back to merged dataset
    merged['FF_year'] = merged['crsp_date'].dt.year + np.where(merged['crsp_date'].dt.month <= 6, 0, 1)
    merged = pd.merge(merged, annual_ga[['gvkey', 'FF_year', 'ga', 'ga_lagged']], 
                      on=['gvkey', 'FF_year'], how='left')

    # ✅ Final output
    print("\n✅ GA successfully merged. GA summary in merged data (excluding NaNs):")
    print(merged['ga'].dropna().describe())
    print(f"Total rows in merged: {merged.shape[0]}")
    return merged

##################################
### 4. Adjust for Delistings
##################################

def adjust_for_delistings(df, crsp_delist):
    print("⚠️ Adjusting for delisting returns...")
    df = pd.merge(df, crsp_delist, left_on=['permno', 'crsp_date'], right_on=['permno', 'dlstdt'], how='left')
    df['ret'] = np.where(df['dlret'].notna(), (1 + df['ret'].fillna(0)) * (1 + df['dlret'].fillna(0)) - 1, df['ret'])
    df = df[(df['dlstdt'].isna()) | (df['crsp_date'] <= df['dlstdt'])]
    print(f"✅ Delisting adjustments applied. Rows: {df.shape[0]}")
    return df

##################################
### 5. Winsorize Returns (Optional)
##################################

def winsorize_and_filter(df):
    print("📊 Winsorizing returns (1%-99%)...")
    lower, upper = df['ret'].quantile([0.01, 0.99])
    df['ret'] = df['ret'].clip(lower, upper)
    df = df[(df['ret'] >= -1) & (df['ret'] <= 1)]
    print(f"✅ Returns winsorized. Final rows: {df.shape[0]}")
    return df

##################################
### 6. Save Final Data
##################################

def save_processed_data(df, filename="processed_data", directory="data/"):
    print("💾 Saving processed data...")
    os.makedirs(directory, exist_ok=True)
    filepath = os.path.join(directory, f"{filename}.csv")
    df.to_csv(filepath, index=False)
    print(f"✅ Data saved to: {filepath}. Rows: {df.shape[0]}")
    return df

##################################
### Main Pipeline
##################################

def main():
    compustat, crsp_ret, crsp_delist, crsp_compustat = load_data()
    merged = merge_compustat_crsp(compustat, crsp_ret, crsp_compustat)
    ga_computed = compute_ga(merged)
    delist_adjusted = adjust_for_delistings(ga_computed, crsp_delist)
    final_data = winsorize_and_filter(delist_adjusted)
    save_processed_data(final_data)
    print("✅ Pipeline completed!")

if __name__ == "__main__":
    main()