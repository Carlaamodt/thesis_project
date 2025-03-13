import pandas as pd
import numpy as np
import os

#########################
### Load Extracted Data ###
#########################

def load_data(directory="data/"):
    """Loads extracted Compustat, CRSP, and linking table data."""
    files = {
        "compustat": None,
        "crsp_ret": None,
        "crsp_delist": None,
        "crsp_compustat": None
    }
    
    # Identify latest timestamped files
    for file in os.listdir(directory):
        for key in files.keys():
            if key in file:
                files[key] = os.path.join(directory, file)

    # Read data
    print("📥 Loading data...")
    compustat = pd.read_csv(files["compustat"], parse_dates=['date'])
    crsp_ret = pd.read_csv(files["crsp_ret"], parse_dates=['date'])
    crsp_delist = pd.read_csv(files["crsp_delist"], parse_dates=['dlstdt'])
    crsp_compustat = pd.read_csv(files["crsp_compustat"], parse_dates=['linkdt', 'linkenddt'])

    return compustat, crsp_ret, crsp_delist, crsp_compustat

#########################
### Merge Datasets ###
#########################

def merge_compustat_crsp(compustat, crsp_ret, crsp_compustat):
    """Merges Compustat (financials) with CRSP (returns) using GVKEY-PERMNO links."""

    print("🔄 Merging datasets...")

    # Adjust linking dates for valid ranges
    crsp_compustat['linkenddt'] = crsp_compustat['linkenddt'].fillna(pd.Timestamp('2023-12-31'))

    # Merge Compustat with CRSP link table
    merged = pd.merge(compustat, crsp_compustat, on="gvkey", how="left")

    # Keep only valid links (dates within range)
    merged = merged[(merged['date'] >= merged['linkdt']) & (merged['date'] <= merged['linkenddt'])]

    # Merge with CRSP returns
    merged = pd.merge(merged, crsp_ret, on=["permno", "date"], how="left")

    # Remove unnecessary columns
    merged = merged.drop(columns=['linkdt', 'linkenddt', 'linktype', 'linkprim'])

    print(f"✅ Merged dataset shape: {merged.shape}")
    return merged

#############################
### Compute Goodwill Intensity ###
#############################

def compute_ga_factors(df):
    """Computes Goodwill Intensity as percentage change in goodwill."""

    print("📊 Calculating Goodwill Intensity...")

    # Sort data by firm and time
    df = df.sort_values(by=['gvkey', 'date'])

    # Compute percentage change in Goodwill
    df['gdwl_prev'] = df.groupby('gvkey')['gdwl'].shift(1)  # Previous year's goodwill
    df['goodwill_intensity'] = (df['gdwl'] - df['gdwl_prev']) / df['gdwl_prev']

    # Handle edge cases: missing, zero denominator, or infinite values
    df['goodwill_intensity'] = df['goodwill_intensity'].fillna(0).replace([float('inf'), -float('inf')], 0)

    # Winsorize to avoid extreme outliers (1st & 99th percentile)
    df['goodwill_intensity'] = df['goodwill_intensity'].clip(
        df['goodwill_intensity'].quantile(0.01), 
        df['goodwill_intensity'].quantile(0.99)
    )

    # Exclude extreme returns
    df = df[(df['ret'] >= -1) & (df['ret'] <= 1)]  # Exclude returns outside [-100%, 100%]

    # Drop rows where goodwill_intensity could not be computed
    df = df.dropna(subset=['goodwill_intensity'])

    print(f"✅ Goodwill Intensity computed. Remaining rows: {df.shape[0]}")
    return df

#############################
### June Scheme Adjustment ###
#############################

def apply_june_scheme(df):
    """Aligns goodwill intensity to July-June portfolio periods."""
    print("🔁 Applying June Scheme Adjustment for portfolio formation...")

    # Fiscal year for info
    df['fiscal_year'] = df['date'].dt.year

    # Lag goodwill intensity for portfolio formation (using last year's data)
    df['ga_lagged'] = df.groupby('gvkey')['goodwill_intensity'].shift(1)

    # Align to Fama-French July-June year
    df['FF_year'] = df['date'].dt.year + np.where(df['date'].dt.month <= 6, 0, 1)

    print(f"✅ June Scheme applied. Sample:\n{df[['gvkey', 'date', 'ga_lagged', 'FF_year']].head(5)}")
    return df

#############################
### Adjust for Delistings ###
#############################

def adjust_for_delistings(df, crsp_delist):
    """Adjusts stock returns for firms that have been delisted."""
    
    print("⚠️ Adjusting for delisting bias...")

    # Merge delisting returns
    df = pd.merge(df, crsp_delist, on="permno", how="left")

    # If a firm has a delisting return, update the last return
    df['ret'] = np.where(df['dlret'].notna(), df['dlret'], df['ret'])

    # Drop delisted stocks after last trade date
    df = df[df['date'] <= df['dlstdt'].fillna(pd.Timestamp('2023-12-31'))]

    print(f"✅ Delisting adjustments done. Rows remaining: {df.shape[0]}")
    return df

#############################
### Save Processed Data ###
#############################

def save_processed_data(df, filename="processed_data", directory="data/"):
    """Saves processed data as a CSV."""
    os.makedirs(directory, exist_ok=True)
    filepath = os.path.join(directory, f"{filename}.csv")
    df.to_csv(filepath, index=False)
    print(f"💾 Processed data saved: {filepath}")

#############################
### Main Function ###
#############################

def main():
    """Main function to process data for factor modeling."""
    
    # Load extracted datasets
    compustat, crsp_ret, crsp_delist, crsp_compustat = load_data()

    # Merge datasets
    merged_data = merge_compustat_crsp(compustat, crsp_ret, crsp_compustat)

    # Compute GA factors
    processed_data = compute_ga_factors(merged_data)

    # Apply June scheme
    processed_data = apply_june_scheme(processed_data)

    # Adjust for delisting bias
    final_data = adjust_for_delistings(processed_data, crsp_delist)

    # Save cleaned dataset
    save_processed_data(final_data)

    print("✅ Data processing completed!")

if __name__ == "__main__":
    main()
