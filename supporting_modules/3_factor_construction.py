import pandas as pd
import numpy as np
import os

#########################
### Load Processed Data ###
#########################

def load_processed_data(directory="data/", filename="processed_data.csv"):
    """Loads the cleaned dataset from previous steps."""
    filepath = os.path.join(directory, filename)
    
    if not os.path.exists(filepath):
        print(f"❌ Error: Processed data file not found at {filepath}")
        return None

    print("📥 Loading processed data...")
    df = pd.read_csv(filepath, parse_dates=['date'])

    if df.empty:
        print("❌ Error: Loaded dataset is empty!")
        return None
    
    print(f"✅ Processed data loaded. Total rows: {df.shape[0]}")
    return df

#############################
### Convert to Quarterly Data ###
#############################

def resample_to_quarterly(df):
    """Converts data to quarterly frequency for portfolio sorting."""
    print("📅 Converting to quarterly holding periods...")

    # Ensure dates are end of quarter
    df['quarter'] = df['date'].dt.to_period("Q").dt.end_time

    # Use latest available GA values per quarter
    df = df.sort_values(['gvkey', 'date'])
    df = df.groupby(['gvkey', 'quarter']).last().reset_index()

    print(f"✅ Data resampled to quarterly. Total rows: {df.shape[0]}")
    return df

#############################
### Sort Firms into Quintiles ###
#############################

def assign_quintiles(df, ga_column="GA1"):
    """
    Assigns firms into quintiles based on GA1 (default) or GA2.
    Ensures all quintiles exist, handles errors if qcut fails.
    """
    print("📊 Sorting firms into quintiles...")

    def safe_qcut(x):
        try:
            return pd.qcut(x, q=5, labels=[1, 2, 3, 4, 5], duplicates='drop')
        except ValueError:
            print(f"⚠️ Warning: Not enough unique values to create 5 quintiles for quarter {x.name}. Filling with NaN.")
            return pd.Series(np.nan, index=x.index)

    df['quintile'] = df.groupby('quarter')[ga_column].transform(safe_qcut)

    # Remove rows where quintile could not be assigned
    df = df.dropna(subset=['quintile'])
    df['quintile'] = df['quintile'].astype(int)

    print(df['quintile'].value_counts())  # Debugging output
    print(f"✅ Firms assigned to quintiles. Total rows: {df.shape[0]}")
    return df

#############################
### Compute Portfolio Returns ###
#############################

def compute_portfolio_returns(df, weighting="equal"):
    """
    Computes quintile portfolio returns using quarterly rebalancing.
    Options:
        - "equal": Equal-weighted portfolio returns (default).
        - "value": Value-weighted portfolio returns (weighted by market cap).
    """
    print(f"📈 Calculating {weighting}-weighted portfolio returns (quarterly)...")

    df['ret'] = pd.to_numeric(df['ret'], errors='coerce')

    if weighting == "value":
        df['mkt_cap'] = df['csho'] * df['prc']  # Market Cap = Shares Outstanding * Price
        df['weight'] = df.groupby(['quarter', 'quintile'])['mkt_cap'].transform(lambda x: x / x.sum())
        df['weighted_ret'] = df['ret'] * df['weight']
        portfolio_returns = df.groupby(['quarter', 'quintile'])['weighted_ret'].sum().unstack()
    else:
        portfolio_returns = df.groupby(['quarter', 'quintile'])['ret'].mean().unstack()

    portfolio_returns.columns = portfolio_returns.columns.astype(int)

    print("✅ Portfolio returns calculated.")
    print("Portfolio Returns Shape:", portfolio_returns.shape)
    print("Portfolio Returns Columns:", portfolio_returns.columns)

    if portfolio_returns.empty:
        print("❌ Error: Portfolio returns are empty. Check the input data!")
        return None

    return portfolio_returns

#############################
### Construct GA Factor ###
#############################

def construct_ga_factor(portfolio_returns):
    """Constructs the GA factor as (High GA Portfolio - Low GA Portfolio)."""
    print("⚖️ Constructing GA Factor...")

    if portfolio_returns is None or portfolio_returns.empty:
        print("❌ Error: Portfolio returns are empty. Cannot compute GA Factor.")
        return None

    print("Available Portfolio Returns Columns:", portfolio_returns.columns)

    for quintile in [1, 5]:
        if quintile not in portfolio_returns.columns:
            print(f"⚠️ Warning: Quintile {quintile} is missing, filling with NaN.")
            portfolio_returns[quintile] = np.nan

    portfolio_returns['GA_factor'] = portfolio_returns[5] - portfolio_returns[1]

    print("✅ GA factor constructed.")
    return portfolio_returns

#############################
### Save Factor Portfolio Returns ###
#############################

def save_factor_data(df, filename="ga_factor_returns_quarterly", directory="data/"):
    """Saves factor returns as a CSV for further analysis."""
    if df is None or df.empty:
        print("❌ Error: No data to save.")
        return

    os.makedirs(directory, exist_ok=True)
    filepath = os.path.join(directory, f"{filename}.csv")
    df.to_csv(filepath, index=True)
    print(f"💾 Factor portfolio returns saved: {filepath}")

#############################
### Main Execution ###
#############################

def main():
    """Main function to construct GA-based factor portfolios (quarterly)."""
    
    df = load_processed_data()
    if df is None:
        return  

    df = resample_to_quarterly(df)

    df = assign_quintiles(df, ga_column="GA1")
    if df is None:
        return  

    portfolio_returns = compute_portfolio_returns(df, weighting="equal")
    if portfolio_returns is None:
        return  

    factor_returns = construct_ga_factor(portfolio_returns)
    if factor_returns is None:
        return  

    save_factor_data(factor_returns)

    print("✅ Factor portfolio construction complete (Quarterly Rebalancing)!")

if __name__ == "__main__":
    main()