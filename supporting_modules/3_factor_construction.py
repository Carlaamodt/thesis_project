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
### Convert to Annual Data ###
#############################

def resample_to_annual(df):
    """Converts data to annual frequency for portfolio sorting."""
    print("📅 Converting to annual holding periods...")
    df['year'] = df['date'].dt.year
    df = df.sort_values(['gvkey', 'date'])
    df = df.groupby(['gvkey', 'year']).last().reset_index()
    print(f"✅ Data resampled to annual. Total rows: {df.shape[0]}")
    return df

#############################
### Sort Firms into Quintiles ###
#############################

def assign_quintiles(df, ga_column="goodwill_intensity"):
    """Assigns firms into quintiles based on goodwill_intensity."""
    print("📊 Sorting firms into quintiles...")

    def safe_qcut(x):
        try:
            return pd.qcut(x, q=5, labels=[1, 2, 3, 4, 5], duplicates='drop')
        except ValueError:
            print(f"⚠️ Warning: Not enough unique values for 5 quintiles in year {x.name}. Filling with NaN.")
            return pd.Series(np.nan, index=x.index)

    df['quintile'] = df.groupby('year')[ga_column].transform(safe_qcut)
    df = df.dropna(subset=['quintile'])
    df['quintile'] = df['quintile'].astype(int)

    print(df['quintile'].value_counts())  # Debugging output
    print(f"✅ Firms assigned to quintiles. Total rows: {df.shape[0]}")
    return df

#############################
### Compute Portfolio Returns ###
#############################

def compute_portfolio_returns(df, weighting="equal"):
    """Computes quintile portfolio returns with annual rebalancing."""
    print(f"📈 Calculating {weighting}-weighted portfolio returns (annual)...")
    df['ret'] = pd.to_numeric(df['ret'], errors='coerce')

    if weighting == "value":
        df['mkt_cap'] = df['csho'] * df['prc'].abs()  # Market Cap = Shares * Price (abs for negative prices)
        df['weight'] = df.groupby(['year', 'quintile'])['mkt_cap'].transform(lambda x: x / x.sum())
        df['weighted_ret'] = df['ret'] * df['weight']
        portfolio_returns = df.groupby(['year', 'quintile'])['weighted_ret'].sum().unstack()
    else:
        portfolio_returns = df.groupby(['year', 'quintile'])['ret'].mean().unstack()

    portfolio_returns.columns = portfolio_returns.columns.astype(int)
    print("✅ Portfolio returns calculated.")
    print("Portfolio Returns Shape:", portfolio_returns.shape)
    print("Portfolio Returns Columns:", portfolio_returns.columns)
    return portfolio_returns

#############################
### Construct GA Factor ###
#############################

def construct_ga_factor(portfolio_returns):
    """Constructs the GA factor as (High - Low)."""
    print("⚖️ Constructing GA Factor...")
    if portfolio_returns is None or portfolio_returns.empty:
        print("❌ Error: Portfolio returns are empty.")
        return None

    for quintile in [1, 5]:
        if quintile not in portfolio_returns.columns:
            print(f"⚠️ Warning: Quintile {quintile} missing, filling with NaN.")
            portfolio_returns[quintile] = np.nan

    portfolio_returns['GA_factor'] = portfolio_returns[5] - portfolio_returns[1]
    print("✅ GA factor constructed.")
    return portfolio_returns

#############################
### Save Factor Portfolio Returns ###
#############################

def save_factor_data(df, filename="ga_factor_returns_annual", directory="data/"):
    """Saves factor returns as a CSV."""
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
    """Constructs GA-based factor portfolios (annual rebalancing)."""
    df = load_processed_data()
    if df is None:
        return  
    df = resample_to_annual(df)
    df = assign_quintiles(df, ga_column="goodwill_intensity")
    if df is None:
        return  
    # Equal-weighted
    portfolio_returns_eq = compute_portfolio_returns(df, weighting="equal")
    if portfolio_returns_eq is None:
        return  
    factor_returns_eq = construct_ga_factor(portfolio_returns_eq)
    save_factor_data(factor_returns_eq, filename="ga_factor_returns_annual_equal")
    # Value-weighted
    portfolio_returns_val = compute_portfolio_returns(df, weighting="value")
    if portfolio_returns_val is None:
        return  
    factor_returns_val = construct_ga_factor(portfolio_returns_val)
    save_factor_data(factor_returns_val, filename="ga_factor_returns_annual_value")
    print("✅ Factor portfolio construction complete (Annual Rebalancing)!")

if __name__ == "__main__":
    main()