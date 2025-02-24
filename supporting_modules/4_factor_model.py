import pandas as pd
import numpy as np
import statsmodels.api as sm
import os

#########################
### Load Factor Data ###
#########################

def load_data():
    """Loads GA factor returns and Fama-French factors, ensuring date alignment."""
    print("📥 Loading GA factor returns...")
    ga_factor = pd.read_csv("data/ga_factor_returns_quarterly.csv", parse_dates=['quarter'])
    
    print("📥 Loading Fama-French factors...")
    ff_factors = pd.read_csv("data/FamaFrench_factors.csv", parse_dates=['quarter'])
    
    # Ensure dates are at the quarter end (no timestamps)
    ga_factor['quarter'] = ga_factor['quarter'].dt.to_period("Q").dt.to_timestamp()
    ff_factors['quarter'] = ff_factors['quarter'].dt.to_period("Q").dt.to_timestamp()
    
    # Convert column names to lowercase for consistency
    ga_factor.columns = ga_factor.columns.str.lower()
    ff_factors.columns = ff_factors.columns.str.lower()
    
    # 🔍 Filter Fama-French dataset to match GA factor date range
    start_date = ga_factor['quarter'].min()
    ff_factors = ff_factors[ff_factors['quarter'] >= start_date]
    
    print("✅ Fama-French dataset filtered. New date range:", ff_factors['quarter'].min(), "to", ff_factors['quarter'].max())
    
    return ga_factor, ff_factors

#############################
### Merge and Prepare Data ###
#############################

def merge_data(ga_factor, ff_factors):
    """Merges GA factor data with Fama-French factors on quarter."""
    print("🔄 Merging factor datasets...")
    merged_df = pd.merge(ga_factor, ff_factors, on="quarter", how="inner")
    
    # Ensure all column names are in lowercase for consistency
    merged_df.columns = merged_df.columns.str.lower()
    
    # Ensure 'ga_factor' column exists
    if 'ga_factor' not in merged_df.columns:
        print("❌ 'ga_factor' column is missing after merge! Check data.")
    else:
        print("✅ 'ga_factor' column found in merged dataset.")
    
    print("✅ Merged dataset. Total periods:", len(merged_df))
    print("📝 Merged Dataset Columns:", merged_df.columns)
    print("📝 Merged Dataset Preview:\n", merged_df.head())
    
    return merged_df

#########################
### Run Factor Models ###
#########################

def run_factor_models(df):
    """Runs CAPM, Fama-French models with GA factor and reports results."""
    print("📊 Running regression models...")
    
    if 'ga_factor' not in df.columns:
        print("❌ Error: 'ga_factor' column is missing in dataset! Check data consistency.")
        return
    
    factor_models = {
        "CAPM": df[["mkt"]],
        "Fama-French 3-Factor": df[["mkt", "smb", "hml"]],
        "Fama-French 5-Factor": df[["mkt", "smb", "hml", "rmw", "cma"]],
        "Fama-French 5 + GA Factor": df[["mkt", "smb", "hml", "rmw", "cma", "ga_factor"]],
    }
    
    results = {}
    for model, factors in factor_models.items():
        factors = sm.add_constant(factors, has_constant='add')  # Add intercept
        y = df["ga_factor"]  # Dependent variable
        model_fit = sm.OLS(y, factors, missing='drop').fit()
        results[model] = model_fit.summary()
        print(f"\n📊 {model} Regression Results:\n", model_fit.summary())
    
    return results

#########################
### Main Execution ###
#########################

def main():
    ga_factor, ff_factors = load_data()
    df = merge_data(ga_factor, ff_factors)
    if df.empty:
        print("❌ Merged dataset is empty! Check data consistency.")
        return
    run_factor_models(df)

if __name__ == "__main__":
    main()