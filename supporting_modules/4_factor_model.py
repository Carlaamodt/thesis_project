import pandas as pd
import numpy as np
import statsmodels.api as sm
import os

#########################
### Load Factor Data ###
#########################

def load_data(weighting="equal"):
    """Loads GA factor returns and Fama-French factors, aligned annually."""
    print(f"📥 Loading {weighting}-weighted GA factor returns...")
    ga_factor = pd.read_csv(f"data/ga_factor_returns_annual_{weighting}.csv", parse_dates=['year'])
    
    print("📥 Loading Fama-French factors...")
    ff_factors = pd.read_csv("data/FamaFrench_factors_with_momentum.csv", parse_dates=['quarter'])
    
    # Convert Fama-French quarterly to annual
    ff_factors['year'] = ff_factors['quarter'].dt.year
    ff_factors = ff_factors.groupby('year').mean().reset_index()  # Average monthly factors to annual
    
    ga_factor.columns = ga_factor.columns.str.lower()
    ff_factors.columns = ff_factors.columns.str.lower()
    
    start_date = ga_factor['year'].min()
    ff_factors = ff_factors[ff_factors['year'] >= start_date]
    
    print("✅ Fama-French dataset filtered. Date range:", ff_factors['year'].min(), "to", ff_factors['year'].max())
    return ga_factor, ff_factors

#############################
### Merge and Prepare Data ###
#############################

def merge_data(ga_factor, ff_factors):
    """Merges GA factor data with Fama-French factors on year."""
    print("🔄 Merging factor datasets...")
    merged_df = pd.merge(ga_factor, ff_factors, on="year", how="inner")
    print("✅ Merged dataset. Total periods:", len(merged_df))
    return merged_df

#########################
### Run Factor Models ###
#########################

def run_factor_models(df):
    """Runs factor models with GA factor."""
    print("📊 Running regression models...")
    if 'ga_factor' not in df.columns:
        print("❌ Error: 'ga_factor' column missing!")
        return
    
    factor_models = {
        "CAPM": df[["mkt"]],
        "Fama-French 3-Factor": df[["mkt", "smb", "hml"]],
        "Fama-French 5-Factor": df[["mkt", "smb", "hml", "rmw", "cma"]],
        "Fama-French 5 + Momentum + GA": df[["mkt", "smb", "hml", "rmw", "cma", "mom", "ga_factor"]],
    }
    
    results = {}
    for model, factors in factor_models.items():
        factors = sm.add_constant(factors, has_constant='add')
        y = df["ga_factor"]
        model_fit = sm.OLS(y, factors, missing='drop').fit()
        results[model] = model_fit.summary()
        print(f"\n📊 {model} Regression Results:\n", model_fit.summary())
    return results

#########################
### Main Execution ###
#########################

def main():
    for weighting in ["equal", "value"]:
        ga_factor, ff_factors = load_data(weighting=weighting)
        df = merge_data(ga_factor, ff_factors)
        if df.empty:
            print(f"❌ Merged dataset empty for {weighting}-weighted!")
            return
        run_factor_models(df)

if __name__ == "__main__":
    main()