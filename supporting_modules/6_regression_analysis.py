import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import statsmodels.api as sm
import os

# Configurations
SIGNIFICANCE_LEVEL = 0.05
ALPHA_THRESHOLD = 0.02  # 2% annualized
ROLLING_WINDOW = 60  # months

output_dir = "analysis_output"
os.makedirs(output_dir, exist_ok=True)

########################################
### Load Regression Results ###
########################################

def load_regression_results(file_path=os.path.join("output", "ga_factor_regression_results_monthly.xlsx")):
    print("📥 Loading regression results...")
    if not os.path.exists(file_path):
        print(f"❌ File not found: {file_path}. Please ensure File 4 has run.")
        return {}
    
    xl = pd.read_excel(file_path, sheet_name=None)
    print(f"Loaded {len(xl)} sheets from {file_path}")
    return xl

########################################
### Visual Summary Plots ###
########################################

def plot_summary_tables(results_dict):
    print("📊 Generating summary plots...")
    for ga_key, df in results_dict.items():
        if df is None or df.empty:
            print(f"⚠️ No data for {ga_key}—skipping plots.")
            continue

        # Alpha by model
        plt.figure(figsize=(10, 6))
        sns.barplot(data=df, x="Model", y="Alpha (annualized)", hue="Weighting")
        plt.axhline(ALPHA_THRESHOLD, linestyle="--", color="green", label="2% Threshold")
        plt.axhline(0, color="black")
        plt.title(f"Annualized Alpha by Model – {ga_key}")
        plt.legend()
        plt.tight_layout()
        filepath = os.path.join(output_dir, f"{ga_key}_alpha_by_model.png")
        plt.savefig(filepath)
        plt.close()
        print(f"✅ Saved: {filepath}")

        # R-squared by model
        plt.figure(figsize=(10, 6))
        sns.barplot(data=df, x="Model", y="R-squared", hue="Weighting")
        plt.title(f"R-squared by Model – {ga_key}")
        plt.axhline(0, color="black")
        plt.tight_layout()
        filepath = os.path.join(output_dir, f"{ga_key}_rsquared_by_model.png")
        plt.savefig(filepath)
        plt.close()
        print(f"✅ Saved: {filepath}")

########################################
### Hypothesis Tests ###
########################################

def test_alpha_significance(results_dict):
    print("🔍 Testing alpha significance...")
    summary = []
    for ga_key, df in results_dict.items():
        if df is None or df.empty:
            print(f"⚠️ No data for {ga_key}—skipping test.")
            continue
        # Assuming File 4 uses "Alpha p-value (HAC)"—adjust if different
        df_filtered = df[(df['Alpha p-value (HAC)'] < SIGNIFICANCE_LEVEL) & 
                         (df['Alpha (annualized)'].abs() >= ALPHA_THRESHOLD)]
        for _, row in df_filtered.iterrows():
            summary.append({
                "GA": ga_key,
                "Model": row["Model"],
                "Weighting": row["Weighting"],
                "Alpha": row["Alpha (annualized)"],
                "p-value": row["Alpha p-value (HAC)"]
            })
    result_df = pd.DataFrame(summary)
    if result_df.empty:
        print("⚠️ No significant alphas found.")
    else:
        filepath = os.path.join(output_dir, "significant_alphas.xlsx")
        result_df.to_excel(filepath, index=False)
        print(f"✅ Saved significant alphas: {filepath}")

########################################
### Rolling Regressions for GA3 ###
########################################

def rolling_regression(ga_path, ff_path, ga_choice="goodwill_to_market_cap_lagged"):
    print(f"🔁 Running rolling regression for {ga_choice} (equal-weighted)...")
    # Use ga_factors.csv for consistency with pipeline
    if not os.path.exists(ga_path):
        print(f"❌ GA factors file not found: {ga_path}")
        return
    
    ga = pd.read_csv(ga_path, parse_dates=['date'])
    ff = pd.read_csv(ff_path, parse_dates=['date'])
    ff.columns = ff.columns.str.lower()
    ff['date'] = ff['date'] + pd.offsets.MonthEnd(0)

    # Extract GA3 equal-weighted from combined file
    ga_col = f"{ga_choice}_ew"
    if ga_col not in ga.columns:
        print(f"❌ Column {ga_col} not found in {ga_path}")
        return
    ga = ga[['date', ga_col]].rename(columns={ga_col: 'ga_factor'})

    merged = pd.merge(ga, ff, on='date', how='inner')
    merged['ga_factor_excess'] = merged['ga_factor'] - merged['rf']
    merged = merged.dropna(subset=['ga_factor_excess'])

    factors = ['mkt_rf', 'smb', 'hml', 'rmw', 'cma', 'mom']
    coefs = []

    for i in range(ROLLING_WINDOW, len(merged)):
        window = merged.iloc[i - ROLLING_WINDOW:i]
        X = sm.add_constant(window[factors], has_constant='add')
        y = window['ga_factor_excess']
        model = sm.OLS(y, X).fit()
        row = model.params.to_dict()
        row['date'] = window['date'].iloc[-1]
        row['alpha_annualized'] = model.params['const'] * 12
        coefs.append(row)

    df_roll = pd.DataFrame(coefs)
    if df_roll.empty:
        print("⚠️ No rolling regression results computed.")
        return
    df_roll.set_index('date', inplace=True)

    # Plot rolling alpha
    plt.figure(figsize=(12, 6))
    df_roll['alpha_annualized'].plot(label="Rolling Alpha (Annualized)")
    plt.axhline(ALPHA_THRESHOLD, linestyle='--', color='green', label='2% Threshold')
    plt.axhline(0, color='black')
    plt.title(f"Rolling 60-Month Alpha – {ga_choice} (Equal-Weighted)")
    plt.ylabel("Alpha")
    plt.legend()
    plt.tight_layout()
    filepath = os.path.join(output_dir, f"{ga_choice}_rolling_alpha.png")
    plt.savefig(filepath)
    plt.close()
    print(f"✅ Saved rolling alpha plot: {filepath}")

    filepath = os.path.join(output_dir, f"{ga_choice}_rolling_regression_results.xlsx")
    df_roll.to_excel(filepath)
    print(f"✅ Saved rolling regression results: {filepath}")

########################################
### Main ###
########################################

def main():
    # Load regression results from File 4
    results = load_regression_results()
    
    # Generate plots and test significance
    plot_summary_tables(results)
    test_alpha_significance(results)
    
    # Run rolling regression for GA3 using ga_factors.csv
    ga_path = os.path.join("output", "factors", "ga_factors.csv")
    ff_path = os.path.join("data", "FamaFrench_factors_with_momentum.csv")
    rolling_regression(ga_path, ff_path, ga_choice="goodwill_to_market_cap_lagged")

if __name__ == "__main__":
    main()