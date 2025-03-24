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

def load_regression_results(file_path="output/ga_factor_regression_results_monthly.xlsx"):
    print("📥 Loading regression results...")
    xl = pd.read_excel(file_path, sheet_name=None)
    return xl

########################################
### Visual Summary Plots ###
########################################

def plot_summary_tables(results_dict):
    for ga_key, df in results_dict.items():
        if df is None or df.empty:
            continue

        # Alpha by model
        plt.figure(figsize=(10, 6))
        sns.barplot(data=df, x="Model", y="Alpha (annualized)", hue="Weighting")
        plt.axhline(ALPHA_THRESHOLD, linestyle="--", color="green", label="2% Threshold")
        plt.axhline(0, color="black")
        plt.title(f"Annualized Alpha by Model – {ga_key}")
        plt.legend()
        plt.tight_layout()
        plt.savefig(f"{output_dir}/{ga_key}_alpha_by_model.png")
        plt.close()

        # R-squared by model
        plt.figure(figsize=(10, 6))
        sns.barplot(data=df, x="Model", y="R-squared", hue="Weighting")
        plt.title(f"R-squared by Model – {ga_key}")
        plt.axhline(0, color="black")
        plt.tight_layout()
        plt.savefig(f"{output_dir}/{ga_key}_rsquared_by_model.png")
        plt.close()

########################################
### Hypothesis Tests ###
########################################

def test_alpha_significance(results_dict):
    summary = []
    for ga_key, df in results_dict.items():
        if df is None or df.empty:
            continue
        df_filtered = df[(df['Alpha p-value (HAC)'] < SIGNIFICANCE_LEVEL) & (df['Alpha (annualized)'].abs() >= ALPHA_THRESHOLD)]
        for _, row in df_filtered.iterrows():
            summary.append({
                "GA": ga_key,
                "Model": row["Model"],
                "Weighting": row["Weighting"],
                "Alpha": row["Alpha (annualized)"],
                "p-value": row["Alpha p-value (HAC)"]
            })
    result_df = pd.DataFrame(summary)
    result_df.to_excel(f"{output_dir}/significant_alphas.xlsx", index=False)
    print("✅ Significant alphas saved.")

########################################
### Rolling Regressions for GA3 ###
########################################

def rolling_regression(ga_path, ff_path):
    print("🔁 Running rolling regression for GA3...")
    ga = pd.read_csv(ga_path, parse_dates=['date'])
    ff = pd.read_csv(ff_path, parse_dates=['date'])
    ff.columns = ff.columns.str.lower()
    ff['date'] = ff['date'] + pd.offsets.MonthEnd(0)

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
    df_roll.set_index('date', inplace=True)

    # Plot rolling alpha
    plt.figure(figsize=(12, 6))
    df_roll['alpha_annualized'].plot(label="Rolling Alpha (Annualized)")
    plt.axhline(ALPHA_THRESHOLD, linestyle='--', color='green', label='2% Threshold')
    plt.axhline(0, color='black')
    plt.title("Rolling 60-Month Alpha – GA3 (Equal-Weighted)")
    plt.ylabel("Alpha")
    plt.legend()
    plt.tight_layout()
    plt.savefig(f"{output_dir}/GA3_rolling_alpha.png")
    plt.close()

    df_roll.to_excel(f"{output_dir}/GA3_rolling_regression_results.xlsx")
    print("✅ Rolling regression results saved.")

########################################
### Main ###
########################################

def main():
    results = load_regression_results()
    plot_summary_tables(results)
    test_alpha_significance(results)
    
    # Run rolling regression for GA3 equal-weighted only
    ga3_path = "output/factors/ga_factor_returns_monthly_equal_GA3_lagged.csv"
    ff_path = "data/FamaFrench_factors_with_momentum.csv"
    rolling_regression(ga3_path, ff_path)

if __name__ == "__main__":
    main()