import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import statsmodels.api as sm
import os

# Set output directory
output_dir = "analysis_output"
os.makedirs(output_dir, exist_ok=True)

########################################
### Load Data ###
########################################

def load_data(directory="data/", ga_choice="GA1_lagged"):
    print(f"📥 Loading data for {ga_choice}...")
    # Processed data
    proc_filepath = os.path.join(directory, "processed_data.csv")
    required_cols = ['permno', 'gvkey', 'crsp_date', 'FF_year', 'GA1_lagged', 'GA2_lagged', 'GA3_lagged', 
                     'ret', 'prc', 'csho', 'naicsh', 'sich', 'gdwl']
    dtypes = {'naicsh': 'str', 'sich': 'str'}
    proc_df = pd.read_csv(proc_filepath, parse_dates=['crsp_date'], usecols=required_cols, 
                          dtype=dtypes, low_memory=False)
    proc_df = proc_df.drop_duplicates(subset=['permno', 'crsp_date'])
    print(f"Processed data rows: {len(proc_df)}, unique permno: {proc_df['permno'].nunique()}")

    # Decile data
    decile_filepath = f"analysis_output/decile_assignments_{ga_choice}.csv"
    if not os.path.exists(decile_filepath):
        print(f"❌ Decile file not found: {decile_filepath}")
        decile_df = pd.DataFrame()
    else:
        decile_df = pd.read_csv(decile_filepath, parse_dates=['crsp_date'])
        decile_df = decile_df[decile_df['decile'].between(1, 10)]
    print(f"Decile data rows: {len(decile_df)}, unique permno: {decile_df['permno'].nunique()}")

    # FF factors
    ff_filepath = os.path.join(directory, "FamaFrench_factors_with_momentum.csv")
    ff_df = pd.read_csv(ff_filepath, parse_dates=['date'])
    ff_df['date'] = ff_df['date'] + pd.offsets.MonthEnd(0)
    ff_df.columns = ff_df.columns.str.lower()
    print(f"FF factors rows: {len(ff_df)}")

    return proc_df, decile_df, ff_df

########################################
### Decile Analysis ###
########################################

def decile_analysis(proc_df, decile_df, ga_choice="GA1_lagged"):
    print(f"🔢 Analyzing {ga_choice} deciles...")
    
    if decile_df.empty:
        print(f"⚠️ No decile data for {ga_choice}—skipping.")
        return
    
    decile_df = decile_df.merge(proc_df[['permno', 'crsp_date', 'gdwl']], 
                               on=['permno', 'crsp_date'], how='left')
    
    decile_df['has_goodwill_firm'] = (decile_df['gdwl'] > 0).astype(int)
    decile_counts = decile_df.groupby(['FF_year', 'decile', 'has_goodwill_firm']).agg(
        num_firms=('permno', 'nunique')
    ).reset_index()
    decile_pivot = decile_counts.pivot_table(
        index=['FF_year', 'decile'], columns='has_goodwill_firm', values='num_firms', fill_value=0
    ).reset_index()
    decile_pivot.rename(columns={0: 'firms_without_goodwill', 1: 'firms_with_goodwill'}, inplace=True)
    decile_pivot['total_firms'] = decile_pivot['firms_with_goodwill'] + decile_pivot['firms_without_goodwill']
    decile_pivot.to_excel(f"{output_dir}/{ga_choice}_firms_with_goodwill_per_decile.xlsx", index=False)

    ga_stats = decile_df.groupby(['FF_year', 'decile']).agg(
        avg_ga=(ga_choice, 'mean'),
        median_ga=(ga_choice, 'median')
    ).reset_index()
    ga_stats.to_excel(f"{output_dir}/{ga_choice}_ga_stats_per_decile.xlsx", index=False)

    if 'ME' in decile_df.columns:
        me_stats = decile_df.groupby(['FF_year', 'decile']).agg(
            avg_me=('ME', 'mean'),
            median_me=('ME', 'median')
        ).reset_index()
        me_stats.to_excel(f"{output_dir}/{ga_choice}_market_cap_per_decile.xlsx", index=False)
    else:
        print(f"⚠️ 'ME' not in decile data—skipping.")
        me_stats = None

    combined_stats = pd.merge(decile_pivot, ga_stats, on=['FF_year', 'decile'], how='left')
    if me_stats is not None:
        combined_stats = pd.merge(combined_stats, me_stats, on=['FF_year', 'decile'], how='left')
    combined_stats.to_excel(f"{output_dir}/{ga_choice}_portfolio_characteristics.xlsx", index=False)

########################################
### GA Factor Diagnostics ###
########################################

def ga_factor_diagnostics(proc_df, ff_df, ga_choice="GA1_lagged"):
    print(f"📊 {ga_choice} factor diagnostics...")
    
    ga_trend = proc_df.groupby(proc_df['crsp_date'].dt.year).agg(
        total_obs=(ga_choice, 'count'),
        ga_zero=(ga_choice, lambda x: (x == 0).sum()),
        ga_nonzero=(ga_choice, lambda x: (x != 0).sum()),
        mean_ga=(ga_choice, 'mean'),
        median_ga=(ga_choice, 'median')
    ).reset_index()
    ga_trend.to_excel(f"{output_dir}/{ga_choice}_zero_nonzero_trend.xlsx", index=False)
    
    plt.figure(figsize=(10, 6))
    sns.lineplot(data=ga_trend, x='crsp_date', y='mean_ga', label=f'Mean {ga_choice}')
    sns.lineplot(data=ga_trend, x='crsp_date', y='median_ga', label=f'Mean {ga_choice}')
    plt.legend()
    plt.title(f'Mean and Median {ga_choice} Over Time')
    plt.savefig(f"{output_dir}/{ga_choice}_mean_median_trend.png")
    plt.close()

    ga_eq_path = f"output/factors/ga_factor_returns_monthly_equal_{ga_choice}.csv"
    ga_val_path = f"output/factors/ga_factor_returns_monthly_value_{ga_choice}.csv"
    
    if not os.path.exists(ga_eq_path) or not os.path.exists(ga_val_path):
        print(f"⚠️ Missing factor return files for {ga_choice}—skipping diagnostics.")
        return
    
    ga_eq = pd.read_csv(ga_eq_path, parse_dates=['date'])
    ga_val = pd.read_csv(ga_val_path, parse_dates=['date'])
    
    ga_eq['cum_return'] = (1 + ga_eq['ga_factor']).cumprod()
    ga_val['cum_return'] = (1 + ga_val['ga_factor']).cumprod()
    
    merged = pd.merge(ga_eq[['date', 'ga_factor']], ff_df, on='date', how='inner')
    merged['mkt_cum_return'] = (1 + merged['mkt_rf']).cumprod() if 'mkt_rf' in merged.columns else np.nan
    
    plt.figure(figsize=(10, 6))
    plt.plot(ga_eq['date'], ga_eq['cum_return'], label='Equal-weighted GA')
    plt.plot(ga_val['date'], ga_val['cum_return'], label='Value-weighted GA')
    if merged['mkt_cum_return'].notna().any():
        plt.plot(merged['date'], merged['mkt_cum_return'], label='Market', linestyle='--', color='green')
    plt.title(f'{ga_choice} Cumulative Returns vs. Market')
    plt.xlabel('Date')
    plt.ylabel('Cumulative Return')
    plt.legend()
    plt.grid(True, alpha=0.7)
    plt.savefig(f"{output_dir}/{ga_choice}_cumulative_returns.png")
    plt.close()

    ff_cols = ['mkt_rf', 'smb', 'hml', 'rmw', 'cma', 'mom']
    available_ff_cols = [col for col in ff_cols if col in merged.columns]
    corr_matrix = merged[['ga_factor'] + available_ff_cols].corr()
    corr_matrix.to_excel(f"{output_dir}/{ga_choice}_correlation_matrix.xlsx")
    
    plt.figure(figsize=(8, 6))
    sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', center=0, fmt='.2f')
    plt.title(f"{ga_choice} Correlation with FF Factors")
    plt.savefig(f"{output_dir}/{ga_choice}_correlation_heatmap.png")
    plt.close()

    ga_summary = pd.DataFrame({
        'Equal Mean': [ga_eq['ga_factor'].mean()],
        'Equal Std': [ga_eq['ga_factor'].std()],
        'Value Mean': [ga_val['ga_factor'].mean()],
        'Value Std': [ga_val['ga_factor'].std()]
    })
    ga_summary.to_excel(f"{output_dir}/{ga_choice}_factor_summary.xlsx", index=False)

########################################
### Industry Analysis ###
########################################

def industry_analysis(proc_df, ff_df, ga_choice="GA1_lagged", industries=None):
    print(f"🏭 {ga_choice} industry analysis...")
    
    proc_df['naics_2digit'] = proc_df['naicsh'].astype(str).str[:2].fillna('00')
    industry_map = {
        '31': 'Manufacturing', '32': 'Manufacturing', '33': 'Manufacturing',
        '51': 'Information', '52': 'Finance', '62': 'Healthcare', '00': 'Unknown'
    }
    proc_df['industry'] = proc_df['naics_2digit'].map(lambda x: industry_map.get(x, 'Other'))
    
    if industries is None:
        industries = proc_df['industry'].unique()
    print(f"Industries: {list(industries)}")

    proc_df['ME'] = np.abs(proc_df['prc']) * proc_df['csho']
    df = proc_df[proc_df['ret'].notna() & (proc_df['ME'] > 0)].copy()
    
    if df[ga_choice].nunique() < 10:
        print(f"⚠️ Too few unique {ga_choice} values ({df[ga_choice].nunique()}) for 10 deciles across all data.")
        return
    df['decile'] = pd.qcut(df[ga_choice], 10, labels=False, duplicates='drop') + 1
    
    industry_factors = {}
    for weighting in ["equal", "value"]:
        factor_returns = []
        for industry in industries:
            ind_df = df[df['industry'] == industry].copy()
            if len(ind_df) < 20:
                print(f"⚠️ Skipping {industry}: insufficient data ({len(ind_df)} rows)")
                continue
            print(f"Industry: {industry}, Unique Deciles: {ind_df['decile'].unique()}")  # Debug
            
            ind_df = ind_df[ind_df['decile'].isin([1, 10])].copy()
            if len(ind_df['decile'].unique()) < 2:
                print(f"⚠️ Skipping {industry}: missing deciles 1 or 10 ({ind_df['decile'].unique()})")
                continue
            
            if weighting == "equal":
                factor = ind_df.groupby(['crsp_date', 'decile'])['ret'].mean().unstack()
            else:
                factor = ind_df.groupby(['crsp_date', 'decile']).apply(
                    lambda x: np.average(x['ret'], weights=x['ME']) if len(x) >= 5 else np.nan,
                    include_groups=False
                ).unstack()
            
            if 10 not in factor.columns or 1 not in factor.columns:
                print(f"⚠️ Missing decile columns in industry {industry} ({weighting}). Skipping.")
                continue
            
            factor['ga_factor'] = factor[10] - factor[1]
            factor = factor[['ga_factor']].dropna().reset_index()
            factor['industry'] = industry
            factor_returns.append(factor)
        
        if not factor_returns:
            print(f"⚠️ No valid factor returns for {weighting}-weighted {ga_choice}")
            continue
        
        combined = pd.concat(factor_returns, axis=0)
        combined.rename(columns={'crsp_date': 'date'}, inplace=True)
        os.makedirs('output/industry_factors', exist_ok=True)
        combined.to_csv(f"output/industry_factors/ga_factor_returns_{weighting}_{ga_choice}_by_industry.csv", index=False)
        industry_factors[weighting] = combined

    factor_models = {"CAPM": ["mkt_rf"], "FF5": ["mkt_rf", "smb", "hml", "rmw", "cma"]}
    all_results = []
    for weighting, factors in industry_factors.items():
        for industry in factors['industry'].unique():
            ind_df = factors[factors['industry'] == industry].merge(ff_df, on='date', how='inner')
            if len(ind_df) < 12:
                print(f"⚠️ Skipping {industry} ({weighting}): too few obs")
                continue
            ind_df['ga_factor_excess'] = ind_df['ga_factor'] - ind_df['rf']
            
            for model_name, factor_list in factor_models.items():
                X = sm.add_constant(ind_df[factor_list], has_constant='add')
                y = ind_df['ga_factor_excess']
                model = sm.OLS(y, X, missing='drop').fit(cov_type='HAC', cov_kwds={'maxlags': 3})
                all_results.append({
                    'Industry': industry, 'Weighting': weighting, 'Model': model_name,
                    'Alpha (annualized)': model.params.get('const', np.nan) * 12,
                    'Alpha p-value': model.pvalues.get('const', np.nan),
                    'R-squared': model.rsquared
                })
    
    if not all_results:
        print(f"⚠️ No regression results for {ga_choice}")
        return
    
    results_df = pd.DataFrame(all_results)
    results_df.to_excel(f"output/industry_regression_results_{ga_choice}.xlsx", index=False)
    print(f"✅ Industry analysis saved for {ga_choice}")

########################################
### Main Execution ###
########################################

def main(ga_choice="GA1_lagged", industries=None):
    try:
        proc_df, decile_df, ff_df = load_data(ga_choice=ga_choice)
        decile_analysis(proc_df, decile_df, ga_choice=ga_choice)
        ga_factor_diagnostics(proc_df, ff_df, ga_choice=ga_choice)
        industry_analysis(proc_df, ff_df, ga_choice=ga_choice, industries=industries)
        print(f"🎉 All {ga_choice} outputs generated!")
    except Exception as e:
        print(f"❌ Failed: {e}")
        raise

if __name__ == "__main__":
    main(ga_choice="GA2_lagged")  # Switch to "GA2_lagged" or "GA3_lagged"