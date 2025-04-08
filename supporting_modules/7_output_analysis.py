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

def load_data(directory="data/", ga_choice="goodwill_to_sales_lagged"):
    print(f"📥 Loading data for {ga_choice}...")
    
    # Processed data from 2_data_processing.py
    proc_filepath = os.path.join(directory, "processed_data.csv")
    required_cols = [
        'permno', 'gvkey', 'crsp_date', 'FF_year', 'ret', 'prc', 'shrout', 'naicsh', 'sich', 'gdwl',
        'goodwill_to_sales_lagged', 'goodwill_to_equity_lagged', 'goodwill_to_market_cap_lagged'
    ]
    dtypes = {'naicsh': 'str', 'sich': 'str'}
    proc_df = pd.read_csv(proc_filepath, parse_dates=['crsp_date'], usecols=required_cols, 
                          dtype=dtypes, low_memory=False)
    proc_df = proc_df.drop_duplicates(subset=['permno', 'crsp_date'])
    print(f"Processed data rows: {len(proc_df)}, unique permno: {proc_df['permno'].nunique()}")

    # Decile data from 3_factor_construction.py
    decile_filepath = os.path.join(output_dir, f"decile_assignments_{ga_choice}_all.csv")
    if not os.path.exists(decile_filepath):
        print(f"❌ Decile file not found: {decile_filepath}. Will assign deciles from raw data.")
        decile_df = pd.DataFrame()
    else:
        decile_df = pd.read_csv(decile_filepath, parse_dates=['crsp_date'])
        decile_df = decile_df[decile_df['decile'].between(1, 10)]
        print(f"Decile data rows: {len(decile_df)}, unique permno: {decile_df['permno'].nunique()}")

    # FF factors from 3_1_download_fama_french.py
    ff_filepath = os.path.join(directory, "FamaFrench_factors_with_momentum.csv")
    ff_df = pd.read_csv(ff_filepath, parse_dates=['date'])
    ff_df['date'] = ff_df['date'] + pd.offsets.MonthEnd(0)
    ff_df.columns = ff_df.columns.str.lower()
    print(f"FF factors rows: {len(ff_df)}")

    # GA factors from 3_factor_construction.py
    ga_factors_path = os.path.join("output", "factors", "ga_factors.csv")
    if not os.path.exists(ga_factors_path):
        print(f"❌ GA factors file not found: {ga_factors_path}. Diagnostics may be limited.")
        ga_factors_df = pd.DataFrame()
    else:
        ga_factors_df = pd.read_csv(ga_factors_path, parse_dates=['date'])
        print(f"GA factors rows: {len(ga_factors_df)}")

    return proc_df, decile_df, ff_df, ga_factors_df

########################################
### Decile Analysis ###
########################################

def decile_analysis(proc_df, decile_df, ga_choice="goodwill_to_sales_lagged"):
    print(f"🔢 Analyzing {ga_choice} deciles...")
    
    if decile_df.empty:
        print(f"⚠️ No decile data—assigning from raw data.")
        df = proc_df.copy()
        if df[ga_choice].nunique() < 10:
            print(f"⚠️ Too few unique {ga_choice} values ({df[ga_choice].nunique()})—skipping.")
            return
        df['decile'] = pd.qcut(df[ga_choice], 10, labels=False, duplicates='drop') + 1
    else:
        df = decile_df.merge(proc_df[['permno', 'crsp_date', 'gdwl']], 
                             on=['permno', 'crsp_date'], how='left')
    
    df['has_goodwill_firm'] = (df['gdwl'] > 0).astype(int)
    decile_counts = df.groupby(['FF_year', 'decile', 'has_goodwill_firm']).agg(
        num_firms=('permno', 'nunique')
    ).reset_index()
    decile_pivot = decile_counts.pivot_table(
        index=['FF_year', 'decile'], columns='has_goodwill_firm', values='num_firms', fill_value=0
    ).reset_index()
    decile_pivot.rename(columns={0: 'firms_without_goodwill', 1: 'firms_with_goodwill'}, inplace=True)
    decile_pivot['total_firms'] = decile_pivot['firms_with_goodwill'] + decile_pivot['firms_without_goodwill']
    filepath = os.path.join(output_dir, f"{ga_choice}_firms_with_goodwill_per_decile.xlsx")
    decile_pivot.to_excel(filepath, index=False)
    print(f"✅ Saved decile counts: {filepath}")

    ga_stats = df.groupby(['FF_year', 'decile']).agg(
        avg_ga=(ga_choice, 'mean'),
        median_ga=(ga_choice, 'median')
    ).reset_index()
    filepath = os.path.join(output_dir, f"{ga_choice}_ga_stats_per_decile.xlsx")
    ga_stats.to_excel(filepath, index=False)
    print(f"✅ Saved GA stats: {filepath}")

    if 'ME' not in df.columns:
        df = df.merge(proc_df[['permno', 'crsp_date', 'prc', 'shrout']], 
                      on=['permno', 'crsp_date'], how='left')
        df['ME'] = np.abs(df['prc']) * df['shrout']
    me_stats = df.groupby(['FF_year', 'decile']).agg(
        avg_me=('ME', 'mean'),
        median_me=('ME', 'median')
    ).reset_index()
    filepath = os.path.join(output_dir, f"{ga_choice}_market_cap_per_decile.xlsx")
    me_stats.to_excel(filepath, index=False)
    print(f"✅ Saved market cap stats: {filepath}")

    combined_stats = pd.merge(decile_pivot, ga_stats, on=['FF_year', 'decile'], how='left')
    combined_stats = pd.merge(combined_stats, me_stats, on=['FF_year', 'decile'], how='left')
    filepath = os.path.join(output_dir, f"{ga_choice}_portfolio_characteristics.xlsx")
    combined_stats.to_excel(filepath, index=False)
    print(f"✅ Saved portfolio characteristics: {filepath}")

########################################
### GA Factor Diagnostics ###
########################################

def ga_factor_diagnostics(proc_df, ff_df, ga_factors_df, ga_choice="goodwill_to_sales_lagged"):
    print(f"📊 {ga_choice} factor diagnostics...")
    
    # GA trend over time
    ga_trend = proc_df.groupby(proc_df['crsp_date'].dt.year).agg(
        total_obs=(ga_choice, 'count'),
        ga_zero=(ga_choice, lambda x: (x == 0).sum()),
        ga_nonzero=(ga_choice, lambda x: (x != 0).sum()),
        mean_ga=(ga_choice, 'mean'),
        median_ga=(ga_choice, 'median')
    ).reset_index()
    filepath = os.path.join(output_dir, f"{ga_choice}_zero_nonzero_trend.xlsx")
    ga_trend.to_excel(filepath, index=False)
    print(f"✅ Saved GA trend: {filepath}")
    
    plt.figure(figsize=(10, 6))
    sns.lineplot(data=ga_trend, x='crsp_date', y='mean_ga', label=f'Mean {ga_choice}')
    sns.lineplot(data=ga_trend, x='crsp_date', y='median_ga', label=f'Median {ga_choice}')
    plt.legend()
    plt.title(f'Mean and Median {ga_choice} Over Time')
    filepath = os.path.join(output_dir, f"{ga_choice}_mean_median_trend.png")
    plt.savefig(filepath)
    plt.close()
    print(f"✅ Saved GA trend plot: {filepath}")

    if ga_factors_df.empty:
        print(f"⚠️ No GA factors data—skipping factor-specific diagnostics.")
        return
    
    ga_eq = ga_factors_df[['date', f'{ga_choice}_ew']].rename(columns={f'{ga_choice}_ew': 'ga_factor'})
    ga_val = ga_factors_df[['date', f'{ga_choice}_vw']].rename(columns={f'{ga_choice}_vw': 'ga_factor'})

    # Cumulative returns
    ga_eq['cum_return'] = (1 + ga_eq['ga_factor']).cumprod()
    ga_val['cum_return'] = (1 + ga_val['ga_factor']).cumprod()
    
    merged = pd.merge(ga_eq[['date', 'ga_factor']], ff_df, on='date', how='inner')
    merged['mkt_cum_return'] = (1 + merged['mkt_rf']).cumprod() if 'mkt_rf' in merged.columns else np.nan
    
    plt.figure(figsize=(10, 6))
    plt.plot(ga_eq['date'], ga_eq['cum_return'], label='Equal-weighted GA (Low - High)')
    plt.plot(ga_val['date'], ga_val['cum_return'], label='Value-weighted GA (Low - High)')
    if merged['mkt_cum_return'].notna().any():
        plt.plot(merged['date'], merged['mkt_cum_return'], label='Market', linestyle='--', color='green')
    plt.title(f'{ga_choice} Cumulative Returns vs. Market (Low GA - High GA)')
    plt.xlabel('Date')
    plt.ylabel('Cumulative Return')
    plt.legend()
    plt.grid(True, alpha=0.7)
    filepath = os.path.join(output_dir, f"{ga_choice}_cumulative_returns.png")
    plt.savefig(filepath)
    plt.close()
    print(f"✅ Saved cumulative returns plot: {filepath}")

    # Correlation with FF factors
    ff_cols = ['mkt_rf', 'smb', 'hml', 'rmw', 'cma', 'mom']
    available_ff_cols = [col for col in ff_cols if col in merged.columns]
    corr_matrix = merged[['ga_factor'] + available_ff_cols].corr()
    filepath = os.path.join(output_dir, f"{ga_choice}_correlation_matrix.xlsx")
    corr_matrix.to_excel(filepath)
    print(f"✅ Saved correlation matrix: {filepath}")

    plt.figure(figsize=(8, 6))
    sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', center=0, fmt='.2f')
    plt.title(f"{ga_choice} Correlation with FF Factors (Low GA - High GA)")
    filepath = os.path.join(output_dir, f"{ga_choice}_correlation_heatmap.png")
    plt.savefig(filepath)
    plt.close()
    print(f"✅ Saved correlation heatmap: {filepath}")

    # Factor summary
    ga_summary = pd.DataFrame({
        'Equal Mean': [ga_eq['ga_factor'].mean()],
        'Equal Std': [ga_eq['ga_factor'].std()],
        'Value Mean': [ga_val['ga_factor'].mean()],
        'Value Std': [ga_val['ga_factor'].std()]
    })
    filepath = os.path.join(output_dir, f"{ga_choice}_factor_summary.xlsx")
    ga_summary.to_excel(filepath, index=False)
    print(f"✅ Saved factor summary: {filepath}")

    ########################################
### Market Return Plot + Decile Alphas ###
########################################

def market_return_reference_plot(ff_df, output_dir="analysis_output"):
    if 'mkt_rf' not in ff_df.columns or 'rf' not in ff_df.columns:
        print("⚠️ Missing market or risk-free columns in FF data.")
        return

    ff_df = ff_df.copy()
    ff_df['mkt'] = ff_df['mkt_rf'] + ff_df['rf']  # reconstruct actual return
    ff_df['mkt_cum_return'] = (1 + ff_df['mkt']).cumprod()

    plt.figure(figsize=(10, 6))
    plt.plot(ff_df['date'], ff_df['mkt_cum_return'], label='Market Return (Actual)', linestyle='-')
    plt.title("📈 Market Cumulative Return (Not Risk-Adjusted)")
    plt.xlabel("Date")
    plt.ylabel("Cumulative Return")
    plt.grid(True, alpha=0.6)
    plt.legend()
    filepath = os.path.join(output_dir, "market_actual_cumulative_return.png")
    plt.savefig(filepath)
    plt.close()
    print(f"✅ Saved market cumulative return plot: {filepath}")

def decile_alpha_regressions(proc_df, decile_df, ff_df, ga_choice="goodwill_to_sales_lagged"):
    print(f"📊 Running alpha regressions for each decile...")

    results = []

    for weighting in ["equal", "value"]:
        for decile in range(1, 11):
            sub_df = decile_df[decile_df['decile'] == decile].copy()
            sub_df = sub_df.merge(proc_df[['permno', 'crsp_date', 'ret', 'prc', 'shrout']], 
                                  on=['permno', 'crsp_date'], how='left')
            sub_df['ME'] = np.abs(sub_df['prc']) * sub_df['shrout']
            sub_df = sub_df[sub_df['ret'].notna() & sub_df['ME'].notna()]
            
            if sub_df.empty:
                print(f"⚠️ Skipping decile {decile} ({weighting}) — no data for regression.")
                continue

            if weighting == "equal":
                ret_series = sub_df.groupby('crsp_date')['ret'].mean()
            else:
                ret_series = sub_df.groupby('crsp_date').apply(
                    lambda x: np.average(x['ret'], weights=x['ME']) if len(x) >= 5 else np.nan
                )

            # Drop NaNs and reset index
            ret_series.name = 'ret'
            ret_df = ret_series.dropna().reset_index()

            # Rename index to crsp_date if needed
            if 'index' in ret_df.columns:
                ret_df.rename(columns={'index': 'crsp_date'}, inplace=True)

            # Check for crsp_date existence
            if 'crsp_date' not in ret_df.columns:
                print(f"❌ Failed: crsp_date column not found in return data for decile {decile} ({weighting})")
                continue

            # Merge with FF factors
            merged = pd.merge(ret_df, ff_df, left_on='crsp_date', right_on='date', how='inner')
            if 'rf' not in merged.columns:
                print("❌ Missing risk-free rate in FF data.")
                continue

            merged['ga_factor_excess'] = merged['ret'] - merged['rf']

            X = sm.add_constant(merged[['mkt_rf']], has_constant='add')
            y = merged['ga_factor_excess']
            model = sm.OLS(y, X, missing='drop').fit(cov_type='HAC', cov_kwds={'maxlags': 3})

            results.append({
                'Decile': decile,
                'Weighting': weighting,
                'Alpha (annualized)': model.params.get('const', np.nan) * 12,
                'Alpha p-value': model.pvalues.get('const', np.nan),
                'R-squared': model.rsquared
            })

    if not results:
        print(f"⚠️ No decile alpha regressions completed for {ga_choice}")
        return

    results_df = pd.DataFrame(results)
    filepath = os.path.join(output_dir, f"decile_alpha_regressions_{ga_choice}.xlsx")
    results_df.to_excel(filepath, index=False)
    print(f"✅ Saved decile alpha regressions: {filepath}")



########################################
### Industry Analysis ###
########################################

def industry_analysis(proc_df, decile_df, ff_df, ga_choice="goodwill_to_sales_lagged", industries=None):
    print(f"🏭 {ga_choice} industry analysis...")
    
    if decile_df.empty:
        print(f"⚠️ No decile data—assigning from raw data.")
        df = proc_df.copy()
        if df[ga_choice].nunique() < 10:
            print(f"⚠️ Too few unique {ga_choice} values ({df[ga_choice].nunique()})—skipping.")
            return
        df['decile'] = pd.qcut(df[ga_choice], 10, labels=False, duplicates='drop') + 1
    else:
        df = proc_df.merge(decile_df[['permno', 'crsp_date', 'decile']], 
                           on=['permno', 'crsp_date'], how='inner')

    # Industry classification
    df['naics_2digit'] = df['naicsh'].astype(str).str[:2].fillna('00')
    industry_map = {
        '31': 'Manufacturing', '32': 'Manufacturing', '33': 'Manufacturing',
        '51': 'Information', '52': 'Finance', '62': 'Healthcare', '00': 'Unknown'
    }
    df['industry'] = df['naics_2digit'].map(lambda x: industry_map.get(x, 'Other'))
    
    if industries is None:
        industries = df['industry'].unique()
    print(f"Industries: {list(industries)}")

    # Compute ME
    df['ME'] = np.abs(df['prc']) * df['shrout']
    df = df[df['ret'].notna() & (df['ME'] > 0)].copy()
    print(f"Data for factor computation: {len(df)} rows")

    industry_factors = {}
    for weighting in ["equal", "value"]:
        factor_returns = []
        for industry in industries:
            ind_df = df[df['industry'] == industry].copy()
            if len(ind_df) < 20:
                print(f"⚠️ Skipping {industry}: insufficient data ({len(ind_df)} rows)")
                continue
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
                print(f"⚠️ Skipping {industry}: missing decile 1 or 10 in returns")
                continue
            
            factor['ga_factor'] = factor[1] - factor[10]  # Low-minus-high (File 3)
            factor = factor[['ga_factor']].dropna().reset_index()
            factor['industry'] = industry
            factor_returns.append(factor)
        
        if not factor_returns:
            print(f"⚠️ No {weighting}-weighted factors computed for {ga_choice}")
            continue
        
        combined = pd.concat(factor_returns, axis=0)
        combined.rename(columns={'crsp_date': 'date'}, inplace=True)
        os.makedirs(os.path.join('output', 'industry_factors'), exist_ok=True)
        filepath = os.path.join('output', 'industry_factors', 
                                f"ga_factor_returns_{weighting}_{ga_choice}_by_industry.csv")
        combined.to_csv(filepath, index=False)
        print(f"✅ Saved {weighting}-weighted factors: {filepath}")
        industry_factors[weighting] = combined

    # Regression analysis
    factor_models = {"CAPM": ["mkt_rf"], "FF5": ["mkt_rf", "smb", 'hml', "rmw", "cma"]}
    all_results = []
    for weighting, factors in industry_factors.items():
        for industry in factors['industry'].unique():
            ind_df = factors[factors['industry'] == industry].merge(ff_df, on='date', how='inner')
            if len(ind_df) < 12:
                print(f"⚠️ Skipping {industry} ({weighting}): too few months ({len(ind_df)})")
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
    filepath = os.path.join(output_dir, f"industry_regression_results_{ga_choice}.xlsx")
    results_df.to_excel(filepath, index=False)
    print(f"✅ Saved industry regression results: {filepath}")

########################################
### Main Execution ###
########################################

def main(ga_choice="goodwill_to_sales_lagged", industries=None):
    try:
        proc_df, decile_df, ff_df, ga_factors_df = load_data(ga_choice=ga_choice)
        decile_analysis(proc_df, decile_df, ga_choice=ga_choice)
        ga_factor_diagnostics(proc_df, ff_df, ga_factors_df, ga_choice=ga_choice)
        industry_analysis(proc_df, decile_df, ff_df, ga_choice=ga_choice, industries=industries)
        market_return_reference_plot(ff_df)  # NEW
        decile_alpha_regressions(proc_df, decile_df, ff_df, ga_choice=ga_choice)  # NEW
        print(f"🎉 All {ga_choice} outputs generated!")
    except Exception as e:
        print(f"❌ Failed: {e}")
        raise


if __name__ == "__main__":
    main(ga_choice="goodwill_to_sales_lagged")