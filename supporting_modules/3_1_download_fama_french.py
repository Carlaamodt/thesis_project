import pandas as pd
import wrds
import os

print("🔗 Connecting to WRDS...")
conn = wrds.Connection(wrds_username="carlaamodt")
print("✅ Connected successfully.")

print("⬇️ Downloading Fama-French 5-factor + Momentum data from WRDS...")
ff_factors = conn.raw_sql("""
SELECT 
    ff.date, 
    ff.mktrf AS mkt_rf, 
    ff.smb, 
    ff.hml, 
    ff.rmw, 
    ff.cma, 
    fm.umd AS mom,
    ff.rf
FROM ff.fivefactors_monthly AS ff
LEFT JOIN ff.factors_monthly AS fm
ON ff.date = fm.date
WHERE ff.date >= '01/01/2002'
""", date_cols=['date'])

print(f"✅ Data downloaded. Rows: {len(ff_factors)}")

# Filter out future years (e.g. 2024+)
ff_factors = ff_factors[ff_factors['date'].dt.year < 2024]

# Convert all factor columns to float safely
factor_cols = ['mkt_rf', 'smb', 'hml', 'rmw', 'cma', 'mom', 'rf']
for col in factor_cols:
    ff_factors[col] = pd.to_numeric(ff_factors[col], errors='coerce')

# Clip implausible values
print("🧹 Clipping outliers...")
ff_factors['rf'] = ff_factors['rf'].clip(upper=0.005)  # 0.5% monthly
for col in ['mkt_rf', 'smb', 'hml', 'rmw', 'cma', 'mom']:
    ff_factors[col] = ff_factors[col].clip(lower=-0.5, upper=0.5)

# Sort and cleanup
ff_factors.columns = ff_factors.columns.str.lower()
ff_factors = ff_factors.sort_values('date').reset_index(drop=True)

# Diagnostics
print("\n📊 Missing values:")
print(ff_factors[factor_cols].isnull().sum())

print(f"\n🗓️ Date range: {ff_factors['date'].min().date()} → {ff_factors['date'].max().date()}")
print("\n📈 Summary stats:")
print(ff_factors[factor_cols].describe())

# Save clean file
output_path = os.path.join("data", "FamaFrench_factors_with_momentum.csv")
os.makedirs("data", exist_ok=True)
ff_factors.to_csv(output_path, index=False)
print(f"\n💾 Saved cleaned factors to: {output_path}")