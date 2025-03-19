import pandas as pd
import wrds
import os

############################
### Connect to WRDS ###
############################
print("🔗 Connecting to WRDS...")
conn = wrds.Connection(wrds_username="carlaamodt")
print("✅ Connected successfully.")

############################
### Fetch Available Tables ###
############################
print("📚 Available 'ff' tables on WRDS:")
available_tables = conn.list_tables('ff')
print(available_tables)

##############################################
### Download Fama-French 5-Factor + Momentum ###
##############################################

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

print(f"✅ Data downloaded. Total rows: {ff_factors.shape[0]}")

#######################################
### Filter, Clean, and Save Dataset ###
#######################################

# ✅ Exclude data from 2024 and beyond
ff_factors = ff_factors[ff_factors['date'].dt.year < 2024]

# Ensure lowercase column names
ff_factors.columns = ff_factors.columns.str.lower()

# Sort by date
ff_factors = ff_factors.sort_values('date').reset_index(drop=True)

# Check missing values
print("\n🧹 Checking for missing values:")
print(ff_factors.isnull().sum())

# Final check on date range
print(f"\n🗓️ Final date range: {ff_factors['date'].min()} to {ff_factors['date'].max()}")

# Save cleaned file
output_dir = "data"
os.makedirs(output_dir, exist_ok=True)
output_path = os.path.join(output_dir, "FamaFrench_factors_with_momentum.csv")
ff_factors.to_csv(output_path, index=False)
print(f"\n💾 Saved to: {output_path}")