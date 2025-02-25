import pandas as pd
import wrds

# Connect to WRDS
conn = wrds.Connection(wrds_username="carlaamodt")

# Download Fama-French 5-Factor Data
ff_factors = conn.raw_sql("""
    SELECT date, mktrf AS MKT, smb, hml, rmw, cma, rf
    FROM ff.fivefactors_monthly
""", date_cols=['date'])

# Convert date to quarter-end format
ff_factors['quarter'] = ff_factors['date'] + pd.offsets.QuarterEnd(0)

# Drop original date column
ff_factors = ff_factors.drop(columns=['date'])

# Save to CSV
ff_factors.to_csv("data/FamaFrench_factors.csv", index=False)

print("✅ Fama-French factors saved successfully!")
