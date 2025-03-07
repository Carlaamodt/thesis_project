import pandas as pd
import wrds

# Connect to WRDS
conn = wrds.Connection(wrds_username="carlaamodt")

# Download Fama-French 5-Factor Data with Momentum starting from January 2002
ff_factors = conn.raw_sql("""
    SELECT ff.date, 
           ff.mktrf AS MKT, 
           ff.smb, 
           ff.hml, 
           ff.rmw, 
           ff.cma, 
           mom.umd AS MOM,
           ff.rf
    FROM ff.fivefactors_monthly AS ff
    LEFT JOIN ff.momentum_monthly AS mom
    ON ff.date = mom.date
    WHERE ff.date >= '2002-01-01'
""", date_cols=['date'])

# Convert date to quarter-end format
ff_factors['quarter'] = ff_factors['date'] + pd.offsets.QuarterEnd(0)

# Drop original date column
ff_factors = ff_factors.drop(columns=['date'])

# Save to CSV
ff_factors.to_csv("data/FamaFrench_factors_with_momentum.csv", index=False)

print("✅ Fama-French factors (including Momentum) saved successfully!")
