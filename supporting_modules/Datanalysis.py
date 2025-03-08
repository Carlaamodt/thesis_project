import pandas as pd
import numpy
import matplotlib.pyplot as plt
import seaborn as sns
import os
df = pd.read_csv("data/processed_data.csv", parse_dates=['date'])
print(df.dtypes)

df = pd.read_csv("data/processed_data.csv", parse_dates=['date'])
print(df['goodwill_intensity'].describe())
print("Missing values:", df['goodwill_intensity'].isna().sum())
zero_count = df[df['goodwill_intensity'] == 0].shape[0]
print("Number of rows with goodwill_intensity = 0:", zero_count)
unique_companies = df['gvkey'].nunique()
print("Number of unique companies:", unique_companies)