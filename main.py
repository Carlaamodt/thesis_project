import pandas as pd
import os
import glob

# Set directory where data is stored
data_dir = os.path.join(os.getcwd(), "data")

# Function to get the latest file for a given prefix
def get_latest_file(prefix):
    files = glob.glob(os.path.join(data_dir, f"{prefix}_*.csv"))  # Find all matching files
    if not files:
        raise FileNotFoundError(f"No files found for prefix: {prefix}")
    latest_file = max(files, key=os.path.getmtime)  # Get the most recent file
    return latest_file

# Load the most recent files dynamically
compustat = pd.read_csv(get_latest_file("compustat"))
crsp_ret = pd.read_csv(get_latest_file("crsp_ret"))
crsp_delist = pd.read_csv(get_latest_file("crsp_delist"))
crsp_compustat = pd.read_csv(get_latest_file("crsp_compustat"))

print("✅ Loaded the most recent datasets into main.py")


#Functions error
