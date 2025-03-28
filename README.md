# Thesis
## Python Files Overview

1) main.py: Orchestrate full pipeline by running extraction, processing, factor construction, and testing in sequence.
2) 0_Datanalysis.py: Perform preliminary data analysis by loading raw data and generating summary statistics and visualizations.
3) 1_data_extraction.py: Extract raw data by pulling Compustat (goodwill: gdwl), CRSP, and link table, saving as CSV files.
4) 2_data_processing.py: Prepare cleaned dataset by merging Compustat-CRSP, computing goodwill intensity (GA), and adjusting returns for delistings.
5) 3_1_download_fama_french.py: Fetch external factors by downloading Fama-French 5 factors plus Momentum and saving as CSV.
6) 3_factor_construction.py: Create portfolios by sorting firms into GA quintiles, computing returns, and constructing the GA factor (Q5-Q1).
7) 4_factor_model.py: Test factor performance by regressing GA factor on CAPM, FF3, FF5, and FF5+Momentum+GA, outputting results.
8) 5_output_analysis.py: Analyze and visualize results by loading factor returns, plotting trends, and saving summaries.

After making changes to your code in VS Code, follow these 3 steps:
1) Stage the changes (prepare them for commit): git add .
2) Commit the changes (save a snapshot in Git): git commit -m "Changes"
3) Push the changes to GitHub (upload them): git push origin main
Code is now in GitHub.

Download Code from GitHub to VS Code (2 Steps)
1) Go to your project folder: cd ~/thesis_project
2) Pull the latest code from GitHub: git pull origin main
Now your local code is up to date.