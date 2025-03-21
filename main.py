import subprocess
import argparse
import logging
import sys
import os

# ------------------- Logging Setup --------------------
logging.basicConfig(
    level=logging.INFO,  # Change to DEBUG for more detailed logs if needed
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("pipeline_log.txt"),  # Save logs to file
        logging.StreamHandler(sys.stdout)  # Also print to terminal
    ]
)

# ------------------- Script List ----------------------
scripts = [
    "supporting_modules/1_data_extraction.py",          # Extract raw data from WRDS
    "supporting_modules/2_data_processing.py",         # Process and clean raw data
    "supporting_modules/3_1_download_fama_french.py",  # Download FF + Momentum data
    "supporting_modules/3_factor_construction.py",     # Construct GA factor (Equal/Value weighted)
    "supporting_modules/4_factor_model.py",            # Run factor model regressions (GA, MKT, SMB, HML etc.)
    "supporting_modules/0_Datanalysis.py",             # Data diagnostic & exploratory analysis
    "supporting_modules/5_output_analysis.py"          # Thesis tables, figures, portfolio characteristics
]

# ------------------- Function to Run Each Script ----------------------
def run_script(script_path):
    """Run a single Python script and handle output/errors."""
    logging.info(f"🚀 Running: {script_path}")
    try:
        result = subprocess.run(
            ["python", script_path], 
            capture_output=True, 
            text=True,
            check=True  # This will raise CalledProcessError if script fails
        )
        logging.info(f"✅ Completed: {script_path}")
        if result.stdout:
            logging.debug(f"Output from {script_path}:\n{result.stdout}")
    except subprocess.CalledProcessError as e:
        logging.error(f"❌ Error in {script_path}:\n{e.stderr}")
        sys.exit(1)  # Stop the pipeline on first error for safety

# ------------------- Main Control ----------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run full thesis pipeline (end-to-end).")
    parser.add_argument(
        "--skip-extraction", 
        action='store_true', 
        help="Skip data extraction step (use pre-downloaded datasets)."
    )
    parser.add_argument(
        "--skip-diagnostics", 
        action='store_true', 
        help="Skip exploratory analysis and diagnostics (0_Datanalysis.py)."
    )
    parser.add_argument(
        "--skip-output", 
        action='store_true', 
        help="Skip final output analysis (5_output_analysis.py)."
    )
    args = parser.parse_args()

    logging.info("🎬 Starting full pipeline for thesis project...")

    for script in scripts:
        if args.skip_extraction and "1_data_extraction.py" in script:
            logging.info("⏩ Skipping data extraction as requested.")
            continue
        if args.skip_diagnostics and "0_Datanalysis.py" in script:
            logging.info("⏩ Skipping diagnostics as requested.")
            continue
        if args.skip_output and "5_output_analysis.py" in script:
            logging.info("⏩ Skipping output analysis as requested.")
            continue
        run_script(script)

    logging.info("🎉 All pipeline steps completed successfully!")