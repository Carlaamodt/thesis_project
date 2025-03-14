import subprocess
import argparse
import logging
import sys
import os

# ------------------- Logging Setup --------------------
logging.basicConfig(
    level=logging.INFO,  # Change to DEBUG for more detailed logs
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("pipeline_log.txt"),  # Save log to file
        logging.StreamHandler(sys.stdout)  # Also print to terminal
    ]
)

# ------------------- Script List ----------------------
scripts = [
    "supporting_modules/1_data_extraction.py",
    "supporting_modules/2_data_processing.py",
    "supporting_modules/3_1_download_fama_french.py",
    "supporting_modules/3_factor_construction.py",
    "supporting_modules/4_factor_model.py",
    "supporting_modules/0_Datanalysis.py",
    "supporting_modules/5_output_analysis.py"
]

# ------------------- Function to Run Scripts ----------------------
def run_script(script_path):
    logging.info(f"🚀 Running: {script_path}")
    result = subprocess.run(["python", script_path], capture_output=True, text=True)
    if result.returncode != 0:
        logging.error(f"❌ Error in {script_path}:\n{result.stderr}")
        exit(1)  # Stop pipeline on failure
    else:
        logging.info(f"✅ Completed: {script_path}")

# ------------------- Main Control ----------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run full thesis pipeline.")
    parser.add_argument("--skip-extraction", action='store_true', help="Skip data extraction step.")
    args = parser.parse_args()

    logging.info("🎬 Starting full pipeline for thesis project...")

    for script in scripts:
        if args.skip_extraction and "1_data_extraction.py" in script:
            logging.info("⏩ Skipping data extraction as requested.")
            continue
        run_script(script)

    logging.info("🎉 All pipeline steps completed successfully!")