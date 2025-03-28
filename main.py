import subprocess
import logging
import os
import sys

# Ensure output directory exists
output_dir = "output"
os.makedirs(output_dir, exist_ok=True)

log_path = "output/full_terminal_output.txt"
os.makedirs("output", exist_ok=True)
sys.stdout = open(log_path, "w")
sys.stderr = sys.stdout

# Redirect stdout and stderr to a log file
log_path = os.path.join(output_dir, "full_terminal_output.txt")
log_file = open(log_path, "w")
sys.stdout = log_file
sys.stderr = log_file

# ------------------- Logging Setup --------------------
logging.basicConfig(
    level=logging.INFO,  # Change to DEBUG for detailed logs
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(os.path.join(output_dir, "pipeline_log.txt")),  # Save logs to file
        logging.StreamHandler(sys.stdout)  # Also print to redirected stdout
    ]
)

# ------------------- Script List ----------------------
scripts = [
    "supporting_modules/1_data_extraction.py",          # Extract raw data from WRDS
    "supporting_modules/2_data_processing.py",          # Process and clean raw data
    "supporting_modules/3_1_download_fama_french.py",   # Download FF + Momentum data
    "supporting_modules/3_factor_construction.py",      # Construct GA factor (Equal/Value weighted)
    "supporting_modules/4_factor_model.py",             # Run factor model regressions (GA, MKT, SMB, HML etc.)
    "supporting_modules/0_Datanalysis.py",              # Data diagnostic & exploratory analysis
    "supporting_modules/5_output_analysis.py",          # Thesis tables, figures, portfolio characteristics
    "supporting_modules/6_visualization_and_hypothesis_tests.py"  # Visualizations and hypothesis tests
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
            check=True  # Raise CalledProcessError if script fails
        )
        logging.info(f"✅ Completed: {script_path}")
        if result.stdout:
            logging.debug(f"Output from {script_path}:\n{result.stdout}")
        if result.stderr:
            logging.warning(f"Warnings/Errors from {script_path}:\n{result.stderr}")
    except subprocess.CalledProcessError as e:
        logging.error(f"❌ Error in {script_path}:\n{e.stderr}")
        raise  # Re-raise to stop pipeline and signal failure
    except Exception as e:
        logging.error(f"❌ Unexpected error in {script_path}: {str(e)}")
        raise

# ------------------- Main Control ----------------------
if __name__ == "__main__":
    logging.info("🎬 Starting full pipeline for thesis project...")

    try:
        for script in scripts:
            run_script(script)

        logging.info("🎉 All pipeline steps completed successfully!")
    except Exception as e:
        logging.error(f"❌ Pipeline failed: {str(e)}")
        sys.exit(1)
    finally:
        # Ensure log file is closed properly
        log_file.close()