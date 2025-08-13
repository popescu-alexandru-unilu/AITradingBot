# src/scheduler.py

import subprocess
import schedule
import time
import os

def run_script(script_path):
    """Executes a Python script and handles errors."""
    try:
        print(f"--- Running {os.path.basename(script_path)} ---")
        # We use sys.executable to ensure we're using the same Python interpreter
        result = subprocess.run(
            [os.sys.executable, script_path],
            check=True,
            capture_output=True,
            text=True
        )
        print(result.stdout)
        print(f"--- Finished {os.path.basename(script_path)} ---")
        return True
    except subprocess.CalledProcessError as e:
        print(f"🚨 Error running {script_path}:")
        print(e.stderr)
        return False
    except FileNotFoundError:
        print(f"🚨 Error: Script not found at {script_path}")
        return False

def data_pipeline_job():
    """
    Runs the entire data collection, processing, and model training pipeline.
    """
    print("🚀 Starting the data pipeline job...")
    
    # Define the sequence of scripts to run
    scripts = [
        "src/collect_price_data.py",
        "src/compute_indicators.py",
        "src/label_generator.py",
        "src/get_news_sentiment.py",
        "src/merge_sentiment.py",
        "src/train_model.py"
    ]
    
    # Execute each script in order
    for script in scripts:
        if not run_script(script):
            print(f"Pipeline failed at script: {script}. Aborting.")
            return # Stop the pipeline if a script fails
            
    print("✅ Data pipeline job completed successfully!")

def schedule_pipeline(interval_hours=24):
    """
    Schedules the data pipeline to run at a regular interval.
    """
    print(f"🕒 Scheduling data pipeline to run every {interval_hours} hours.")
    
    # Schedule the job
    schedule.every(interval_hours).hours.do(data_pipeline_job)
    
    # Initial run
    data_pipeline_job()
    
    # Keep the scheduler running
    while True:
        schedule.run_pending()
        time.sleep(60) # Check for pending jobs every minute

if __name__ == "__main__":
    # This allows running the scheduler directly for testing or deployment
    # For example, you could run this in a Docker container or a background process
    schedule_pipeline(interval_hours=24)
