# run_bot.py

import sys
import os
import argparse
from src.predict_and_trade import run_trader
from src.scheduler import schedule_pipeline

# Ensure the src directory is in the Python path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), 'src')))

def main():
    """
    Main entry point for the application.
    Use command-line arguments to choose between running the trading bot
    or updating the data pipeline.
    """
    parser = argparse.ArgumentParser(
        description="ML Trading Bot",
        formatter_class=argparse.RawTextHelpFormatter
    )
    
    parser.add_argument(
        "mode",
        choices=["trade", "update-data"],
        help=(
            "trade:       Run the live trading bot.\n"
            "update-data: Run the data pipeline to refresh data and retrain the model."
        )
    )
    
    parser.add_argument(
        "--interval",
        type=int,
        default=24,
        help="Interval in hours for the data update scheduler (only for 'update-data' mode)."
    )

    args = parser.parse_args()

    if args.mode == "trade":
        print("🚀 Starting the trading bot...")
        run_trader()
    elif args.mode == "update-data":
        print("🔄 Starting the data pipeline scheduler...")
        schedule_pipeline(interval_hours=args.interval)

if __name__ == "__main__":
    main()
