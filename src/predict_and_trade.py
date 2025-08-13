# src/predict_and_trade.py

import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import pandas as pd
import joblib
from utils.binance_utils import get_latest_klines, place_order
from src.compute_indicators import compute_indicators
import time
from datetime import datetime
import numpy as np

# --- Load Trained Model ---
try:
    model = joblib.load("models/xgb_meta_labeler.pkl")
except FileNotFoundError:
    print("Model not found. Please train the model first.")
    model = None

# --- Trading State ---
time_since_last_signal = 0

def get_prediction_features():
    """
    Fetch latest price data, compute indicators, and return features for prediction.
    """
    df_price = get_latest_klines(limit=100)
    df_features = compute_indicators(df_price)

    # Add stateful features
    df_features['time_since_last_signal'] = time_since_last_signal

    # Match model's feature order
    model_features = model.feature_names_in_
    missing = [f for f in model_features if f not in df_features.columns]
    if missing:
        raise ValueError(f"Missing features: {missing}")

    return df_features[model_features].iloc[-1]

def execute_trade_logic():
    """
    Predict and execute buy/sell logic based on model confidence.
    """
    global time_since_last_signal

    print("🤖 Checking for trading opportunities...")

    if model is None:
        return

    try:
        features = get_prediction_features()
    except Exception as e:
        print(f"❌ Failed to get prediction features: {e}")
        return

    proba_win = model.predict_proba([features.values])[0][1]

    action = "HOLD"
    trade_size = 0

    if proba_win > 0.55: # Adjusted threshold for scalping
        trade_size = 1.0
        action = "BUY"
        print(f"🟢 High confidence ({proba_win:.2f}) → Placing FULL size BUY order.")
        place_order('BUY', trade_size)
        time_since_last_signal = 0

    elif proba_win < 0.45: # Adjusted threshold for scalping
        trade_size = 1.0
        action = "SELL"
        print(f"🔻 High confidence of DROP ({proba_win:.2f}) → Placing FULL size SELL order.")
        place_order('SELL', trade_size)
        time_since_last_signal = 0

    else:
        print(f"🔴 Low confidence ({proba_win:.2f}) → No trade.")
        time_since_last_signal += 1

    # --- Log Trade ---
    log_entry = {
        "timestamp": pd.Timestamp.utcnow(),
        "confidence": round(proba_win, 4),
        "action": action,
        "size": trade_size
    }

    log_df = pd.DataFrame([log_entry])
    os.makedirs("data", exist_ok=True)
    log_df.to_csv("data/trade_log.csv", mode='a', header=not os.path.exists("data/trade_log.csv"), index=False)

def run_backtest(data_path="data/meta_labeled_dataset.csv"):
    """
    Run a simplified backtest on historical data.
    """
    if model is None:
        return

    df = pd.read_csv(data_path, index_col="timestamp", parse_dates=True)
    df.dropna(subset=["label"], inplace=True)
    
    # Ensure all required features are present
    missing_features = set(model.feature_names_in_) - set(df.columns)
    if missing_features:
        print(f"Missing features in the dataset: {missing_features}")
        return

    X_holdout = df[model.feature_names_in_]
    
    # Predict probabilities
    pred_proba = model.predict_proba(X_holdout)[:, 1]
    
    # Create a simple output dataframe
    results = pd.DataFrame(index=X_holdout.index)
    results['predictions'] = pred_proba
    
    # Save results to a file for analysis
    results.to_csv("data/backtest_pnl.csv")
    print("Saved backtest predictions to data/backtest_pnl.csv")

def run_trader():
    """
    Continuously run the trading bot.
    """
    while True:
        execute_trade_logic()
        print("--- Waiting for next candle ---")
        time.sleep(10)

if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == 'backtest':
        print("Starting backtest...")
        run_backtest()
    else:
        print("Starting Binance Testnet Trading Bot...")
        execute_trade_logic()
