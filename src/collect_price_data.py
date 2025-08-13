# src/collect_price_data.py

from binance.client import Client
import pandas as pd
import os
from datetime import datetime

# --- Config ---
API_KEY = "11ea01f8b333f7ed4b09d8aa6acd628f632130a419fbac6a6de4d420cf1dab52"
API_SECRET = "7af16a78efd0e48f2bb71c8a9f35031e691a78cf0a59b0e703954b7d18bdd0fe"
SYMBOL = "BTCUSDT"
INTERVAL = Client.KLINE_INTERVAL_15MINUTE
OUTPUT_FILE = "data/historical_prices.csv"

# --- Connect to Binance ---
client = Client(API_KEY, API_SECRET)

# --- Paginated Fetch Function ---
def fetch_all_klines(symbol, interval, start_str, end_str=None, limit=1000):
    columns = ['timestamp', 'open', 'high', 'low', 'close', 'volume',
               'close_time', 'quote_asset_volume', 'number_of_trades',
               'taker_buy_base', 'taker_buy_quote', 'ignore']
    
    df_all = pd.DataFrame()
    start_ts = int(pd.to_datetime(start_str).timestamp() * 1000)
    end_ts = int(pd.to_datetime(end_str).timestamp() * 1000) if end_str else None

    while True:
        klines = client.get_klines(
            symbol=symbol,
            interval=interval,
            startTime=start_ts,
            endTime=end_ts,
            limit=limit
        )
        if not klines:
            break

        temp_df = pd.DataFrame(klines, columns=columns)
        temp_df['timestamp'] = pd.to_datetime(temp_df['timestamp'], unit='ms')
        temp_df = temp_df[['timestamp', 'open', 'high', 'low', 'close', 'volume']]
        temp_df = temp_df.astype({
            'open': float, 'high': float, 'low': float,
            'close': float, 'volume': float
        })

        df_all = pd.concat([df_all, temp_df], ignore_index=True)

        if len(klines) < limit:
            break

        start_ts = int(klines[-1][0]) + 1
        if end_ts and start_ts >= end_ts:
            break

    return df_all

# --- Fetch & Save ---
START_DATE = "2024-01-01"
df = fetch_all_klines(SYMBOL, INTERVAL, START_DATE)
os.makedirs("data", exist_ok=True)
df.to_csv(OUTPUT_FILE, index=False)
print(f" Saved {len(df)} candles to {OUTPUT_FILE}")
