from binance.client import Client
import os
import pandas as pd
from utils.config import API_Key, API_Secret

# Load your keys
API_KEY = API_Key
API_SECRET = API_Secret

client = Client(API_KEY, API_SECRET, testnet=True)

def place_order(side, quantity):
    print(f"📡 Placing {side} order of size {quantity} on Binance Testnet...")

    order = client.futures_create_order(
        symbol="BTCUSDT",
        side=side,
        type="MARKET",
        quantity=quantity
    )

    print(" Order placed:", order)
    return order

def get_latest_klines(symbol="BTCUSDT", interval=Client.KLINE_INTERVAL_1HOUR, limit=100):
    klines = client.futures_klines(symbol=symbol, interval=interval, limit=limit)
    df = pd.DataFrame(klines, columns=[
        'timestamp', 'open', 'high', 'low', 'close', 'volume',
        '_', '_', '_', '_', '_', '_'
    ])
    df = df[['timestamp', 'open', 'high', 'low', 'close', 'volume']]
    df[['open', 'high', 'low', 'close', 'volume']] = df[['open', 'high', 'low', 'close', 'volume']].astype(float)
    df['timestamp'] = pd.to_datetime(df['timestamp'], unit="ms")
    return df[["timestamp", "open", "high", "low", "close", "volume"]]
