import pandas as pd
from pandas import DataFrame

def compute_indicators(candles: list) -> DataFrame:
    """
    Computes technical indicators from a list of candle data.
    """
    df = DataFrame(candles).set_index("ts")
    # convert ms→datetime if needed
    df.index = pd.to_datetime(df.index, unit="ms")
    
    # Moving averages
    df["MA50"]  = df["c"].rolling(50).mean()
    df["MA200"] = df["c"].rolling(200).mean()
    
    # RSI (14)
    delta = df["c"].diff()
    up  = delta.clip(lower=0).rolling(14).mean()
    dn  = -delta.clip(upper=0).rolling(14).mean()
    df["RSI"] = 100 - 100/(1 + up/dn)
    
    # VWAP per bar
    df["VWAP"] = (df["c"] * df["v"]).cumsum() / df["v"].cumsum()
    
    # recent swing highs/lows
    df["swing_high"] = df["c"].rolling(20).max().shift(1)
    df["swing_low"]  = df["c"].rolling(20).min().shift(1)
    
    return df.dropna()
