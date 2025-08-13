import os
import pandas as pd
import pandas_ta as ta
import numpy as np

def _add_multitimeframe_features(df, ohlc):
    """Adds 4h and 1d features to the 1h base data."""
    # 1) RESAMPLE without dropping
    df_4h = df.resample("4H").agg(ohlc)
    df_1d = df.resample("1D").agg(ohlc)  # keep full daily for warm-up

    # 2) 4h RSI & MACD
    df_4h["rsi_4h"] = ta.rsi(df_4h["close"], length=14)
    macd_4h = df_4h.ta.macd(fast=12, slow=26, signal=9, append=False)
    df_4h["macd_4h"] = (
        macd_4h["MACD_12_26_9"]
        if (macd_4h is not None and "MACD_12_26_9" in macd_4h)
        else np.nan
    )

    # 3) Daily RSI & MACD on the full 1d series
    df_1d["rsi_1d"] = (
        ta.rsi(df_1d["close"], length=14)
        if df_1d.shape[0] >= 14
        else np.nan
    )
    # only compute if we have at least slow+signal days
    min_days = 26 + 9
    valid_days = df_1d["close"].dropna().shape[0]

    if valid_days >= min_days:
        macd_1d = df_1d.ta.macd(
            fast=12, slow=26, signal=9, append=False
        )
        df_1d["macd_1d"] = macd_1d["MACD_12_26_9"]
    else:
        # fill with NaN rather than crash
        df_1d["macd_1d"] = np.nan

    # 4) Merge back and forward-fill
    df = df.join(df_4h[["rsi_4h", "macd_4h"]], how="left")
    df = df.join(df_1d[["rsi_1d", "macd_1d"]], how="left")
    df.ffill(inplace=True)  # fill forward to propagate the first valid values

    return df

def compute_swing_indicators(df):
    """
    Takes a minute/tick-level DataFrame with columns
    ['timestamp','open','high','low','close','volume',...]
    and returns an hourly-resampled DataFrame enriched with
    swing-trading features.
    """
    df = df.copy()
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    df.set_index("timestamp", inplace=True)

    ohlc = {
        "open":   "first",
        "high":   "max",
        "low":    "min",
        "close":  "last",
        "volume": "sum",
    }
    hourly = df.resample("1H").agg(ohlc).dropna()

    # Multi-timeframe enrichments
    hourly = _add_multitimeframe_features(hourly, ohlc)

    # (…rest of your hourly indicators as before…)

    # propagate the first real indicator values forward
    hourly.ffill(inplace=True)

    # Optional: if you really want to remove any remaining NaNs,
    # do it only *after* a minimal warm-up, e.g. 200 hours:
    # hourly = hourly.iloc[200:]

    hourly.reset_index(inplace=True)
    return hourly

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("input",  help="raw historical minute/tick CSV")
    parser.add_argument("output", help="where to write hourly features CSV")
    args = parser.parse_args()

    df = pd.read_csv(args.input, parse_dates=["timestamp"])
    swing_df = compute_swing_indicators(df)

    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
    swing_df.to_csv(args.output, index=False)
    print(f"[OK] Saved swing indicators to {args.output} (rows: {len(swing_df)})")
