from datetime import datetime, timedelta
import yfinance as yf
import pandas as pd
import os

symbol   = "AAPL"
interval = "1m"
end_dt   = datetime.utcnow()
limit_dt = end_dt - timedelta(days=30)

all_bars = []
while end_dt > limit_dt:
    start_dt = max(limit_dt, end_dt - timedelta(days=7))
    print(f"Fetching {interval} bars from {start_dt:%Y-%m-%d} → {end_dt:%Y-%m-%d}")
    try:
        chunk = yf.download(
            tickers=symbol,
            start=start_dt,
            end=end_dt,
            interval=interval,
            auto_adjust=True,   # or False if you prefer raw prices
            progress=False,
        )
        if not chunk.empty:
            chunk = chunk.reset_index()
            # rename/reset index column → timestamp, flatten OHLCV MultiIndex, etc.
            cols = list(chunk.columns)
            cols[0] = "timestamp"
            chunk.columns = cols

            flat = []
            for c in chunk.columns:
                if isinstance(c, tuple):
                    flat.append(c[0].lower())
                else:
                    flat.append(str(c).lower())
            chunk.columns = flat

            chunk = chunk[["timestamp","open","high","low","close","volume"]]
            all_bars.append(chunk)

    except Exception as e:
        print("  Failed:", e)

    end_dt = start_dt  # move window back by 7 days

df = pd.concat(all_bars, ignore_index=True).drop_duplicates("timestamp")
os.makedirs("data/raw", exist_ok=True)
df.to_csv("data/raw/minute_bars.csv", index=False)
print(f"[OK] Wrote {len(df)} minute bars.")
