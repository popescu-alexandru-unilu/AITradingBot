# src/label_generator_swing.py

import pandas as pd
import numpy as np

INPUT_FILE  = "data/technical_indicators.csv"  # your daily-bar features
OUTPUT_FILE = "data/swing_labeled_dataset.csv"

# ─── swing parameters ────────────────────────────────────────
TP      = 0.03    # 3% take-profit
SL      = 0.02    # 2% stop-loss
HOLD    = 5       # hold at most 5 daily bars
MIN_RET = 0.005   # ignore moves smaller than 0.5%

# ─── 1) load your daily data ─────────────────────────────────
df = pd.read_csv(INPUT_FILE, parse_dates=["timestamp"])
df.set_index("timestamp", inplace=True)

# ─── 2) define direction (long/short) via SMA crossover ─────
#    (you might already have this in your features; adjust as needed)
df["sma_fast"] = df["close"].rolling(10).mean()
df["sma_slow"] = df["close"].rolling(50).mean()
df["side"]     = np.where(df["sma_fast"] > df["sma_slow"], 1, -1)

# ─── 3) volatility proxy (daily returns std over 10d) ───────
df["vol"] = (
    df["close"].pct_change()
      .rolling(10, min_periods=1)
      .std()
      .bfill()
)

# ─── 4) build event barriers ─────────────────────────────────
events = pd.DataFrame(index=df.index)
events["t1"]   = df.index.to_series().shift(-HOLD)  # look HOLD days ahead
events["pt"]   = TP * df["vol"]                    # horiz. profit barrier
events["sl"]   = SL * df["vol"]                    # horiz. stop barrier
events["side"] = df["side"]

# ─── 5) label each t0 ───────────────────────────────────────
labels = pd.Series(index=df.index, dtype=int)

for t0, (t1, pt, sl, side) in events[["t1","pt","sl","side"]].iterrows():
    # if we don't have a full HOLD window, give neutral label
    if pd.isna(t1):
        labels[t0] = 0
        continue

    price_path = df["close"].loc[t0:t1]
    if len(price_path) < 2:
        labels[t0] = 0
        continue

    # side-adjusted returns from t0
    ret = (price_path / price_path.iloc[0] - 1) * side

    # skip if never exceeds your minimum meaningful move
    if ret.abs().max() < MIN_RET:
        labels[t0] = 0
        continue

    # did we hit TP or SL first?
    hits = ret[(ret >= pt) | (ret <= -sl)]
    if hits.empty:
        labels[t0] = 0
    else:
        first_idx = hits.index[0]
        labels[t0] = 1 if ret.loc[first_idx] >= pt else -1
        # shorten vertical barrier down to when we hit it
        events.at[t0, "t1"] = first_idx

# ─── 6) see your distribution ───────────────────────────────
print("Label distribution:\n", labels.value_counts())

# ─── 7) merge back & write out ───────────────────────────────
df_labeled = (
    df
      .join(events[["t1"]])
      .join(labels.rename("label"))
)
df_labeled.to_csv(OUTPUT_FILE)

print(f"[OK] Saved {len(df_labeled)} swing-labels to '{OUTPUT_FILE}'")
