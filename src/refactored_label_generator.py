# src/refactored_label_generator.py

import pandas as pd
import numpy as np

# --- Configuration ---
INPUT_FILE = "data/raw/minute_bars.csv"
OUTPUT_FILE = "data/refactored_swing_labeled_dataset.csv"
RESAMPLE_FREQ = "1H"

# --- Labeling Parameters ---
# Note: HOLD is in units of the RESAMPLE_FREQ (e.g., 96 hours)
HOLD_PERIODS = 96
ATR_PERIOD = 24  # ATR lookback on hourly data
TP_ATR_MULTIPLIER = 3.0  # Take-profit at 3x ATR
SL_ATR_MULTIPLIER = 1.5  # Stop-loss at 1.5x ATR

# --- Signal Definition ---
SMA_FAST = 20   # Hourly SMA
SMA_SLOW = 100  # Hourly SMA

def get_atr(high, low, close, period):
    """Computes Average True Range."""
    tr1 = pd.DataFrame(high - low)
    tr2 = pd.DataFrame(abs(high - close.shift(1)))
    tr3 = pd.DataFrame(abs(low - close.shift(1)))
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    atr = tr.ewm(alpha=1/period, adjust=False).mean()
    return atr

def triple_barrier_labeling(
    prices: pd.DataFrame,
    t_events: pd.Index,
    side: pd.Series,
    hold_periods: int,
    tp_atr: pd.Series,
    sl_atr: pd.Series,
) -> pd.Series:
    """
    Generates labels for swing trades based on the triple-barrier method.

    Args:
        prices: DataFrame with columns ['open', 'high', 'low', 'close'].
        t_events: The timestamps of the events or signals.
        side: Series indicating trade direction (1 for long, -1 for short).
        hold_periods: Maximum number of bars to hold the position.
        tp_atr: Series of take-profit levels (absolute price change).
        sl_atr: Series of stop-loss levels (absolute price change).

    Returns:
        A Series of labels (1 for TP, -1 for SL, 0 for timeout or no hit).
    """
    labels = pd.Series(0, index=t_events)
    
    for t0 in t_events:
        entry_price = prices.at[t0, "close"]
        trade_side = side.at[t0]
        
        # Define barriers
        tp_price = entry_price + tp_atr.at[t0] * trade_side
        sl_price = entry_price - sl_atr.at[t0] * trade_side
        
        # Get the price path for the holding period
        t1_idx = prices.index.get_loc(t0) + hold_periods
        if t1_idx >= len(prices.index):
            t1 = prices.index[-1]
        else:
            t1 = prices.index[t1_idx]
            
        path = prices.loc[t0:t1]
        
        # Check for hits
        if trade_side == 1: # Long trade
            tp_hits = path[path["high"] >= tp_price]
            sl_hits = path[path["low"] <= sl_price]
        else: # Short trade
            tp_hits = path[path["low"] <= tp_price]
            sl_hits = path[path["high"] >= sl_price]
            
        # Determine first hit
        first_tp_hit = tp_hits.index.min() if not tp_hits.empty else pd.NaT
        first_sl_hit = sl_hits.index.min() if not sl_hits.empty else pd.NaT
        
        if pd.isna(first_tp_hit) and pd.isna(first_sl_hit):
            labels.at[t0] = 0 # Timed out
        elif pd.isna(first_sl_hit) or first_tp_hit <= first_sl_hit:
            labels.at[t0] = 1 # Take-profit hit
        else:
            labels.at[t0] = -1 # Stop-loss hit
            
    return labels

def main():
    """Main function to generate and save labels."""
    # 1. Load and resample data
    df_min = pd.read_csv(
        INPUT_FILE,
        parse_dates=["timestamp"],
        index_col="timestamp",
        usecols=["timestamp", "open", "high", "low", "close", "volume"]
    )
    ohlc = {"open": "first", "high": "max", "low": "min", "close": "last", "volume": "sum"}
    df1h = df_min.resample(RESAMPLE_FREQ).agg(ohlc).dropna()
    print(f"Resampled to {len(df1h)} hourly bars.")

    # 2. Define trade direction (signal)
    df1h["sma_fast"] = df1h["close"].rolling(SMA_FAST).mean()
    df1h["sma_slow"] = df1h["close"].rolling(SMA_SLOW).mean()
    df1h["side"] = 0
    df1h.loc[df1h["sma_fast"] > df1h["sma_slow"], "side"] = 1
    df1h.loc[df1h["sma_fast"] < df1h["sma_slow"], "side"] = -1
    
    # Only consider events where the signal is non-zero
    t_events = df1h[df1h["side"] != 0].index
    side = df1h.loc[t_events, "side"]
    
    # 3. Calculate ATR for dynamic barriers
    df1h["atr"] = get_atr(df1h["high"], df1h["low"], df1h["close"], period=ATR_PERIOD)
    tp_atr = df1h["atr"] * TP_ATR_MULTIPLIER
    sl_atr = df1h["atr"] * SL_ATR_MULTIPLIER
    
    # 4. Generate labels
    print(f"Generating labels for {len(t_events)} events...")
    labels = triple_barrier_labeling(
        prices=df1h,
        t_events=t_events,
        side=side,
        hold_periods=HOLD_PERIODS,
        tp_atr=tp_atr,
        sl_atr=sl_atr,
    )
    
    # 5. Save results
    output_df = pd.DataFrame({"label": labels})
    output_df.to_csv(OUTPUT_FILE)
    print("Label distribution:\n", labels.value_counts())
    print(f"✅ Saved {len(output_df)} labels to '{OUTPUT_FILE}'")

if __name__ == "__main__":
    main()
