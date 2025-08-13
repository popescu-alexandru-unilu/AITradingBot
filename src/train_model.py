# src/train_model.py
import pandas as pd
import numpy as np
import joblib

# 1) Load labeler and raw data
# new
clf = joblib.load("models/trained_model.pkl")

df  = pd.read_csv("data/technical_indicators.csv", parse_dates=["timestamp"], index_col="timestamp")

# Recompute features exactly as in labeling
df["sma_fast"] = df["close"].rolling(5).mean()
df["sma_slow"] = df["close"].rolling(20).mean()
feature_cols = [c for c in df.columns if c not in ["timestamp"]]
X = df[feature_cols].fillna(method="bfill")

# 2) Predict labels & turn into positions
df["pred_label"] = clf.predict(X)
# positions: +1 for long, –1 for short, 0 flat
df["position"]   = df["pred_label"].shift().fillna(0)

# 3) Simulate daily returns
df["return"]        = df["close"].pct_change().fillna(0)
df["strategy_ret"]  = df["position"] * df["return"]

# 4) Compute performance
df["cum_strategy"] = (1 + df["strategy_ret"]).cumprod()
df["cum_buyhold"]  = (1 + df["return"]).cumprod()

print("Final strategy return:", df["cum_strategy"].iloc[-1])
print("Final buy & hold   :", df["cum_buyhold"].iloc[-1])

# 5) Quick plot
import matplotlib.pyplot as plt
plt.plot(df["cum_strategy"], label="Strategy")
plt.plot(df["cum_buyhold"],  label="Buy & Hold")
plt.legend()
plt.show()
