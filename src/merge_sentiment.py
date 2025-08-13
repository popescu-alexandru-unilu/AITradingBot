# src/merge_sentiment.py

import pandas as pd
import os

# Load files
price_data = pd.read_csv("data/combined_dataset.csv", parse_dates=["timestamp"])
news_data = pd.read_csv("data/news_sentiment.csv", parse_dates=["timestamp"])

# Resample news sentiment into hourly averages
news_data.set_index("timestamp", inplace=True)
# Only average the sentiment column
news_hourly = news_data[['sentiment']].resample("1h").mean().rename(columns={"sentiment": "avg_sentiment"})

# Align timestamp to hourly UTC
price_data["hour"] = price_data["timestamp"].dt.floor("h").dt.tz_localize("UTC")

# Merge sentiment into price data
merged = price_data.merge(news_hourly, left_on="hour", right_index=True, how="left")

# Fill any missing sentiment values
merged["avg_sentiment"].fillna(method="ffill", inplace=True)
merged.drop(columns=["hour"], inplace=True)

# Save updated dataset
os.makedirs("data", exist_ok=True)
merged.to_csv("data/combined_dataset_with_sentiment.csv", index=False)
print(f"Merged dataset saved with shape: {merged.shape}")
