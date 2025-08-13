import requests
import feedparser
from textblob import TextBlob
from datetime import datetime
from dateutil import parser as dateparser
import pandas as pd
import os

# News source
RSS_URL = "https://www.coindesk.com/arc/outboundfeeds/rss/"
SENTIMENT_FILE = "data/news_sentiment.csv"

# Create data directory if needed
os.makedirs("data", exist_ok=True)

# Make HTTP request with headers
headers = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64)"
}
response = requests.get(RSS_URL, headers=headers)

# Parse feed content
feed = feedparser.parse(response.content)
print(f"Found {len(feed.entries)} headlines")

# Extract and score headlines
data = []
for entry in feed.entries:
    title = entry.title
    time_str = entry.get("published", entry.get("updated"))
    if not time_str:
        continue
    score = TextBlob(title).sentiment.polarity
    timestamp = dateparser.parse(time_str)
    data.append((timestamp, title, score))

# Save to CSV
df = pd.DataFrame(data, columns=["timestamp", "headline", "sentiment"])
df.to_csv(SENTIMENT_FILE, index=False)
print(f" Saved {len(df)} sentiment entries to {SENTIMENT_FILE}")
