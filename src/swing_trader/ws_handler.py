import asyncio
import websockets
import json
from collections import deque
from utils.config import SYMBOL

# Use a deque to store the last N candles
candles = deque(maxlen=200)  # Store last 200 1h candles

async def ws_handler(queue):
    """
    Connects to the Binance WebSocket stream for 1-hour K-lines,
    processes the messages, and puts closed candles into a queue.
    """
    uri = f"wss://stream.binance.com:9443/ws/{SYMBOL.lower()}@kline_1h"
    async with websockets.connect(uri) as ws:
        print(f"✅ WebSocket connected to {uri}")
        while True:
            msg = await ws.recv()
            data = json.loads(msg)["k"]
            
            is_closed = data["x"]

            # If the candle is closed, we process it
            if is_closed:
                candle = {
                    "ts": data["t"],
                    "o": float(data["o"]),
                    "h": float(data["h"]),
                    "l": float(data["l"]),
                    "c": float(data["c"]),
                    "v": float(data["v"]),
                }
                candles.append(candle)
                # Put the latest full candle data into the queue for the signal engine
                await queue.put(list(candles))
                print(f"🕯️ New 1h candle closed. Price: {candle['c']:.2f}")
