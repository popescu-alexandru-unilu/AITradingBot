import asyncio
import sys
import os

# Add the project root to the Python path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), ".")))

from src.swing_trader.ws_handler import ws_handler
from src.swing_trader.signal_engine import swing_signal_engine

async def main():
    # Create a queue that we will use to pass messages from the websocket to the signal engine
    queue = asyncio.Queue()

    # Start the websocket handler and the signal engine concurrently
    ws_task = asyncio.create_task(ws_handler(queue))
    engine_task = asyncio.create_task(swing_signal_engine(queue))

    # Run both tasks until they complete
    await asyncio.gather(ws_task, engine_task)

if __name__ == "__main__":
    print("🚀 Starting swing trading bot...")
    asyncio.run(main())
