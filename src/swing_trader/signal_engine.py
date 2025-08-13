import asyncio
from src.swing_trader.indicators import compute_indicators
from src.swing_trader.risk_management import enter_trade, risk_manager, open_positions

async def swing_signal_engine(queue):
    """
    Waits for new candle data from the queue, computes indicators,
    and generates trading signals based on the swing trading strategy.
    """
    while True:
        # Wait for a new message from the websocket handler
        candles = await queue.get()
        
        if len(candles) < 200:
            print(f"⏳ Need 200 candles to start, have {len(candles)}. Waiting...")
            continue

        print("🧠 Analyzing new candle for swing trade signals...")
        df = compute_indicators(candles)
        latest = df.iloc[-1]
        price  = latest["c"]
        ma50, ma200 = latest["MA50"], latest["MA200"]
        rsi  = latest["RSI"]
        vwap = latest["VWAP"]
        high20 = latest["swing_high"]
        low20  = latest["swing_low"]

        # 1) Trend filter: only go long if MA50 > MA200, short if opposite
        trend = 1 if ma50 > ma200 else -1

        # 2) Entry: long pullback to VWAP + RSI < 30
        if trend == 1 and price < vwap and rsi < 30 and "long" not in open_positions:
            await enter_trade("BUY", price, sl=0.03, tp=0.06, hold_bars=5)

        # 3) Entry: breakout above last swing high
        elif trend == 1 and price > high20 and "long" not in open_positions:
            await enter_trade("BUY", price, sl=0.02, tp=0.05, hold_bars=7)

        # 4) Mirror for shorts
        elif trend == -1 and price > vwap and rsi > 70 and "short" not in open_positions:
            await enter_trade("SELL", price, sl=0.03, tp=0.06, hold_bars=5)

        elif trend == -1 and price < low20 and "short" not in open_positions:
            await enter_trade("SELL", price, sl=0.02, tp=0.05, hold_bars=7)

        # 5) checks/risk manager only once per new bar
        await risk_manager(candles)
