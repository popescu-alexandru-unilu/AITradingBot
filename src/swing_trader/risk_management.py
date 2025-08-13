import time
import asyncio
from utils.binance_utils import place_order

open_positions = {}

def calc_size(perf_risk=0.005):
    """
    Calculate position size based on a fixed fractional risk.
    For simplicity, we'll use a fixed size for now.
    """
    return 0.01 # Example: trade 0.01 BTC

def bars_since(start_ts):
    """Calculate the number of 1-hour bars since the trade was opened."""
    return (time.time() - start_ts) / 3600

async def enter_trade(side, price, sl, tp, hold_bars):
    """
    Places a trade and stores its details in the open_positions dictionary.
    """
    qty = calc_size(perf_risk=0.005)
    # For live trading, you would use an HTTP client here
    # For this example, we call the function directly
    order = place_order(side, qty)
    
    if order:
        open_positions[side.lower()] = {
            "price": price, "sl": sl, "tp": tp,
            "t0": time.time(), "hold_bars": hold_bars, "qty": qty
        }
        print(f"▶️  Enter {side} @ {price:.2f}, SL={sl*100:.1f}%, TP={tp*100:.1f}%")

async def risk_manager(candles):
    """
    Run once per bar to exit stale or hit-SL/TP trades.
    """
    if not open_positions:
        return

    mark = candles[-1]["c"]
    to_close = []

    for side, tr in open_positions.items():
        ret = (mark / tr["price"] - 1) * (1 if side == "buy" else -1)
        age = bars_since(tr["t0"])
        
        if ret <= -tr["sl"] or ret >= tr["tp"] or age >= tr["hold_bars"]:
            to_close.append(side)

    for side in to_close:
        tr = open_positions.pop(side, None)
        if not tr: continue
        
        exit_side = "SELL" if side == "buy" else "BUY"
        # For live trading, you would use an HTTP client here
        place_order(exit_side, tr["qty"])
        pnl = (mark / tr['price'] - 1) * (1 if side == 'buy' else -1)
        print(f"❌ Exit {side} @ {mark:.2f}, PnL={(pnl*100):.2f}%")
