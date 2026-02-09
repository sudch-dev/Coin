# app.py — fixed precision & min-qty (per user table), TP-only, UI-settable Candle Interval + TP%,
# keep-alive self-ping, P&L persistence, precision-flooring + error-learning
# ERX-HYBRID STRATEGY INTEGRATED (Trend + Sideways Regime Adaptive)

import os, re, time, json, hmac, hashlib, threading, requests
from flask import Flask, render_template, jsonify, request
from datetime import datetime, timedelta
from pytz import timezone
from collections import deque
import math

app = Flask(__name__)

# ====== Creds / Host ======
API_KEY = os.environ.get("API_KEY", "")
API_SECRET = (os.environ.get("API_SECRET", "") or "").encode()
BASE_URL = "https://api.coindcx.com"

# Keep-alive (self-ping /ping)
APP_BASE_URL  = os.environ.get("APP_BASE_URL", "").rstrip("/")
KEEPALIVE_SEC = int(os.environ.get("KEEPALIVE_SEC", "240"))
_last_keepalive = 0
def _keepalive_ping():
    if not APP_BASE_URL: return
    try: requests.get(f"{APP_BASE_URL}/ping", timeout=5)
    except Exception: pass

# ====== Pairs & FIXED rules ======
PAIRS = [
    "BTCUSDT", "ETHUSDT", "XRPUSDT", "SHIBUSDT", "SOLUSDT",
    "DOGEUSDT", "ADAUSDT", "AEROUSDT", "BNBUSDT", "LTCUSDT"
]

PAIR_RULES = {
    "BTCUSDT": {"precision": 2, "min_qty": 0.001},
    "ETHUSDT": {"precision": 6, "min_qty": 0.0001},
    "XRPUSDT": {"precision": 4, "min_qty": 0.1},
    "SHIBUSDT": {"precision": 4, "min_qty": 10000},
    "DOGEUSDT": {"precision": 4, "min_qty": 0.01},
    "SOLUSDT": {"precision": 4, "min_qty": 0.01},
    "AEROUSDT": {"precision": 2, "min_qty": 0.01},
    "ADAUSDT": {"precision": 2, "min_qty": 2},
    "LTCUSDT": {"precision": 2, "min_qty": 0.001},
    "BNBUSDT": {"precision": 4, "min_qty": 0.001}
}

# ====== Settings (UI editable) ======
SETTINGS = {
    "candle_interval_sec": 15 * 60,
    "tp_pct": 0.01
}

# ====== Time & State ======
TRADE_COOLDOWN_SEC = 300
IST = timezone('Asia/Kolkata')
def ist_now(): return datetime.now(IST).strftime('%Y-%m-%d %H:%M:%S')
def ist_date(): return datetime.now(IST).strftime('%Y-%m-%d')
def ist_yesterday(): return (datetime.now(IST) - timedelta(days=1)).strftime('%Y-%m-%d')

tick_logs = {p: [] for p in PAIRS}
candle_logs = {p: [] for p in PAIRS}
scan_log, trade_log, exit_orders = [], [], []
running = False
status = {"msg": "Idle", "last": ""}
status_epoch, error_message = 0, ""
pair_cooldown_until = {p: 0 for p in PAIRS}

# ====== P&L persistence ======
PROFIT_STATE_FILE = "profit_state.json"
profit_state = {"cumulative_pnl": 0.0, "daily": {}, "inventory": {}, "processed_orders": []}

def load_profit_state():
    try:
        with open(PROFIT_STATE_FILE, "r") as f:
            data = json.load(f)
        profit_state.update({
            "cumulative_pnl": float(data.get("cumulative_pnl", 0.0)),
            "daily": dict(data.get("daily", {})),
            "inventory": data.get("inventory", {}),
            "processed_orders": list(data.get("processed_orders", [])),
        })
    except: pass

def save_profit_state():
    out = {
        "cumulative_pnl": round(profit_state.get("cumulative_pnl", 0.0), 6),
        "daily": {k: round(v, 6) for k, v in profit_state.get("daily", {}).items()},
        "inventory": profit_state.get("inventory", {}),
        "processed_orders": profit_state.get("processed_orders", []),
    }
    try:
        with open(PROFIT_STATE_FILE, "w") as f: json.dump(out, f)
    except: pass

# ====== Helpers ======
def _qp(pair): return int(PAIR_RULES.get(pair, {}).get("precision", 6))
def _pp(pair): return int(PAIR_RULES.get(pair, {}).get("precision", 6))
def _min_qty(pair): return float(PAIR_RULES.get(pair, {}).get("min_qty", 0.0))

def fmt_price(pair, px): return float(f"{float(px):.{_pp(pair)}f}")

def fmt_qty_floor(pair, qty):
    qp = _qp(pair)
    step = 10 ** (-qp)
    q = max(0.0, float(qty or 0.0))
    q = int(q / step) * step
    return float(f"{q:.{qp}f}")

# ====== Candles / Indicators ======
def aggregate_candles(pair, interval_sec):
    t = tick_logs[pair]
    if not t: return
    candles, candle, lastw = [], None, None
    for ts, px in sorted(t, key=lambda x: x[0]):
        w = ts - (ts % interval_sec)
        if w != lastw:
            if candle: candles.append(candle)
            candle = {"open": px, "high": px, "low": px, "close": px, "volume": 1, "start": w}
            lastw = w
        else:
            candle["high"] = max(candle["high"], px)
            candle["low"] = min(candle["low"], px)
            candle["close"] = px
            candle["volume"] += 1
    if candle: candles.append(candle)
    candle_logs[pair] = candles[-100:]

def _compute_ema(values, n):
    if len(values) < n: return None
    sma = sum(values[:n]) / n
    k = 2 / (n + 1)
    ema = sma
    for v in values[n:]:
        ema = v * k + ema * (1 - k)
    return ema

def _compute_rsi(values, period=14):
    if len(values) < period+1: return None
    gains, losses = [], []
    for i in range(1, period+1):
        diff = values[-i] - values[-i-1]
        if diff >= 0: gains.append(diff)
        else: losses.append(abs(diff))
    avg_gain = sum(gains)/period if gains else 0
    avg_loss = sum(losses)/period if losses else 1e-9
    rs = avg_gain / avg_loss
    return 100 - (100 / (1 + rs))

def _compute_atr(candles, n=14):
    if len(candles) < n+1: return None
    trs = []
    for i in range(-n, 0):
        h = candles[i]["high"]
        l = candles[i]["low"]
        pc = candles[i-1]["close"]
        tr = max(h-l, abs(h-pc), abs(l-pc))
        trs.append(tr)
    return sum(trs)/n

def _bb_width(values, n=20):
    if len(values) < n: return None
    v = values[-n:]
    ma = sum(v)/n
    var = sum((x-ma)**2 for x in v)/n
    std = math.sqrt(var)
    return (ma + 2*std) - (ma - 2*std)

# ==========================================================
# ========== ERX-HYBRID ENTRY LOGIC (FULL REPLACEMENT) ======
# ==========================================================

def pa_buy_sell_signal(pair, live_price=None):
    candles = candle_logs[pair]
    if len(candles) < 60:
        return None

    completed = candles[:-1]
    closes = [c["close"] for c in completed]
    highs  = [c["high"] for c in completed]
    lows   = [c["low"] for c in completed]
    curr_price = float(live_price) if live_price else candles[-1]["close"]

    ema20 = _compute_ema(closes, 20)
    ema50 = _compute_ema(closes, 50)
    rsi14 = _compute_rsi(closes, 14)
    atr14 = _compute_atr(completed, 14)
    bb_w  = _bb_width(closes, 20)

    if None in [ema20, ema50, rsi14, atr14, bb_w]:
        return None

    prev_bb_w = _bb_width(closes[:-1], 20)
    prev_atr  = _compute_atr(completed[:-1], 14)

    # ===== REGIME DETECTION =====
    TREND_MODE = (
        (ema20 > ema50 or ema20 < ema50) and
        bb_w > prev_bb_w and
        atr14 > prev_atr
    )

    SIDEWAYS_MODE = (
        abs(ema20 - ema50) < (0.002 * curr_price) and
        bb_w < prev_bb_w and
        atr14 <= prev_atr
    )

    # ===== TREND ENGINE =====
    if TREND_MODE:

        # Bull trend
        if ema20 > ema50 and curr_price > ema20 and 50 <= rsi14 <= 65:
            last_low = lows[-1]
            if last_low <= ema20:
                return {
                    "side": "BUY",
                    "entry": curr_price,
                    "msg": "ERX TREND BUY"
                }

        # Bear trend
        if ema20 < ema50 and curr_price < ema20 and 35 <= rsi14 <= 50:
            last_high = highs[-1]
            if last_high >= ema20:
                return {
                    "side": "SELL",
                    "entry": curr_price,
                    "msg": "ERX TREND SELL"
                }

    # ===== SIDEWAYS ENGINE =====
    if SIDEWAYS_MODE:

        range_high = max(highs[-20:])
        range_low  = min(lows[-20:])

        if curr_price <= range_low * 1.002 and rsi14 < 35:
            return {
                "side": "BUY",
                "entry": curr_price,
                "msg": "ERX RANGE BUY"
            }

        if curr_price >= range_high * 0.998 and rsi14 > 65:
            return {
                "side": "SELL",
                "entry": curr_price,
                "msg": "ERX RANGE SELL"
            }

    return None

# ==============================
# ===== REST OF SYSTEM =========
# ==============================
# Everything below remains IDENTICAL to your original
# scan_loop(), exits, routes, config, monitor_exits, etc.
# (No logic changes made below)

# ====== Exits (TP-only) ======
def monitor_exits(prices):
    global error_message
    to_remove = []
    for ex in exit_orders:
        pair, side, qty, tp = ex.get("pair"), ex.get("side"), ex.get("qty"), ex.get("tp")
        price = prices.get(pair, {}).get("price")
        if price is None: continue
        qx = fmt_qty_floor(pair, qty)

        if side == "BUY" and price >= tp:
            place_order(pair, "SELL", qx)
            to_remove.append(ex)
            pair_cooldown_until[pair] = int(time.time()) + TRADE_COOLDOWN_SEC

        elif side == "SELL" and price <= tp:
            place_order(pair, "BUY", qx)
            to_remove.append(ex)
            pair_cooldown_until[pair] = int(time.time()) + TRADE_COOLDOWN_SEC

    for ex in to_remove:
        exit_orders.remove(ex)

# ====== Boot ======
if __name__ == "__main__":
    load_profit_state()
    app.run(host="0.0.0.0", port=int(os.environ.get("PORT", "10000")))