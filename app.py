import os
import time
import threading
import hmac
import hashlib
import requests
import json
from datetime import datetime
from pytz import timezone
from flask import Flask, render_template, jsonify
from statistics import mean

# =========================
# Flask
# =========================
app = Flask(__name__)

# =========================
# Config / Constants
# =========================
API_KEY = os.environ.get("API_KEY")
API_SECRET = (os.environ.get("API_SECRET") or "").encode()
BASE_URL = "https://api.coindcx.com"

# Spot pairs to scan
PAIRS = [
    "BTCUSDT", "ETHUSDT", "XRPUSDT", "SHIBUSDT", "SOLUSDT",
    "DOGEUSDT", "ADAUSDT", "AEROUSDT", "BNBUSDT", "LTCUSDT"
]

# Pair precision (quantity) + min qty (coin-wise)
PAIR_RULES = {
    "BTCUSDT": {"precision": 6, "min_qty": 0.0001},
    "ETHUSDT": {"precision": 6, "min_qty": 0.0001},
    "XRPUSDT": {"precision": 4, "min_qty": 0.1},
    "SHIBUSDT": {"precision": 0, "min_qty": 10000},
    "DOGEUSDT": {"precision": 2, "min_qty": 1},
    "SOLUSDT": {"precision": 3, "min_qty": 0.01},
    "AEROUSDT": {"precision": 4, "min_qty": 1},
    "ADAUSDT": {"precision": 2, "min_qty": 1},
    "LTCUSDT": {"precision": 2, "min_qty": 0.01},
    "BNBUSDT": {"precision": 4, "min_qty": 0.001},
}

# Candle interval (seconds) -> 5s scalper
CANDLE_INTERVAL_SEC = 5

# Loop cadence (seconds)
POLL_SEC = 1.0

# Allocation per BUY from USDT
BUY_USDT_ALLOC = 0.35

# Target & SL policy
TP_PCT = 0.002   # +0.2% take-profit
# SL: reverse signal (signal-based exit), not a price level

# =========================
# Time helpers (IST)
# =========================
IST = timezone("Asia/Kolkata")
def ist_now(): return datetime.now(IST).strftime("%Y-%m-%d %H:%M:%S")

# =========================
# State
# =========================
session = requests.Session()

tick_logs   = {p: [] for p in PAIRS}   # [(ts, price)]
candle_logs = {p: [] for p in PAIRS}   # dicts: open/high/low/close/volume/start
scan_log    = []                       # rolling text logs for UI
trade_log   = []                       # entries & exits
running     = False
status      = {"msg": "Idle", "last": ""}
error_message = ""

# Track open LONG positions only (spot)
# positions[pair] = {"side":"BUY","qty":float,"entry":float,"tp":float}
positions = {p: None for p in PAIRS}

# =========================
# Utils / API
# =========================
def hmac_signature(payload: str) -> str:
    return hmac.new(API_SECRET, payload.encode(), hashlib.sha256).hexdigest()

def get_wallet_balances():
    payload = json.dumps({"timestamp": int(time.time() * 1000)})
    sig = hmac_signature(payload)
    headers = {"X-AUTH-APIKEY": API_KEY, "X-AUTH-SIGNATURE": sig, "Content-Type": "application/json"}
    balances = {}
    try:
        r = session.post(f"{BASE_URL}/exchange/v1/users/balances", headers=headers, data=payload, timeout=8)
        if r.ok:
            for b in r.json():
                balances[b["currency"]] = float(b["balance"])
        else:
            scan_log.append(f"{ist_now()} | BAL_ERR: {r.status_code} {r.text[:160]}")
    except Exception as e:
        scan_log.append(f"{ist_now()} | BAL_ERR: {e}")
    return balances

def fetch_all_prices():
    try:
        r = session.get(f"{BASE_URL}/exchange/ticker", timeout=8)
        if r.ok:
            now = int(time.time())
            return {
                item["market"]: {"price": float(item["last_price"]), "ts": now}
                for item in r.json() if item.get("market") in PAIRS
            }
        else:
            scan_log.append(f"{ist_now()} | PRICE_ERR: {r.status_code} {r.text[:160]}")
    except Exception as e:
        scan_log.append(f"{ist_now()} | PRICE_ERR: {e}")
    return {}

def clamp_buy_qty(pair, qty):
    rule = PAIR_RULES.get(pair, {"precision": 6, "min_qty": 0.0})
    p = int(rule["precision"]); minq = float(rule["min_qty"])
    step = (10 ** (-p)) if p > 0 else 1.0
    qty = (int(max(qty, 0.0) / step)) * step
    if qty < minq: return 0.0
    return round(qty, p)

def precise_sell_qty(pair, balance_qty):
    rule = PAIR_RULES.get(pair, {"precision": 6, "min_qty": 0.0})
    p = int(rule["precision"]); minq = float(rule["min_qty"])
    step = (10 ** (-p)) if p > 0 else 1.0
    qty = (int(max(balance_qty, 0.0) / step)) * step
    if qty < minq: return 0.0
    return round(qty, p)

def place_order(pair, side, qty):
    payload = {
        "market": pair,
        "side": side.lower(),             # "buy" / "sell"
        "order_type": "market_order",
        "total_quantity": str(qty),
        "timestamp": int(time.time() * 1000),
    }
    body = json.dumps(payload)
    sig = hmac_signature(body)
    headers = {"X-AUTH-APIKEY": API_KEY, "X-AUTH-SIGNATURE": sig, "Content-Type": "application/json"}
    try:
        r = session.post(f"{BASE_URL}/exchange/v1/orders/create", headers=headers, data=body, timeout=8)
        return r.json()
    except Exception as e:
        return {"error": str(e)}

# =========================
# Indicators (5s candles)
# =========================
def sma(values, period):
    if len(values) < period: return None
    return sum(values[-period:]) / period

def rsi_wilder_from_closes(closes, period=14):
    if len(closes) < period + 1: return None
    gains, losses = [], []
    for i in range(1, period + 1):
        ch = closes[i] - closes[i-1]
        gains.append(max(0.0, ch)); losses.append(max(0.0, -ch))
    avg_gain = sum(gains)/period; avg_loss = sum(losses)/period
    for i in range(period + 1, len(closes)):
        ch = closes[i] - closes[i-1]
        gain = max(0.0, ch); loss = max(0.0, -ch)
        avg_gain = (avg_gain*(period-1) + gain)/period
        avg_loss = (avg_loss*(period-1) + loss)/period
    if avg_loss == 0: return 100.0
    rs = avg_gain / avg_loss
    return 100 - (100 / (1 + rs))

def stoch_kd(c, period=14, smooth_k=3, smooth_d=3):
    if len(c) < period + max(smooth_k, smooth_d): return (None, None)
    ks = []
    for i in range(period-1, len(c)):
        win = c[i-period+1:i+1]
        hi = max(x["high"] for x in win); lo = min(x["low"] for x in win)
        cl = c[i]["close"]
        k_raw = 0.0 if hi == lo else (cl - lo) / (hi - lo) * 100.0
        ks.append(k_raw)
    def smooth(arr, times):
        out = list(arr)
        for _ in range(times-1):
            tmp = [out[0]] + [(out[j-1]+out[j])/2.0 for j in range(1, len(out))]
            out = tmp
        return out
    k_s = smooth(ks, smooth_k)
    d_s = smooth(k_s, smooth_d)
    return k_s[-1], d_s[-1]

# =========================
# Aggregation (5s candles)
# =========================
def aggregate_candles(pair, interval=CANDLE_INTERVAL_SEC):
    ticks = tick_logs[pair]
    if not ticks: return
    candles, candle, last_window = [], None, None
    for ts, price in sorted(ticks, key=lambda x: x[0]):
        wstart = ts - (ts % interval)
        if last_window != wstart:
            if candle: candles.append(candle)
            candle = {"open": price, "high": price, "low": price, "close": price, "volume": 1, "start": wstart}
            last_window = wstart
        else:
            candle["high"] = max(candle["high"], price)
            candle["low"]  = min(candle["low"], price)
            candle["close"]= price
            candle["volume"] += 1
    if candle: candles.append(candle)
    candle_logs[pair] = candles[-600:]  # enough history

# =========================
# Signals (entry/exit; reverse = SL)
# =========================
def compute_signal(candles):
    """Return 'BUY' or 'SELL' or None based on last CLOSED candle."""
    if len(candles) < 10:  # need at least SMA10
        return None

    closes = [c["close"] for c in candles]
    sma5  = sma(closes, 5)
    sma10 = sma(closes, 10)
    rsi14 = rsi_wilder_from_closes(closes[-15:], 14) if len(closes) >= 15 else None
    k, d  = stoch_kd(candles, 14, 3, 3) if len(candles) >= 20 else (None, None)

    # Tolerant rule: SMA cross is primary; RSI/Stoch confirm if available
    buy_core  = sma5 is not None and sma10 is not None and (sma5 > sma10)
    sell_core = sma5 is not None and sma10 is not None and (sma5 < sma10)

    buy_ok  = buy_core
    sell_ok = sell_core

    if rsi14 is not None:
        buy_ok  = buy_ok  and (rsi14 > 55)
        sell_ok = sell_ok and (rsi14 < 45)
    if k is not None and d is not None:
        buy_ok  = buy_ok  and (k > d)  # optional: and k < 80
        sell_ok = sell_ok and (k < d)  # optional: and k > 20

    if buy_ok:  return "BUY"
    if sell_ok: return "SELL"
    return None

# =========================
# Trading Loop (5s candles)
# =========================
def scan_loop():
    global running, error_message
    scan_log.clear()
    last_candle_ts = {p: 0 for p in PAIRS}

    while running:
        prices = fetch_all_prices()
        now = int(time.time())
        balances = get_wallet_balances()

        for pair in PAIRS:
            info = prices.get(pair)
            if not info:
                continue

            # Tick ingest
            price = info["price"]
            tick_logs[pair].append((now, price))
            if len(tick_logs[pair]) > 10000:
                tick_logs[pair] = tick_logs[pair][-10000:]

            # Candle aggregation
            aggregate_candles(pair, CANDLE_INTERVAL_SEC)
            C = candle_logs[pair]
            if not C:
                continue

            # Only act on a *new* closed candle
            if C[-1]["start"] == last_candle_ts[pair]:
                # While waiting for a new bar, still check TP for open longs
                pos = positions.get(pair)
                if pos and pos["side"] == "BUY":
                    if price >= pos["tp"]:
                        qty = precise_sell_qty(pair, pos["qty"])
                        if qty > 0:
                            res = place_order(pair, "SELL", qty)
                            scan_log.append(f"{ist_now()} | {pair} | TP HIT: SELL {qty} @ {price} (TP {pos['tp']}) | {res}")
                            trade_log.append({
                                "time": ist_now(), "pair": pair, "side": "SELL",
                                "entry": price, "msg": "TP +0.2%", "qty": qty, "order_result": res
                            })
                            positions[pair] = None
                            if "error" in res: error_message = res["error"]
                continue

            # New closed candle
            last_candle_ts[pair] = C[-1]["start"]

            # Generate signal on the *closed* bar
            sig = compute_signal(C)

            # Manage open long: reverse-signal = SL
            pos = positions.get(pair)
            if pos and pos["side"] == "BUY":
                # TP check (again on bar close)
                if price >= pos["tp"]:
                    qty = precise_sell_qty(pair, pos["qty"])
                    if qty > 0:
                        res = place_order(pair, "SELL", qty)
                        scan_log.append(f"{ist_now()} | {pair} | TP HIT: SELL {qty} @ {price} (TP {pos['tp']}) | {res}")
                        trade_log.append({
                            "time": ist_now(), "pair": pair, "side": "SELL",
                            "entry": price, "msg": "TP +0.2%", "qty": qty, "order_result": res
                        })
                        positions[pair] = None
                        if "error" in res: error_message = res["error"]
                    continue  # position closed by TP

                # Reverse signal (SL as per reverse trade)
                if sig == "SELL":
                    qty = precise_sell_qty(pair, pos["qty"])
                    if qty > 0:
                        res = place_order(pair, "SELL", qty)
                        scan_log.append(f"{ist_now()} | {pair} | SL (Reverse): SELL {qty} @ {price} | {res}")
                        trade_log.append({
                            "time": ist_now(), "pair": pair, "side": "SELL",
                            "entry": price, "msg": "SL via reverse signal", "qty": qty, "order_result": res
                        })
                        positions[pair] = None
                        if "error" in res: error_message = res["error"]
                    continue  # handled SL

            # No open long or it was closed above; evaluate fresh actions
            if sig == "BUY":
                # Enter new long
                usdt = balances.get("USDT", 0.0)
                if price > 0 and usdt > 0:
                    raw_qty = (BUY_USDT_ALLOC * usdt) / price
                    qty = clamp_buy_qty(pair, raw_qty)
                    if qty > 0:
                        res = place_order(pair, "BUY", qty)
                        tp = round(price * (1.0 + TP_PCT), 6)
                        positions[pair] = {"side": "BUY", "qty": qty, "entry": price, "tp": tp}
                        scan_log.append(f"{ist_now()} | {pair} | BUY {qty} @ {price} | TP {tp} (+0.2%) | {res}")
                        trade_log.append({
                            "time": ist_now(), "pair": pair, "side": "BUY",
                            "entry": price, "msg": "BUY entry", "qty": qty, "order_result": res
                        })
                        if "error" in res: error_message = res["error"]
                else:
                    scan_log.append(f"{ist_now()} | {pair} | BUY sig but no USDT or price invalid")

            elif sig == "SELL":
                # For spot, SELL = liquidate wallet coin (no short opens)
                coin = pair[:-4]
                bal_qty = balances.get(coin, 0.0)
                qty = precise_sell_qty(pair, bal_qty)
                if qty > 0:
                    res = place_order(pair, "SELL", qty)
                    scan_log.append(f"{ist_now()} | {pair} | SELL {qty} @ {price} (Spot liquidation) | {res}")
                    trade_log.append({
                        "time": ist_now(), "pair": pair, "side": "SELL",
                        "entry": price, "msg": "SELL (spot liquidation)", "qty": qty, "order_result": res
                    })
                    positions[pair] = None  # ensure no lingering pos
                    if "error" in res: error_message = res["error"]
                else:
                    scan_log.append(f"{ist_now()} | {pair} | SELL sig but no {coin} balance ≥ min qty")

            else:
                scan_log.append(f"{ist_now()} | {pair} | No Signal")

        # Trim logs to keep memory bounded
        if len(scan_log) > 600: scan_log[:] = scan_log[-600:]
        if len(trade_log) > 200: trade_log[:] = trade_log[-200:]

        status["msg"], status["last"] = "Running", ist_now()
        time.sleep(POLL_SEC)

    status["msg"] = "Idle"

# =========================
# Routes
# =========================
@app.route("/")
def index():
    return render_template("index.html")

@app.route("/start", methods=["POST"])
def start():
    global running
    if not running:
        running = True
        t = threading.Thread(target=scan_loop, daemon=True)
        t.start()
    return jsonify({"status": "started"})

@app.route("/stop", methods=["POST"])
def stop():
    global running
    running = False
    return jsonify({"status": "stopped"})

@app.route("/status")
def get_status():
    balances = get_wallet_balances()
    coins = {pair[:-4]: balances.get(pair[:-4], 0.0) for pair in PAIRS}
    pos_view = {p: positions[p] for p in PAIRS if positions[p]}
    return jsonify({
        "status": status["msg"],
        "last": status["last"],
        "usdt": balances.get("USDT", 0.0),
        "coins": coins,
        "positions": pos_view,
        "trades": trade_log[-12:][::-1],
        "scans": scan_log[-40:][::-1],
        "error": error_message
    })

@app.route("/ping")
def ping():
    return "pong"

# =========================
# Main
# =========================
if __name__ == "__main__":
    # Program ID: ABCD
    app.run(host="0.0.0.0", port=10000)
