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

# Candle interval (seconds). 60 = 1m candles.
CANDLE_INTERVAL_SEC = 60

# PSAR parameters
PSAR_STEP = 0.02
PSAR_MAX  = 0.20

# Poll speed
POLL_SEC = 1.0

# =========================
# Time helpers (IST)
# =========================
IST = timezone("Asia/Kolkata")
def ist_now(): return datetime.now(IST).strftime("%Y-%m-%d %H:%M:%S")

# =========================
# State
# =========================
tick_logs   = {p: [] for p in PAIRS}   # [(ts, price)]
candle_logs = {p: [] for p in PAIRS}   # dicts: open/high/low/close/volume/start
scan_log    = []                       # rolling text logs for UI
trade_log   = []                       # entries & exits
running     = False
status      = {"msg": "Idle", "last": ""}
error_message = ""

# Optional position hint (not required for SELL-anyway behavior)
positions = {p: None for p in PAIRS}   # {"side":"BUY","qty":float,"entry":float}

session = requests.Session()

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
            scan_log.append(f"{ist_now()} | BAL_ERR: {r.status_code} {r.text[:120]}")
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
            scan_log.append(f"{ist_now()} | PRICE_ERR: {r.status_code} {r.text[:120]}")
    except Exception as e:
        scan_log.append(f"{ist_now()} | PRICE_ERR: {e}")
    return {}

def clamp_buy_qty(pair, qty):
    """Clamp for BUY sizing (floor to step, ensure >= min)."""
    rule = PAIR_RULES.get(pair, {"precision": 6, "min_qty": 0.0})
    precision = int(rule["precision"])
    min_qty = float(rule["min_qty"])
    step = (10 ** (-precision)) if precision > 0 else 1.0
    qty = (int(max(qty, 0.0) / step)) * step
    if qty < min_qty:
        return 0.0
    return round(qty, precision)

def precise_sell_qty(pair, balance_qty):
    """
    SELL-specific clamp:
    - Floors to step so qty <= balance
    - Enforces min_qty
    """
    rule = PAIR_RULES.get(pair, {"precision": 6, "min_qty": 0.0})
    precision = int(rule["precision"])
    min_qty = float(rule["min_qty"])
    step = (10 ** (-precision)) if precision > 0 else 1.0
    qty = (int(max(balance_qty, 0.0) / step)) * step
    if qty < min_qty:
        return 0.0
    return round(qty, precision)

def place_order(pair, side, qty):
    """Order protocol unchanged: market order with total_quantity."""
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
# Indicators
# =========================
def sma(values, period):
    if len(values) < period: return None
    return sum(values[-period:]) / period

def ema(values, period):
    if len(values) < period: return None
    k = 2 / (period + 1)
    e = float(values[0])
    for v in values[1:]:
        e = float(v) * k + e * (1 - k)
    return e

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

def psar_series(highs, lows, step=0.02, max_step=0.2):
    n = len(highs)
    if n < 2: return [], []
    sar = [0.0] * n; bull = [False] * n
    uptrend = highs[1] >= highs[0]
    bull[1] = uptrend
    ep = highs[1] if uptrend else lows[1]
    sar[1] = lows[0] if uptrend else highs[0]
    af = step
    for i in range(2, n):
        prev_sar = sar[i-1]; prev_up = bull[i-1]
        sar_i = prev_sar + af * (ep - prev_sar)
        if prev_up:
            sar_i = min(sar_i, lows[i-1], lows[i-2] if i-2 >= 0 else lows[i-1])
        else:
            sar_i = max(sar_i, highs[i-1], highs[i-2] if i-2 >= 0 else highs[i-1])
        if prev_up:
            if lows[i] < sar_i:
                bull[i] = False; sar[i] = ep; af = step; ep = lows[i]
            else:
                bull[i] = True; sar[i] = sar_i
                if highs[i] > ep:
                    ep = highs[i]; af = min(max_step, af + step)
        else:
            if highs[i] > sar_i:
                bull[i] = True; sar[i] = ep; af = step; ep = highs[i]
            else:
                bull[i] = False; sar[i] = sar_i
                if lows[i] < ep:
                    ep = lows[i]; af = min(max_step, af + step)
    sar[0] = sar[1]; bull[0] = bull[1]
    return sar, bull

# =========================
# Candle aggregation (1m)
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
            candle["low"] = min(candle["low"], price)
            candle["close"] = price
            candle["volume"] += 1
    if candle: candles.append(candle)
    candle_logs[pair] = candles[-500:]  # keep history

# =========================
# Signal (SMA5/10 + RSI14 + PSAR flip as bias)
# =========================
def gen_signal(pair):
    C = candle_logs[pair]
    if len(C) < 20:
        return None
    closes = [c["close"] for c in C]
    highs  = [c["high"]  for c in C]
    lows   = [c["low"]   for c in C]
    curr   = C[-1]

    sma5  = sma(closes, 5)
    sma10 = sma(closes, 10)
    rsi14 = rsi_wilder_from_closes(closes[-15:], 14)
    sar, bull = psar_series(highs, lows, step=PSAR_STEP, max_step=PSAR_MAX)

    if any(x is None for x in [sma5, sma10, rsi14]) or len(sar) < 2:
        return None

    crossed_up   = (sma5 > sma10) or (closes[-1] > sma10)
    crossed_down = (sma5 < sma10) or (closes[-1] < sma10)
    sar_bull     = bool(bull[-1])
    sar_bear     = not sar_bull

    # BUY
    if crossed_up and sar_bull and rsi14 >= 48.0:
        return {"side": "BUY", "entry": curr["close"], "msg": "BUY: SMA(5>10)/Close>SMA10 + SAR bull + RSI>=48"}

    # SELL
    if crossed_down and sar_bear and rsi14 <= 52.0:
        return {"side": "SELL", "entry": curr["close"], "msg": "SELL: SMA(5<10)/Close<SMA10 + SAR bear + RSI<=52"}

    return None

# =========================
# Trading Loop (BUY when USDT; SELL when coin exists)
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
            if len(tick_logs[pair]) > 5000:
                tick_logs[pair] = tick_logs[pair][-5000:]

            # Candle aggregation
            aggregate_candles(pair, CANDLE_INTERVAL_SEC)
            last_candle = candle_logs[pair][-1] if candle_logs[pair] else None

            # Act only when a *new* candle has closed
            if last_candle and last_candle["start"] != last_candle_ts[pair]:
                last_candle_ts[pair] = last_candle["start"]
                signal = gen_signal(pair)

                if signal:
                    side  = signal["side"]
                    entry = signal["entry"]
                    coin  = pair[:-4]

                    if side == "SELL":
                        # ✅ SELL even if we never bought in-code: if wallet has that coin, SELL it
                        raw_qty = balances.get(coin, 0.0)
                        qty = precise_sell_qty(pair, raw_qty)
                        if qty > 0:
                            res = place_order(pair, "SELL", qty)
                            scan_log.append(f"{ist_now()} | {pair} | SELL qty={qty} @ {entry} | {res} | {signal['msg']}")
                            trade_log.append({
                                "time": ist_now(), "pair": pair, "side": "SELL",
                                "entry": entry, "msg": signal["msg"], "qty": qty, "order_result": res
                            })
                            if "error" in res:
                                error_message = res["error"]
                            positions[pair] = None
                            balances = get_wallet_balances()  # refresh after trade
                        else:
                            scan_log.append(f"{ist_now()} | {pair} | SELL signal, but {coin} balance < min qty")

                    else:  # BUY
                        # Only buy if we have USDT available
                        usdt = balances.get("USDT", 0.0)
                        raw_qty = (0.35 * usdt) / entry if entry else 0.0
                        qty = clamp_buy_qty(pair, raw_qty)
                        if qty > 0:
                            res = place_order(pair, "BUY", qty)
                            scan_log.append(f"{ist_now()} | {pair} | BUY qty={qty} @ {entry} | {res} | {signal['msg']}")
                            trade_log.append({
                                "time": ist_now(), "pair": pair, "side": "BUY",
                                "entry": entry, "msg": signal["msg"], "qty": qty, "order_result": res
                            })
                            if "error" in res:
                                error_message = res["error"]
                            else:
                                positions[pair] = {"side": "BUY", "qty": qty, "entry": entry}
                            balances = get_wallet_balances()  # refresh after trade
                        else:
                            scan_log.append(f"{ist_now()} | {pair} | BUY signal, qty=0 after clamp; skip")
                else:
                    scan_log.append(f"{ist_now()} | {pair} | No Signal")

        # Trim logs a bit to prevent runaway memory
        if len(scan_log) > 400: scan_log[:] = scan_log[-400:]
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
