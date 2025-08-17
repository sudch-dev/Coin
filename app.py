import os
import time
import threading
import hmac
import hashlib
import requests
import json
from datetime import datetime, timedelta
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

# Trade universe
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

# =========================
# Time helpers (IST)
# =========================
IST = timezone("Asia/Kolkata")
def ist_now(): return datetime.now(IST).strftime("%Y-%m-%d %H:%M:%S")
def ist_date(): return datetime.now(IST).strftime("%Y-%m-%d")
def ist_yesterday(): return (datetime.now(IST) - timedelta(days=1)).strftime("%Y-%m-%d")

# =========================
# State
# =========================
tick_logs = {p: [] for p in PAIRS}     # [(ts, price)]
candle_logs = {p: [] for p in PAIRS}   # dicts: open/high/low/close/volume/start
scan_log = []                          # text logs for UI
trade_log = []                         # recent trades (entries only)
running = False
status = {"msg": "Idle", "last": ""}
error_message = ""

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
        r = requests.post(f"{BASE_URL}/exchange/v1/users/balances", headers=headers, data=payload, timeout=10)
        if r.ok:
            for b in r.json():
                balances[b["currency"]] = float(b["balance"])
    except Exception as e:
        scan_log.append(f"{ist_now()} | BAL_ERR: {e}")
    return balances

def fetch_all_prices():
    try:
        r = requests.get(f"{BASE_URL}/exchange/ticker", timeout=10)
        if r.ok:
            now = int(time.time())
            return {item["market"]: {"price": float(item["last_price"]), "ts": now}
                    for item in r.json() if item["market"] in PAIRS}
    except Exception as e:
        scan_log.append(f"{ist_now()} | PRICE_ERR: {e}")
    return {}

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
    candle_logs[pair] = candles[-500:]

def clamp_qty(pair, qty):
    rule = PAIR_RULES.get(pair, {"precision": 6, "min_qty": 0.0})
    precision = int(rule["precision"])
    min_qty = float(rule["min_qty"])
    qty = max(qty, min_qty)
    step = 10 ** (-precision)
    qty = (int(qty / step)) * step
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
        r = requests.post(f"{BASE_URL}/exchange/v1/orders/create", headers=headers, data=body, timeout=10)
        return r.json()
    except Exception as e:
        return {"error": str(e)}

# =========================
# Indicators / Signal
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
    return round(100 - (100 / (1 + rs)), 2)

def stoch_kd(candles, k_period=14, smooth_k=3, d_period=3):
    """Stoch(14,3,3): %K = SMA(%K_raw,3); %D = SMA(%K,3). Returns (k, d, k_prev, d_prev)."""
    need = k_period + smooth_k + d_period
    if len(candles) < need: return None, None, None, None

    highs  = [c["high"] for c in candles]
    lows   = [c["low"]  for c in candles]
    closes = [c["close"] for c in candles]

    k_raw_series = []
    for i in range(k_period-1, len(candles)):
        hh = max(highs[i-k_period+1:i+1])
        ll = min(lows[i-k_period+1:i+1])
        denom = (hh - ll) if (hh - ll) != 0 else 1e-9
        k_raw = (closes[i] - ll) * 100.0 / denom
        k_raw_series.append(k_raw)

    if len(k_raw_series) < smooth_k + d_period: return None, None, None, None

    k_smooth = []
    for j in range(smooth_k-1, len(k_raw_series)):
        k_smooth.append(sum(k_raw_series[j-smooth_k+1:j+1]) / smooth_k)

    if len(k_smooth) < d_period + 1: return None, None, None, None

    d_smooth = []
    for j in range(d_period-1, len(k_smooth)):
        d_smooth.append(sum(k_smooth[j-d_period+1:j+1]) / d_period)

    k = round(k_smooth[-1], 2)
    d = round(d_smooth[-1], 2)
    k_prev = round(k_smooth[-2], 2) if len(k_smooth) >= 2 else None
    d_prev = round(d_smooth[-2], 2) if len(d_smooth) >= 2 else None
    return k, d, k_prev, d_prev

def pa_buy_sell_signal(pair):
    """
    1m entry trigger based on: SMA(5/10) cross + RSI(14) + candle bias + Stoch(14,3,3).
    Returns dict: {"side","entry","msg"} or None.
    """
    candles = candle_logs[pair]
    need = max(10, 15, 20)  # SMA10, RSI14, Stoch(14,3,3)
    if len(candles) < need:
        return None

    closes = [c["close"] for c in candles]
    curr   = candles[-1]

    # SMAs for cross detection
    sma5_now   = sma(closes, 5)
    sma10_now  = sma(closes, 10)
    sma5_prev  = sma(closes[:-1], 5)
    sma10_prev = sma(closes[:-1], 10)
    if None in (sma5_now, sma10_now, sma5_prev, sma10_prev):
        return None

    cross_up   = (sma5_prev <= sma10_prev) and (sma5_now > sma10_now)
    cross_down = (sma5_prev >= sma10_prev) and (sma5_now < sma10_now)

    # RSI(14) on last 15 closes
    rsi = rsi_wilder_from_closes(closes[-15:], period=14)

    # Stoch(14,3,3)
    k, d, k_prev, d_prev = stoch_kd(candles, k_period=14, smooth_k=3, d_period=3)

    # Candle bias
    bullish = curr["close"] > curr["open"]
    bearish = curr["close"] < curr["open"]

    # BUY
    buy_ok = (
        cross_up and
        rsi is not None and 50 < rsi < 70 and
        bullish and curr["close"] > sma10_now and
        k is not None and d is not None and k_prev is not None and d_prev is not None and
        (k_prev <= d_prev) and (k > d) and (k < 80)
    )
    if buy_ok:
        return {
            "side": "BUY",
            "entry": curr["close"],
            "msg": f"BUY 1m: SMA5>10 cross, RSI={rsi}, Stoch {k}>{d}, bullish"
        }

    # SELL
    sell_ok = (
        cross_down and
        rsi is not None and 30 < rsi < 50 and
        bearish and curr["close"] < sma10_now and
        k is not None and d is not None and k_prev is not None and d_prev is not None and
        (k_prev >= d_prev) and (k < d) and (k > 20)
    )
    if sell_ok:
        return {
            "side": "SELL",
            "entry": curr["close"],
            "msg": f"SELL 1m: SMA5<10 cross, RSI={rsi}, Stoch {k}<{d}, bearish"
        }

    return None

# =========================
# Trading Loop (entry-only)
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

            price = info["price"]
            tick_logs[pair].append((now, price))
            if len(tick_logs[pair]) > 5000:
                tick_logs[pair] = tick_logs[pair][-5000:]

            aggregate_candles(pair, CANDLE_INTERVAL_SEC)
            last_candle = candle_logs[pair][-1] if candle_logs[pair] else None

            # act only at the *close* of a new candle
            if last_candle and last_candle["start"] != last_candle_ts[pair]:
                last_candle_ts[pair] = last_candle["start"]
                signal = pa_buy_sell_signal(pair)

                if signal:
                    error_message = ""
                    entry = signal["entry"]
                    coin = pair[:-4]

                    # Sizing: BUY up to 30% USDT; SELL 100% coin balance
                    if signal["side"] == "BUY":
                        usdt = balances.get("USDT", 0.0)
                        raw_qty = (0.3 * usdt) / entry if entry else 0.0
                    else:
                        raw_qty = balances.get(coin, 0.0)

                    qty = clamp_qty(pair, raw_qty)
                    if qty <= 0:
                        scan_log.append(f"{ist_now()} | {pair} | Qty=0 after clamp — skip")
                        continue

                    # Place order (protocol unchanged)
                    res = place_order(pair, signal["side"], qty)

                    scan_log.append(f"{ist_now()} | {pair} | {signal['side']} qty={qty} @ {entry} | {res}")
                    trade_log.append({
                        "time": ist_now(),
                        "pair": pair,
                        "side": signal["side"],
                        "entry": entry,
                        "msg": signal["msg"],
                        "qty": qty,
                        "order_result": res
                    })

                    if "error" in res:
                        error_message = res["error"]
                else:
                    scan_log.append(f"{ist_now()} | {pair} | No Signal")

        status["msg"], status["last"] = "Running", ist_now()
        time.sleep(5)

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
    return jsonify({
        "status": status["msg"],
        "last": status["last"],
        "usdt": balances.get("USDT", 0.0),
        "coins": coins,
        "trades": trade_log[-10:][::-1],
        "scans": scan_log[-30:][::-1],
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
