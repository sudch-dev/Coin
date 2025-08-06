import os
import time
import threading
from datetime import datetime
from flask import Flask, render_template, jsonify, request
import requests

app = Flask(__name__)

API_KEY = os.environ.get("API_KEY")
API_SECRET = os.environ.get("API_SECRET").encode()
PAIR_LIST = [
    "BTCUSDT", "ETHUSDT", "XRPUSDT", "SHIBUSDT", "SOLUSDT",
    "DOGEUSDT", "ADAUSDT", "MATICUSDT", "BNBUSDT", "LTCUSDT"
]

FETCH_INTERVAL = 5       # seconds (for live price)
CANDLE_INTERVAL = 300    # seconds (5 minutes)

live_price_log = {pair: [] for pair in PAIR_LIST}
candle_log = {pair: [] for pair in PAIR_LIST}
trade_log = []
scan_log = []
running = False
status = {"msg": "Idle"}

def fetch_live_price(pair):
    try:
        r = requests.get(
            f"https://public.coindcx.com/market_data/current_price?pair={pair}",
            timeout=5
        )
        if r.ok and r.json().get("bid"):
            return float(r.json()["bid"])
    except Exception as e:
        return None

def build_candle(pair):
    """Aggregate last CANDLE_INTERVAL seconds into OHLC."""
    prices = live_price_log[pair]
    if not prices:
        return None
    o = prices[0][1]
    h = max(x[1] for x in prices)
    l = min(x[1] for x in prices)
    c = prices[-1][1]
    ts = prices[-1][0]
    return {"timestamp": ts, "open": o, "high": h, "low": l, "close": c}

def ema(closes, n):
    if len(closes) < n:
        return []
    alpha = 2 / (n + 1)
    result = []
    ema_val = sum(closes[:n]) / n
    result.append(ema_val)
    for price in closes[n:]:
        ema_val = (price - ema_val) * alpha + ema_val
        result.append(ema_val)
    return result

def bot_loop():
    global running, status
    candle_start = {pair: time.time() for pair in PAIR_LIST}
    while running:
        for pair in PAIR_LIST:
            now = time.time()
            # Fetch price
            price = fetch_live_price(pair)
            if price is not None:
                live_price_log[pair].append((now, price))
                scan_log.append(
                    {"time": datetime.fromtimestamp(now).strftime("%Y-%m-%d %H:%M:%S"),
                     "pair": pair, "price": price}
                )
                # Prune log (keep last 2*CANDLE_INTERVAL)
                live_price_log[pair] = [
                    (t, p) for (t, p) in live_price_log[pair]
                    if t >= now - 2 * CANDLE_INTERVAL
                ]
            # Build new candle every 5 minutes
            if now - candle_start[pair] >= CANDLE_INTERVAL:
                if live_price_log[pair]:
                    cndl = build_candle(pair)
                    candle_log[pair].append(cndl)
                    # Only keep last 50 candles
                    candle_log[pair] = candle_log[pair][-50:]
                    candle_start[pair] = now
                    # Run EMA logic and trigger trade
                    closes = [c["close"] for c in candle_log[pair] if c]
                    if len(closes) >= 11:
                        ema5 = ema(closes, 5)
                        ema10 = ema(closes, 10)
                        # Detect cross (simple logic)
                        if len(ema5) >= 2 and len(ema10) >= 2:
                            # Buy: EMA5 crosses above EMA10
                            if ema5[-2] < ema10[-2] and ema5[-1] > ema10[-1]:
                                trade_log.append({
                                    "time": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                                    "pair": pair, "side": "BUY", "price": closes[-1],
                                    "msg": "EMA(5) crossed above EMA(10)"
                                })
                                status["msg"] = f"BUY triggered for {pair} at {closes[-1]}"
                            # Sell: EMA5 crosses below EMA10
                            if ema5[-2] > ema10[-2] and ema5[-1] < ema10[-1]:
                                trade_log.append({
                                    "time": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                                    "pair": pair, "side": "SELL", "price": closes[-1],
                                    "msg": "EMA(5) crossed below EMA(10)"
                                })
                                status["msg"] = f"SELL triggered for {pair} at {closes[-1]}"
        status["msg"] = "Scanning pairs"
        time.sleep(FETCH_INTERVAL)
    status["msg"] = "Idle"

@app.route("/")
def index():
    return render_template("index.html")

@app.route("/start", methods=["POST"])
def start():
    global running
    if not running:
        running = True
        thread = threading.Thread(target=bot_loop)
        thread.daemon = True
        thread.start()
    return jsonify({"status": "started"})

@app.route("/stop", methods=["POST"])
def stop():
    global running
    running = False
    return jsonify({"status": "stopped"})

@app.route("/status")
def get_status():
    return jsonify({
        "status": status["msg"],
        "trades": trade_log[-10:][::-1],
        "scans": scan_log[-20:][::-1],
        "candles": {p: candle_log[p][-2:] for p in PAIR_LIST}
    })

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=10000)
