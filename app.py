import os
import time
import json
import threading
import requests
from flask import Flask, jsonify, request, render_template

# ======================================================
# CONFIG
# ======================================================
API_KEY = os.environ.get("API_KEY", "")
API_SECRET = os.environ.get("API_SECRET", "")
BASE_URL = "https://api.coindcx.com"

# Render / hosting keepalive config
APP_BASE_URL = os.environ.get("APP_BASE_URL", "")  # e.g. https://your-app.onrender.com
KEEPALIVE_SEC = int(os.environ.get("KEEPALIVE_SEC", "60"))

SCAN_INTERVAL = 5   # seconds
CANDLE_INTERVAL = 60

# ======================================================
# APP
# ======================================================
app = Flask(__name__)

# ======================================================
# GLOBAL STATE
# ======================================================
running = False
MODE = "PAPER"   # PAPER | LIVE

PAIRS = [
    "BTCUSDT", "ETHUSDT", "SOLUSDT",
    "BNBUSDT", "XRPUSDT", "ADAUSDT", "DOGEUSDT"
]

prices = {}
price_history = {}
positions = {}
orders = []
scan_log = []

pnl = {
    "realized": 0.0,
    "unrealized": 0.0
}

SETTINGS = {
    "tp_pct": 0.01,
    "sl_pct": 0.005,
    "risk_per_trade": 0.01,
    "breakout_window": 20
}

# ======================================================
# KEEPALIVE SYSTEM (RENDER SAFE)
# ======================================================
_last_keepalive = 0

def _keepalive_ping():
    if not APP_BASE_URL:
        return
    try:
        requests.get(f"{APP_BASE_URL}/ping", timeout=5)
    except:
        pass

def keepalive_loop():
    global _last_keepalive
    while True:
        now = time.time()
        if now - _last_keepalive >= KEEPALIVE_SEC:
            _keepalive_ping()
            _last_keepalive = now
        time.sleep(5)

# ======================================================
# EXCHANGE ABSTRACTION (mock now, live later)
# ======================================================
def fetch_all_prices():
    data = {}
    for p in PAIRS:
        if p not in prices:
            prices[p] = 100.0

        # simulated movement (replace with real API/WebSocket)
        drift = (0.0005 - 0.001 * ((int(time.time()) % 2)))
        prices[p] = prices[p] * (1 + drift)

        data[p] = {
            "price": round(prices[p], 4),
            "ts": int(time.time())
        }
    return data

def place_order(pair, side, qty, price, mode="PAPER"):
    order = {
        "id": f"ORD-{int(time.time()*1000)}",
        "pair": pair,
        "side": side,
        "qty": qty,
        "price": price,
        "mode": mode,
        "time": int(time.time()),
        "status": "FILLED"
    }
    orders.append(order)
    return order

# ======================================================
# BREAKOUT ENGINE (REAL LOGIC STRUCTURE)
# ======================================================
def detect_breakout(pair, price):
    hist = price_history.setdefault(pair, [])
    hist.append(price)

    if len(hist) < SETTINGS["breakout_window"]:
        return None

    window = hist[-SETTINGS["breakout_window"]:]
    high = max(window[:-1])
    low = min(window[:-1])

    # structure breakout
    if price > high:
        return "UP"
    if price < low:
        return "DOWN"

    return None

# ======================================================
# TRADE ENGINE
# ======================================================
def open_position(pair, side, price):
    if pair in positions:
        return

    qty = 1  # replace with risk engine later
    order = place_order(pair, side, qty, price, MODE)

    positions[pair] = {
        "pair": pair,
        "side": side,
        "entry": price,
        "qty": qty,
        "tp": price * (1 + SETTINGS["tp_pct"] if side == "BUY" else 1 - SETTINGS["tp_pct"]),
        "sl": price * (1 - SETTINGS["sl_pct"] if side == "BUY" else 1 + SETTINGS["sl_pct"]),
        "time": int(time.time()),
        "order_id": order["id"]
    }

def monitor_exits(price_map):
    global pnl
    for pair in list(positions.keys()):
        pos = positions[pair]
        price = price_map[pair]["price"]

        closed = False
        profit = 0

        if pos["side"] == "BUY":
            if price >= pos["tp"]:
                profit = (price - pos["entry"]) * pos["qty"]
                closed = True
            elif price <= pos["sl"]:
                profit = (price - pos["entry"]) * pos["qty"]
                closed = True
        else:
            if price <= pos["tp"]:
                profit = (pos["entry"] - price) * pos["qty"]
                closed = True
            elif price >= pos["sl"]:
                profit = (pos["entry"] - price) * pos["qty"]
                closed = True

        if closed:
            pnl["realized"] += profit
            scan_log.append(f"EXIT {pair} PNL {round(profit,4)}")
            del positions[pair]

# ======================================================
# MAIN SCAN LOOP
# ======================================================
def scan_loop():
    global running
    while running:
        try:
            price_map = fetch_all_prices()
            monitor_exits(price_map)

            for pair in PAIRS:
                price = price_map[pair]["price"]
                signal = detect_breakout(pair, price)

                if signal == "UP":
                    open_position(pair, "BUY", price)
                    scan_log.append(f"{pair} BREAKOUT UP @ {price}")

                elif signal == "DOWN":
                    open_position(pair, "SELL", price)
                    scan_log.append(f"{pair} BREAKOUT DOWN @ {price}")

            time.sleep(SCAN_INTERVAL)

        except Exception as e:
            scan_log.append(f"ERROR: {str(e)}")
            time.sleep(2)

# ======================================================
# ROUTES
# ======================================================
@app.route("/")
def index():
    return render_template("index.html")

@app.route("/ping")
def ping():
    return "pong"

@app.route("/status")
def status():
    return jsonify({
        "running": running,
        "mode": MODE,
        "positions": positions,
        "orders": orders[-50:],
        "pnl": pnl,
        "settings": SETTINGS
    })

@app.route("/start", methods=["POST"])
def start():
    global running
    if not running:
        running = True
        threading.Thread(target=scan_loop, daemon=True).start()
    return jsonify({"ok": True})

@app.route("/stop", methods=["POST"])
def stop():
    global running
    running = False
    return jsonify({"ok": True})

@app.route("/mode", methods=["POST"])
def mode():
    global MODE
    data = request.json
    MODE = data.get("mode", "PAPER")
    return jsonify({"ok": True, "mode": MODE})

@app.route("/orders")
def get_orders():
    return jsonify(orders)

@app.route("/positions")
def get_positions():
    return jsonify(positions)

@app.route("/log")
def get_log():
    return jsonify(scan_log[-200:])

# ======================================================
# BOOT
# ======================================================
if __name__ == "__main__":
    # Keepalive thread (Render-safe)
    threading.Thread(target=keepalive_loop, daemon=True).start()

    # Flask server
    app.run(host="0.0.0.0", port=int(os.environ.get("PORT", 5000)))