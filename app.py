import os, time, json, hmac, hashlib, threading, requests
from flask import Flask, jsonify, request, render_template

# ======================================================
# CONFIG
# ======================================================
API_KEY = os.environ.get("API_KEY", "")
API_SECRET = os.environ.get("API_SECRET", "")
BASE_URL = "https://api.coindcx.com"

APP_BASE_URL = os.environ.get("APP_BASE_URL", "")
KEEPALIVE_SEC = int(os.environ.get("KEEPALIVE_SEC", "60"))

SCAN_INTERVAL = 5

# ======================================================
# APP
# ======================================================
app = Flask(__name__)

# ======================================================
# GLOBAL STATE
# ======================================================
running = False
MODE = "PAPER"   # PAPER | LIVE

PAIRS = ["BTCUSDT","ETHUSDT","SOLUSDT","BNBUSDT","XRPUSDT"]

prices = {}
price_history = {}
positions = {}
orders = []
scan_log = []

pnl = {"realized":0.0,"unrealized":0.0}

SETTINGS = {
    "tp_pct": 0.02,
    "sl_pct": 0.005,
    "risk_per_trade": 0.1,
    "breakout_window": 20
}

# ======================================================
# AUTH UTILS
# ======================================================
def sign(payload:str):
    return hmac.new(API_SECRET.encode(), payload.encode(), hashlib.sha256).hexdigest()

def api_post(path, payload):
    body = json.dumps(payload)
    headers = {
        "X-AUTH-APIKEY": API_KEY,
        "X-AUTH-SIGNATURE": sign(body),
        "Content-Type": "application/json"
    }
    return requests.post(BASE_URL + path, headers=headers, data=body, timeout=15)

# ======================================================
# KEEPALIVE SYSTEM
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
# EXCHANGE (REAL)
# ======================================================
def fetch_prices():
    r = requests.get(BASE_URL + "/exchange/ticker", timeout=10)
    data = {}
    if r.ok:
        for t in r.json():
            if t["market"] in PAIRS:
                data[t["market"]] = {
                    "price": float(t["last_price"]),
                    "ts": int(time.time())
                }
    return data

def fetch_balances():
    payload = {"timestamp": int(time.time()*1000)}
    r = api_post("/exchange/v1/users/balances", payload)
    data = {}
    if r.ok:
        for b in r.json():
            data[b["currency"]] = float(b["balance"])
    return data

def place_spot_order(pair, side, qty):
    payload = {
        "side": side.lower(),   # buy / sell
        "order_type": "market_order",
        "market": pair,
        "total_quantity": qty,
        "timestamp": int(time.time()*1000)
    }
    r = api_post("/exchange/v1/orders/create", payload)
    if r.ok:
        return r.json()
    else:
        return {"error": r.text}

# ======================================================
# BREAKOUT ENGINE
# ======================================================
def detect_breakout(pair, price):
    hist = price_history.setdefault(pair, [])
    hist.append(price)

    if len(hist) < SETTINGS["breakout_window"]:
        return None

    window = hist[-SETTINGS["breakout_window"]:]
    high = max(window[:-1])
    low = min(window[:-1])

    if price > high:
        return "UP"
    if price < low:
        return "DOWN"
    return None

# ======================================================
# POSITION ENGINE
# ======================================================
def position_size(usdt_balance, price):
    risk_amount = usdt_balance * SETTINGS["risk_per_trade"]
    qty = risk_amount / price
    return round(qty, 6)

def open_position(pair, side, price):
    if pair in positions:
        return

    balances = fetch_balances()
    usdt = balances.get("USDT",0)
    if usdt <= 10:
        scan_log.append("LOW BALANCE - NO TRADE")
        return

    qty = position_size(usdt, price)

    order_data = None
    if MODE == "LIVE":
        order_data = place_spot_order(pair, side, qty)
    else:
        order_data = {"mode":"PAPER","pair":pair,"side":side,"qty":qty}

    orders.append(order_data)

    positions[pair] = {
        "pair": pair,
        "side": side,
        "entry": price,
        "qty": qty,
        "tp": price * (1 + SETTINGS["tp_pct"] if side=="BUY" else 1 - SETTINGS["tp_pct"]),
        "sl": price * (1 - SETTINGS["sl_pct"] if side=="BUY" else 1 + SETTINGS["sl_pct"]),
        "time": int(time.time()),
        "mode": MODE
    }

# ======================================================
# EXIT ENGINE
# ======================================================
def monitor_exits(price_map):
    global pnl
    for pair in list(positions.keys()):
        pos = positions[pair]
        price = price_map[pair]["price"]

        exit_signal = False
        profit = 0

        if pos["side"] == "BUY":
            if price >= pos["tp"] or price <= pos["sl"]:
                profit = (price - pos["entry"]) * pos["qty"]
                exit_signal = True
        else:
            if price <= pos["tp"] or price >= pos["sl"]:
                profit = (pos["entry"] - price) * pos["qty"]
                exit_signal = True

        if exit_signal:
            if MODE == "LIVE":
                place_spot_order(pair, "SELL" if pos["side"]=="BUY" else "BUY", pos["qty"])

            pnl["realized"] += profit
            scan_log.append(f"EXIT {pair} PNL {round(profit,4)}")
            del positions[pair]

# ======================================================
# MAIN LOOP
# ======================================================
def scan_loop():
    global running
    while running:
        try:
            price_map = fetch_prices()
            if not price_map:
                time.sleep(2)
                continue

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
            scan_log.append(f"ENGINE ERROR: {str(e)}")
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
    return jsonify({"ok":True})

@app.route("/stop", methods=["POST"])
def stop():
    global running
    running = False
    return jsonify({"ok":True})

@app.route("/mode", methods=["POST"])
def mode():
    global MODE
    MODE = request.json.get("mode","PAPER")
    return jsonify({"ok":True,"mode":MODE})

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
    threading.Thread(target=keepalive_loop, daemon=True).start()
    app.run(host="0.0.0.0", port=int(os.environ.get("PORT",5000)))