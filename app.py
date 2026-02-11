import os, time, json, hmac, hashlib, threading, requests
from flask import Flask, request, jsonify

# ================== CONFIG ==================
API_KEY = os.environ.get("API_KEY", "")
API_SECRET = os.environ.get("API_SECRET", "")
BASE_URL = "https://api.coindcx.com"

APP_BASE_URL = os.environ.get("APP_BASE_URL")  # e.g. https://coin-xxxx.onrender.com
KEEPALIVE_SEC = int(os.environ.get("KEEPALIVE_SEC", "300"))

app = Flask(__name__)

# ================== STATE ==================
MODE = "PAPER"   # PAPER / LIVE
running = True

state = {
    "running": True,
    "server_time": 0,
    "heartbeat": 0,
    "last_scan": 0,
    "last_keepalive": 0,
    "pnl": {
        "realized": 0.0,
        "unrealized": 0.0
    }
}

positions = {}
orders = []

# ================== MARKETS ==================
PAIRS = ["BTCUSDT", "ETHUSDT", "BNBUSDT", "XRPUSDT", "SOLUSDT"]

PAIR_PRECISION = {
    "BTCUSDT": 6,
    "ETHUSDT": 5,
    "BNBUSDT": 4,
    "XRPUSDT": 4,
    "SOLUSDT": 3
}

MIN_NOTIONAL = {
    "BTCUSDT": 5,
    "ETHUSDT": 5,
    "BNBUSDT": 5,
    "XRPUSDT": 5,
    "SOLUSDT": 5
}

RISK_PCT = 0.05     # 5% per trade
TP_PCT = 0.02       # 2%
SL_PCT = 0.01       # 1%
CANDLE_INTERVAL = 60  # seconds

# ================== UTIL ==================
def sign(payload):
    payload_str = json.dumps(payload, separators=(',', ':'))
    return hmac.new(API_SECRET.encode(), payload_str.encode(), hashlib.sha256).hexdigest()

def headers(payload):
    return {
        "X-AUTH-APIKEY": API_KEY,
        "X-AUTH-SIGNATURE": sign(payload),
        "Content-Type": "application/json"
    }

def now():
    return int(time.time())

# ================== KEEP ALIVE ==================
def _keepalive_ping():
    if not APP_BASE_URL: 
        return
    try:
        requests.get(f"{APP_BASE_URL}/ping", timeout=10)
        state["last_keepalive"] = int(time.time())
    except:
        pass

# ================== EXCHANGE ==================
def get_wallet():
    payload = {"timestamp": int(time.time()*1000)}
    r = requests.post(f"{BASE_URL}/exchange/v1/users/balances",
                      headers=headers(payload), json=payload, timeout=15)
    data = r.json()
    balances = {}
    for b in data:
        balances[b["currency"]] = float(b["available_balance"])
    return balances

def fetch_price(pair):
    r = requests.get(f"{BASE_URL}/exchange/ticker", timeout=10)
    data = r.json()
    for m in data:
        if m["market"] == pair:
            return float(m["last_price"])
    return None

def format_qty(qty, pair):
    prec = PAIR_PRECISION.get(pair, 6)
    return float(f"{qty:.{prec}f}")

def calc_qty(usdt, price, pair):
    risk = usdt * RISK_PCT
    if risk < MIN_NOTIONAL[pair]:
        return 0
    raw = risk / price
    return format_qty(raw, pair)

def place_order(pair, side, qty):
    payload = {
        "side": side.lower(),
        "order_type": "market",
        "market": pair,
        "quantity": str(qty),
        "timestamp": int(time.time()*1000)
    }

    if MODE == "PAPER":
        return {"paper": True, "payload": payload}

    r = requests.post(
        f"{BASE_URL}/exchange/v1/orders/create",
        headers=headers(payload),
        json=payload,
        timeout=20
    )
    return r.json()

# ================== STRATEGY ==================
def breakout(price, hist):
    if len(hist) < 20:
        return False
    high = max(hist[-20:])
    return price > high

def monitor_exits(prices):
    for pair, pos in list(positions.items()):
        price = prices.get(pair)
        if not price:
            continue

        if pos["side"] == "BUY":
            if price >= pos["tp"] or price <= pos["sl"]:
                qty = pos["qty"]
                res = place_order(pair, "SELL", qty)

                pnl = (price - pos["entry"]) * pos["qty"]
                state["pnl"]["realized"] += pnl

                orders.append({
                    "time": now(),
                    "pair": pair,
                    "side": "EXIT",
                    "qty": qty,
                    "exit_price": price,
                    "pnl": pnl,
                    "mode": MODE,
                    "exchange": res
                })

                del positions[pair]

# ================== MAIN LOOP ==================
price_hist = {p: [] for p in PAIRS}

def scan_loop():
    global running

    while True:

        if not running:
            time.sleep(2)
            continue

        # system state
        state["heartbeat"] = int(time.time())
        state["server_time"] = int(time.time())

        # keepalive
        if APP_BASE_URL and (time.time() - state["last_keepalive"] > KEEPALIVE_SEC):
            _keepalive_ping()

        prices = {}
        for p in PAIRS:
            pr = fetch_price(p)
            if pr:
                prices[p] = pr
                price_hist[p].append(pr)
                if len(price_hist[p]) > 200:
                    price_hist[p] = price_hist[p][-200:]

        # exits
        monitor_exits(prices)

        # unrealized pnl
        unreal = 0.0
        for pair, pos in positions.items():
            if pair in prices:
                unreal += (prices[pair] - pos["entry"]) * pos["qty"]
        state["pnl"]["unrealized"] = unreal

        # wallet
        wallet = get_wallet()
        usdt = wallet.get("USDT", 0)

        # entries
        for pair in PAIRS:
            if pair in positions:
                continue

            price = prices.get(pair)
            if not price:
                continue

            if breakout(price, price_hist[pair]):
                qty = calc_qty(usdt, price, pair)
                if qty <= 0:
                    continue

                side = "BUY"
                res = place_order(pair, side, qty)

                entry = price
                positions[pair] = {
                    "pair": pair,
                    "side": side,
                    "qty": qty,
                    "entry": entry,
                    "tp": entry * (1 + TP_PCT),
                    "sl": entry * (1 - SL_PCT),
                    "time": now(),
                    "mode": MODE
                }

                orders.append({
                    "time": now(),
                    "pair": pair,
                    "side": side,
                    "qty": qty,
                    "entry_price": entry,
                    "mode": MODE,
                    "exchange": res
                })

        state["last_scan"] = int(time.time())
        time.sleep(CANDLE_INTERVAL)

# ================== API ==================
@app.route("/")
def ui():
    return """
    <html>
    <head>
    <title>Trading Engine</title>
    <style>
    body{background:#0b0b0b;color:#00ffcc;font-family:monospace;padding:20px}
    button{padding:10px;margin:5px;background:#111;color:#0f0;border:1px solid #0f0}
    pre{background:#050505;padding:10px;max-height:70vh;overflow:auto}
    </style>
    </head>
    <body>
    <h2>Institutional Breakout Engine</h2>

    <button onclick="setMode('PAPER')">Paper</button>
    <button onclick="setMode('LIVE')">Live</button>
    <button onclick="toggle(true)">Start</button>
    <button onclick="toggle(false)">Stop</button>

    <pre id="out"></pre>

    <script>
    function load(){
        fetch('/status').then(r=>r.json()).then(d=>{
            document.getElementById('out').innerText = JSON.stringify(d,null,2);
        });
    }
    function setMode(m){
        fetch('/mode',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({mode:m})})
    }
    function toggle(v){
        fetch('/control',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({run:v})})
    }
    setInterval(load,2000)
    </script>
    </body>
    </html>
    """

@app.route("/ping")
def ping():
    return "pong"

@app.route("/status")
def status():
    return jsonify({
        "mode": MODE,
        "state": state,
        "positions": positions,
        "orders": orders[-50:]
    })

@app.route("/mode", methods=["POST"])
def set_mode():
    global MODE
    MODE = request.json.get("mode", "PAPER")
    return jsonify({"ok": True, "mode": MODE})

@app.route("/control", methods=["POST"])
def control():
    global running
    running = request.json.get("run", True)
    state["running"] = running
    return jsonify({"running": running})

# ================== BOOT ==================
if __name__ == "__main__":
    threading.Thread(target=scan_loop, daemon=True).start()
    app.run(host="0.0.0.0", port=int(os.environ.get("PORT", 10000)))