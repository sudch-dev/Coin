import os, time, json, hmac, hashlib, threading, requests
from flask import Flask, request, jsonify, render_template_string

# ================== CONFIG ==================
API_KEY = os.environ.get("API_KEY", "")
API_SECRET = os.environ.get("API_SECRET", "")
BASE_URL = "https://api.coindcx.com"

APP_BASE_URL = os.environ.get("APP_BASE_URL")  # https://coin-xxxx.onrender.com
KEEPALIVE_SEC = int(os.environ.get("KEEPALIVE_SEC", "120"))

app = Flask(__name__)

# ================== STATE ==================
MODE = "PAPER"   # PAPER / LIVE
running = True

STATE = {
    "running": False,
    "last_scan": None,
    "last_keepalive": None,
    "server_time": None,
    "heartbeat": None,
    "pnl": {"realized": 0.0, "unrealized": 0.0}
}

positions = {}
orders = []
scan_log = []

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

RISK_PCT = 0.05
TP_PCT = 0.02
SL_PCT = 0.01
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
        STATE["last_keepalive"] = now()
    except:
        pass

# ================== EXCHANGE ==================
def safe_json(resp):
    try:
        return resp.json()
    except:
        return {}

def get_wallet():
    balances = {}
    payload = {"timestamp": int(time.time()*1000)}
    try:
        r = requests.post(
            f"{BASE_URL}/exchange/v1/users/balances",
            headers=headers(payload),
            json=payload,
            timeout=15
        )
        data = safe_json(r)

        if isinstance(data, list):
            for b in data:
                balances[b["currency"]] = float(b.get("available_balance", 0))
        else:
            scan_log.append({"time": now(), "error": data})
    except Exception as e:
        scan_log.append({"time": now(), "wallet_error": str(e)})

    return balances

def fetch_price(pair):
    try:
        r = requests.get(f"{BASE_URL}/exchange/ticker", timeout=10)
        data = safe_json(r)
        if isinstance(data, list):
            for m in data:
                if m["market"] == pair:
                    return float(m["last_price"])
    except:
        pass
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
        return {"mode": "PAPER", "payload": payload, "status": "filled"}

    try:
        r = requests.post(
            f"{BASE_URL}/exchange/v1/orders/create",
            headers=headers(payload),
            json=payload,
            timeout=20
        )
        return safe_json(r)
    except Exception as e:
        return {"error": str(e)}

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

                pnl = (price - pos["entry"]) * qty
                STATE["pnl"]["realized"] += pnl

                orders.append({
                    "pair": pair,
                    "exit_price": price,
                    "qty": qty,
                    "side": "SELL",
                    "res": res,
                    "pnl": pnl,
                    "time": now()
                })
                del positions[pair]

# ================== MAIN LOOP ==================
price_hist = {p: [] for p in PAIRS}

def scan_loop():
    global running
    STATE["running"] = True

    while True:
        if not running:
            time.sleep(1)
            continue

        STATE["heartbeat"] = now()
        STATE["server_time"] = now()

        # keepalive
        if APP_BASE_URL and (
            not STATE["last_keepalive"] or now() - STATE["last_keepalive"] > KEEPALIVE_SEC
        ):
            _keepalive_ping()

        prices = {}
        for p in PAIRS:
            pr = fetch_price(p)
            if pr:
                prices[p] = pr
                price_hist[p].append(pr)
                if len(price_hist[p]) > 200:
                    price_hist[p] = price_hist[p][-200:]

        monitor_exits(prices)

        wallet = get_wallet()
        usdt = wallet.get("USDT", 0)

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

                res = place_order(pair, "BUY", qty)

                entry = price
                positions[pair] = {
                    "pair": pair,
                    "side": "BUY",
                    "qty": qty,
                    "entry": entry,
                    "tp": entry * (1 + TP_PCT),
                    "sl": entry * (1 - SL_PCT),
                    "time": now(),
                    "mode": MODE
                }

                orders.append({
                    "pair": pair,
                    "side": "BUY",
                    "qty": qty,
                    "price": entry,
                    "mode": MODE,
                    "res": res,
                    "time": now()
                })

        # unrealized PnL
        upnl = 0
        for pair, pos in positions.items():
            p = prices.get(pair)
            if p:
                upnl += (p - pos["entry"]) * pos["qty"]

        STATE["pnl"]["unrealized"] = upnl
        STATE["last_scan"] = now()

        time.sleep(CANDLE_INTERVAL)

# ================== API ==================
@app.route("/")
def ui():
    return render_template_string(index_html)

@app.route("/ping")
def ping():
    return "pong"

@app.route("/status")
def status():
    return jsonify({
        "mode": MODE,
        "state": STATE,
        "positions": positions,
        "orders": orders[-30:]
    })

@app.route("/mode", methods=["POST"])
def set_mode():
    global MODE
    MODE = request.json.get("mode", "PAPER")
    return jsonify({"ok": True, "mode": MODE})

@app.route("/control", methods=["POST"])
def control():
    global running
    running = bool(request.json.get("run", True))
    STATE["running"] = running
    return jsonify({"running": running})

# ================== UI ==================
index_html = """
<!DOCTYPE html>
<html>
<head>
<title>Institutional Breakout Engine</title>
<style>
body{background:#0b0b0b;color:#00ffd0;font-family:monospace;padding:20px}
h2{color:#00ffaa}
.panel{border:1px solid #00ffaa;padding:10px;margin:10px 0}
button{padding:8px 12px;margin:5px;background:#000;color:#00ffaa;border:1px solid #00ffaa}
pre{background:#050505;padding:10px;max-height:300px;overflow:auto}
.green{color:#00ff00}
.red{color:#ff4444}
</style>
</head>
<body>

<h2>Institutional Breakout Engine</h2>

<div class="panel">
<b>Controls</b><br>
<button onclick="setMode('PAPER')">Paper</button>
<button onclick="setMode('LIVE')">Live</button>
<button onclick="toggle(true)">Start</button>
<button onclick="toggle(false)">Stop</button>
</div>

<div class="panel" id="statusPanel"></div>

<div class="panel">
<b>Positions</b>
<pre id="pos"></pre>
</div>

<div class="panel">
<b>Orders</b>
<pre id="ord"></pre>
</div>

<script>
function load(){
 fetch('/status').then(r=>r.json()).then(d=>{
   document.getElementById('statusPanel').innerHTML = `
   Mode: <b>${d.mode}</b><br>
   Bot: <b>${d.state.running}</b><br>
   Server Time: ${new Date(d.state.server_time*1000).toLocaleTimeString()}<br>
   Heartbeat: ${new Date(d.state.heartbeat*1000).toLocaleTimeString()}<br>
   Last Scan: ${new Date(d.state.last_scan*1000).toLocaleTimeString()}<br>
   KeepAlive: ${d.state.last_keepalive ? new Date(d.state.last_keepalive*1000).toLocaleTimeString() : '--'}<br>
   PnL Realized: <span class="green">${d.state.pnl.realized.toFixed(4)}</span><br>
   PnL Unrealized: <span class="green">${d.state.pnl.unrealized.toFixed(4)}</span>
   `;

   document.getElementById('pos').innerText = JSON.stringify(d.positions,null,2);
   document.getElementById('ord').innerText = JSON.stringify(d.orders,null,2);
 });
}

function setMode(m){
 fetch('/mode',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({mode:m})})
}

function toggle(v){
 fetch('/control',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({run:v})})
}

setInterval(load,2000);
</script>

</body>
</html>
"""

# ================== BOOT ==================
if __name__ == "__main__":
    threading.Thread(target=scan_loop, daemon=True).start()
    app.run(host="0.0.0.0", port=int(os.environ.get("PORT", 10000)))