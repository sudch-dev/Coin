import os, time, json, hmac, hashlib, threading, requests
from flask import Flask, request, jsonify

# ================== CONFIG ==================
API_KEY = os.environ.get("API_KEY", "")
API_SECRET = os.environ.get("API_SECRET", "")
BASE_URL = "https://api.coindcx.com"

APP_BASE_URL = os.environ.get("APP_BASE_URL")
KEEPALIVE_SEC = int(os.environ.get("KEEPALIVE_SEC", "300"))

app = Flask(__name__)

# ================== STATE ==================
MODE = "PAPER"
running = True
engine_alive = False
last_heartbeat = 0
_last_keepalive = 0
last_scan = 0

positions = {}
orders = []
trade_history = []

realized_pnl = 0.0

# ================== MARKETS ==================
PAIRS = ["BTCUSDT", "ETHUSDT", "BNBUSDT", "XRPUSDT", "SOLUSDT"]

PAIR_PRECISION = {
    "BTCUSDT": 6,
    "ETHUSDT": 5,
    "BNBUSDT": 4,
    "XRPUSDT": 4,
    "SOLUSDT": 3
}

MIN_NOTIONAL = {p:5 for p in PAIRS}

RISK_PCT = 0.05
TP_PCT = 0.02
SL_PCT = 0.01
CANDLE_INTERVAL = 60

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
        balances[b["currency"]] = float(b.get("available_balance",0))
    return balances

def fetch_price(pair):
    r = requests.get(f"{BASE_URL}/exchange/ticker", timeout=10)
    data = r.json()
    for m in data:
        if m["market"] == pair:
            return float(m["last_price"])
    return None

def format_qty(qty, pair):
    return float(f"{qty:.{PAIR_PRECISION.get(pair,6)}f}")

def calc_qty(usdt, price, pair):
    risk = usdt * RISK_PCT
    if risk < MIN_NOTIONAL[pair]:
        return 0
    return format_qty(risk / price, pair)

def place_order(pair, side, qty):
    payload = {
        "side": side.lower(),
        "order_type": "market",
        "market": pair,
        "quantity": str(qty),
        "timestamp": int(time.time()*1000)
    }

    if MODE == "PAPER":
        return {"status":"paper", "order_id":f"PAPER-{now()}", "filled":True}

    r = requests.post(f"{BASE_URL}/exchange/v1/orders/create",
                      headers=headers(payload), json=payload, timeout=20)
    return r.json()

# ================== STRATEGY ==================
def breakout(price, hist):
    if len(hist) < 20:
        return False
    return price > max(hist[-20:])

# ================== EXIT ENGINE ==================
def monitor_exits(prices):
    global realized_pnl
    for pair, pos in list(positions.items()):
        price = prices.get(pair)
        if not price:
            continue

        hit_tp = price >= pos["tp"]
        hit_sl = price <= pos["sl"]

        if hit_tp or hit_sl:
            qty = pos["qty"]
            res = place_order(pair, "SELL", qty)

            pnl = (price - pos["entry"]) * qty
            realized_pnl += pnl

            trade_history.append({
                "pair":pair,
                "entry":pos["entry"],
                "exit":price,
                "qty":qty,
                "pnl":pnl,
                "mode":MODE,
                "time":now()
            })

            orders.append({
                "time":now(),
                "pair":pair,
                "side":"SELL",
                "qty":qty,
                "price":price,
                "mode":MODE,
                "exchange_status":res.get("status","unknown"),
                "exchange_msg":res
            })

            del positions[pair]

# ================== MAIN LOOP ==================
price_hist = {p: [] for p in PAIRS}

def scan_loop():
    global _last_keepalive, engine_alive, last_heartbeat, last_scan

    engine_alive = True

    while True:
        last_heartbeat = now()

        if not running:
            time.sleep(1)
            continue

        # keepalive
        if APP_BASE_URL and (time.time() - _last_keepalive > KEEPALIVE_SEC):
            _keepalive_ping()
            _last_keepalive = time.time()

        prices = {}
        for p in PAIRS:
            pr = fetch_price(p)
            if pr:
                prices[p] = pr
                price_hist[p].append(pr)
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
                    "time":now(),
                    "pair": pair,
                    "side": "BUY",
                    "qty": qty,
                    "price": entry,
                    "mode": MODE,
                    "exchange_status": res.get("status","unknown"),
                    "exchange_msg": res
                })

        last_scan = now()
        time.sleep(CANDLE_INTERVAL)

# ================== API ==================
@app.route("/ping")
def ping():
    return "pong"

@app.route("/status")
def status():
    unrealized = 0.0
    prices = {p:fetch_price(p) for p in PAIRS}

    for p,pos in positions.items():
        pr = prices.get(p)
        if pr:
            unrealized += (pr - pos["entry"]) * pos["qty"]

    return jsonify({
        "mode": MODE,
        "running": running,
        "engine_alive": engine_alive,
        "server_time": now(),
        "last_heartbeat": last_heartbeat,
        "last_scan": last_scan,
        "positions": positions,
        "orders": orders[-50:],
        "pnl":{
            "realized": round(realized_pnl,2),
            "unrealized": round(unrealized,2)
        }
    })

@app.route("/mode", methods=["POST"])
def set_mode():
    global MODE
    MODE = request.json.get("mode","PAPER")
    return jsonify({"ok":True,"mode":MODE})

@app.route("/control", methods=["POST"])
def control():
    global running
    running = bool(request.json.get("run",True))
    return jsonify({"running":running})

# ================== BOOT ==================
if __name__ == "__main__":
    threading.Thread(target=scan_loop, daemon=True).start()
    app.run(host="0.0.0.0", port=int(os.environ.get("PORT", 10000)))