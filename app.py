
import os, hmac, hashlib, time, requests, json, sqlite3
from flask import Flask, jsonify, request
from datetime import datetime
import pytz

app = Flask(__name__)
API_KEY = os.environ.get('API_KEY')
API_SECRET = os.environ.get('API_SECRET').encode()
BASE_URL = "https://api.coindcx.com"
symbol = "BTCUSDT"; market = "BTC/USDT"

conn = sqlite3.connect('trading_data.db', check_same_thread=False)
cursor = conn.cursor()
cursor.execute("CREATE TABLE IF NOT EXISTS candle_data (timestamp TEXT PRIMARY KEY, symbol TEXT, open REAL, high REAL, low REAL, close REAL, volume REAL)")
conn.commit()

def get_ist_time():
    return datetime.now(pytz.timezone('Asia/Kolkata')).strftime('%Y-%m-%d %H:%M:%S')

def save_candle_to_db(symbol, c):
    cursor.execute("INSERT OR IGNORE INTO candle_data (timestamp, symbol, open, high, low, close, volume) VALUES (?, ?, ?, ?, ?, ?, ?)",
        (c['timestamp'], symbol, c['open'], c['high'], c['low'], c['close'], c['volume']))
    conn.commit()

def get_last_n_closes(symbol, n):
    cursor.execute("SELECT close FROM candle_data WHERE symbol = ? ORDER BY timestamp DESC LIMIT ?", (symbol, n))
    return [row[0] for row in reversed(cursor.fetchall())]

def calculate_ema(prices, window):
    ema = prices[0]; k = 2 / (window + 1)
    for price in prices[1:]: ema = price * k + ema * (1 - k)
    return round(ema, 2)

def get_wallet_balances():
    ts = int(time.time() * 1000)
    body = {'timestamp': ts}
    sig = hmac.new(API_SECRET, json.dumps(body, separators=(',', ':')).encode(), hashlib.sha256).hexdigest()
    headers = {'X-AUTH-APIKEY': API_KEY, 'X-AUTH-SIGNATURE': sig, 'Content-Type': 'application/json'}
    r = requests.post(BASE_URL + "/exchange/v1/users/balances", data=json.dumps(body), headers=headers)
    if r.status_code == 200:
        b = r.json()
        usdt = next((float(x['balance']) for x in b if x['currency'] == 'USDT'), 0)
        coin = next((float(x['balance']) for x in b if x['currency'] == 'BTC'), 0)
        return usdt, coin
    return 0, 0

def place_order(side, price, qty):
    ts = int(time.time() * 1000)
    body = {
        "market": market, "side": side, "order_type": "market",
        "price_per_unit": str(price), "total_quantity": str(qty), "timestamp": ts
    }
    sig = hmac.new(API_SECRET, json.dumps(body, separators=(',', ':')).encode(), hashlib.sha256).hexdigest()
    headers = {'X-AUTH-APIKEY': API_KEY, 'X-AUTH-SIGNATURE': sig, 'Content-Type': 'application/json'}
    return requests.post(BASE_URL + "/exchange/v1/orders/create", data=json.dumps(body), headers=headers).json()

def fetch_live_candle():
    r = requests.get(f"https://public.coindcx.com/market_data/candles?pair={symbol}&interval=1m&limit=2")
    if r.status_code == 200:
        return [{
            'timestamp': datetime.fromtimestamp(c[0] / 1000).strftime('%Y-%m-%d %H:%M:%S'),
            'open': float(c[1]), 'high': float(c[2]), 'low': float(c[3]),
            'close': float(c[4]), 'volume': float(c[5])
        } for c in r.json()]
    return None

trades = []; bot_running = False; net_pnl = 0.0

@app.route("/start", methods=["POST"])
def start(): global bot_running; bot_running = True; return jsonify({"status": "started"})

@app.route("/stop", methods=["POST"])
def stop(): global bot_running; bot_running = False; return jsonify({"status": "stopped"})

@app.route("/status")
def status():
    usdt, coin = get_wallet_balances()
    return jsonify({"status": "Running" if bot_running else "Stopped", "last": get_ist_time(),
                    "usdt": usdt, "coins": {"BTC": coin}, "net_pnl": net_pnl, "trades": trades[-5:]})

@app.route("/scan")
def scan():
    global net_pnl
    if not bot_running: return jsonify({"status": "Bot stopped"})
    candles = fetch_live_candle()
    if not candles: return jsonify({"status": "No candle"})
    for c in candles: save_candle_to_db(symbol, c)
    closes = get_last_n_closes(symbol, 10)
    if len(closes) < 10: return jsonify({"status": "Need more data"})
    ema5 = calculate_ema(closes[-5:], 5); ema10 = calculate_ema(closes, 10)
    usdt, coin = get_wallet_balances(); decision = None
    if ema5 > ema10 and usdt > 10:
        qty = round((usdt * 0.3) / closes[-1], 6)
        resp = place_order("buy", closes[-1], qty)
        trades.append({"type": "BUY", "price": closes[-1], "time": get_ist_time()})
        decision = f"BUY {qty} BTC"
    elif ema5 < ema10 and coin > 0.001:
        resp = place_order("sell", closes[-1], coin)
        trades.append({"type": "SELL", "price": closes[-1], "time": get_ist_time()})
        decision = f"SELL {coin} BTC"
    return jsonify({"timestamp": get_ist_time(), "price": closes[-1], "ema5": ema5, "ema10": ema10,
                    "decision": decision or "HOLD", "trades": trades[-5:]})

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=10000)
