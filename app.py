import os, time, json, hmac, hashlib, requests, logging
from flask import Flask, jsonify, render_template, request

# Setup Professional Logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s')
logger = logging.getLogger("BTC_Bot")

app = Flask(__name__)

# Credentials & Config
API_KEY = os.environ.get("API_KEY")
API_SECRET = os.environ.get("API_SECRET", "")
BASE = "https://api.coindcx.com"
SYMBOL = "I-BTC_INR"  # Standard CoinDCX BTC/INR Symbol

# Persistent Log Storage (simplified for demo)
trade_logs = []

def sign(payload_str):
    return hmac.new(API_SECRET.encode(), payload_str.encode(), hashlib.sha256).hexdigest()

def signed_post(endpoint, body):
    body["timestamp"] = int(time.time() * 1000)
    payload = json.dumps(body, separators=(',', ':'))
    headers = {"X-AUTH-APIKEY": API_KEY, "X-AUTH-SIGNATURE": sign(payload), "Content-Type": "application/json"}
    try:
        r = requests.post(BASE + endpoint, headers=headers, data=payload, timeout=10)
        return r.json()
    except Exception as e:
        return {"error": str(e)}

# Professional Precision & Portfolio Fetching
@app.route("/api/portfolio")
def get_portfolio():
    balances = signed_post("/exchange/v1/users/balances", {})
    btc_bal = next((item for item in balances if item["currency"] == "BTC"), {"balance": 0})
    inr_bal = next((item for item in balances if item["currency"] == "INR"), {"balance": 0})
    
    # Get Live Price for P&L
    ticker = requests.get(f"{BASE}/exchange/ticker").json()
    price = next((float(x["last_price"]) for x in ticker if x["market"] == SYMBOL), 0)
    
    return jsonify({
        "btc": float(btc_bal["balance"]),
        "inr": float(inr_bal["balance"]),
        "current_price": price,
        "logs": trade_logs[-10:] # Last 10 trades
    })

@app.route("/trade", methods=["POST"])
def execute_trade():
    data = request.json
    side = data.get("side") # "buy" or "sell"
    qty = float(data.get("quantity", 0))
    
    # 1. Fetch Market Details for Step/Min Qty
    details = requests.get(f"{BASE}/exchange/v1/market_details").json()
    m = next((x for x in details if x['symbol'] == SYMBOL), {})
    
    # 2. Precision Formatting
    final_qty = round(qty, m.get('target_currency_precision', 6))
    
    # 3. Order Posting
    order_body = {"market": SYMBOL, "total_quantity": final_qty, "side": side, "order_type": "market_order"}
    res = signed_post("/exchange/v1/orders/create", order_body)
    
    # 4. Log Execution status
    log_entry = {
        "time": time.strftime("%Y-%m-%d %H:%M:%S"),
        "side": side, "qty": final_qty, "status": res.get("status", "error"),
        "order_id": res.get("id", "N/A")
    }
    trade_logs.append(log_entry)
    logger.info(f"Order Executed: {log_entry}")
    
    return jsonify(res)

# Keep-Alive Route for Render
@app.route("/keepalive")
def keepalive():
    return "Bot is pulse-active", 200

@app.route("/")
def index():
    return render_template("index.html")

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=int(os.environ.get("PORT", 10000)))
