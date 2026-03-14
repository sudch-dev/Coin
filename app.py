import os
import time
import json
import hmac
import hashlib
import requests
import threading
import pandas as pd
from flask import Flask, render_template
from datetime import datetime

app = Flask(__name__)

# --- SECURE CONFIG FROM RENDER ENV ---
API_KEY = os.environ.get('API_KEY')
API_SECRET = os.environ.get('API_SECRET')
TRADE_MODE = os.environ.get('TRADE_MODE', 'paper') # 'live' or 'paper'
RENDER_URL = "https://coin-4k37.onrender.com"
SYMBOLS = ["B-BTC_USDT", "B-ETH_USDT", "B-SOL_USDT"]

# --- GLOBAL STATE ---
state = {
    "logs": [],
    "positions": {s: None for s in SYMBOLS},
    "prices": {s: 0 for s in SYMBOLS},
    "emas": {s: {"ema5": 0, "ema10": 0} for s in SYMBOLS},
    "last_sync": "Initializing..."
}

def log_event(msg):
    ts = datetime.now().strftime('%H:%M:%S')
    state["logs"].insert(0, f"[{ts}] {msg}")
    if len(state["logs"]) > 30: state["logs"].pop()
    print(f"[{ts}] {msg}")

# --- COINDCX API EXECUTION ---
def send_order(symbol, side):
    if TRADE_MODE != 'live':
        log_event(f"🧪 [PAPER] {side.upper()} order simulated for {symbol}")
        return True

    url = "https://api.coindcx.com"
    ts = int(round(time.time() * 1000))
    
    # Payload for Market Order (adjust total_quantity as needed)
    payload = {
        "side": "buy" if side.upper() == "BUY" else "sell",
        "order_type": "market_order",
        "market": symbol,
        "total_quantity": 0.001, 
        "timestamp": ts
    }
    
    json_payload = json.dumps(payload, separators=(',', ':'))
    signature = hmac.new(API_SECRET.encode(), json_payload.encode(), hashlib.sha256).hexdigest()
    
    headers = {
        'Content-Type': 'application/json',
        'X-AUTH-APIKEY': API_KEY,
        'X-AUTH-SIGNATURE': signature
    }

    try:
        r = requests.post(url, data=json_payload, headers=headers, timeout=10)
        res = r.json()
        if "id" in res:
            log_event(f"✅ LIVE ORDER SUCCESS: {side} {symbol}")
            return True
        else:
            log_event(f"❌ LIVE ORDER FAILED: {res.get('message', 'Unknown Error')}")
            return False
    except Exception as e:
        log_event(f"⚠️ API Connection Error: {str(e)[:50]}")
        return False

# --- TRADING ENGINE ---
def trading_loop():
    log_event(f"🚀 Bot Started in {TRADE_MODE.upper()} mode")
    while True:
        for sym in SYMBOLS:
            try:
                # Fetch Market Data
                res = requests.get(f"https://public.coindcx.com{sym}&interval=1m&limit=50", timeout=5).json()
                df = pd.DataFrame(res)
                df['close'] = df['close'].astype(float)
                
                # Indicators
                df['ema5'] = df['close'].ewm(span=5, adjust=False).mean()
                df['ema10'] = df['close'].ewm(span=10, adjust=False).mean()
                
                curr = df['close'].iloc[-1]
                e5, e10 = df['ema5'].iloc[-1], df['ema10'].iloc[-1]
                
                # Update Dashboard State
                state["prices"][sym] = round(curr, 2)
                state["emas"][sym] = {"ema5": round(e5, 2), "ema10": round(e10, 2)}
                state["last_sync"] = datetime.now().strftime('%Y-%m-%d %H:%M:%S')

                pos = state["positions"][sym]
                
                if pos:
                    # 1% TP / 0.5% SL Logic
                    pnl = (curr - pos['price'])/pos['price'] if pos['side']=='BUY' else (pos['price'] - curr)/pos['price']
                    
                    exit_condition = (pnl >= 0.01) or (pnl <= -0.005) or \
                                     (pos['side']=='BUY' and e5 < e10) or \
                                     (pos['side']=='SELL' and e5 > e10)

                    if exit_condition:
                        if send_order(sym, "sell" if pos['side']=="BUY" else "buy"):
                            log_event(f"🔴 EXIT {sym} | PnL: {pnl:.2%}")
                            state["positions"][sym] = None
                else:
                    # Entry Logic (EMA 5/10 Cross)
                    if e5 > e10:
                        if send_order(sym, "buy"):
                            state["positions"][sym] = {'side': 'BUY', 'price': curr}
                            log_event(f"🟢 ENTRY LONG {sym} @ {curr}")
                    elif e5 < e10:
                        if send_order(sym, "sell"):
                            state["positions"][sym] = {'side': 'SELL', 'price': curr}
                            log_event(f"🟠 ENTRY SHORT {sym} @ {curr}")
                            
            except Exception as e:
                log_event(f"⚠️ Loop Error ({sym}): {str(e)[:30]}")
        
        time.sleep(60) # Poll every 1 minute

# --- 5-MINUTE KEEP-ALIVE ---
def pinger():
    while True:
        time.sleep(300) 
        try:
            requests.get(RENDER_URL, timeout=10)
            log_event("📡 Keep-alive: Pong!")
        except:
            pass

@app.route('/')
def index():
    return render_template('index.html', state=state, mode=TRADE_MODE)

if __name__ == "__main__":
    threading.Thread(target=trading_loop, daemon=True).start()
    threading.Thread(target=pinger, daemon=True).start()
    app.run(host='0.0.0.0', port=10000)
