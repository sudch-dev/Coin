import os
import json
import time
import hmac
import hashlib
import requests
import threading
from datetime import datetime
from zoneinfo import ZoneInfo
import numpy as np
import pandas as pd
import pandas_ta as ta
from fastapi import FastAPI, Request, Form
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
from fastapi.staticfiles import StaticFiles

app = FastAPI()
templates = Jinja2Templates(directory="templates")  # Create a 'templates' folder in repo for HTML
app.mount("/static", StaticFiles(directory="static"), name="static")  # Optional for CSS/JS

# Env vars
API_KEY = os.getenv('API_KEY')
API_SECRET = os.getenv('API_SECRET')
RENDER_URL = 'https://coin-4k37.onrender.com/pi'

# Global state (for simplicity; use Redis for prod)
bot_running = False
trades_df = pd.DataFrame(columns=['Time', 'Pair', 'Action', 'PnL'])
last_scan = None
last_update = None
capital = 10000.0
paper_trade = True

class CoinDCXAPI:
    # (Same as before - paste the full class here from previous app.py)
    def __init__(self, api_key, api_secret, paper_trade=False):
        self.base_url = 'https://api.coindcx.com'
        self.api_key = api_key
        self.api_secret = api_secret
        self.paper_trade = paper_trade
        self.balances = {'INR': 10000.0 if paper_trade else 0.0}  # Default for paper
        self.orders = []
        self.trades = []

    # ... (include all methods: _generate_signature, get_markets, get_ticker, get_candles, get_balance, place_order, get_ticker_for_pair)

# Strategy functions (same as before)
def calculate_rsi(closes, period=14):
    df = pd.DataFrame({'close': closes})
    df['rsi'] = ta.rsi(df['close'], length=period)
    return df['rsi'].iloc[-1]

def select_interval(volatility):
    if volatility > 0.05:
        return '1m'
    elif volatility > 0.02:
        return '5m'
    else:
        return '15m'

def get_signal(candles):
    df = pd.DataFrame(candles, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
    df['close'] = df['close'].astype(float)
    volatility = df['close'].pct_change().std()
    rsi = calculate_rsi(df['close'].values)
    if rsi < 30 and df['close'].iloc[-1] > df['close'].iloc[-3]:
        return 'buy', volatility
    elif rsi > 70 and df['close'].iloc[-1] < df['close'].iloc[-3]:
        return 'sell', volatility
    return None, volatility

def find_best_pair(api, capital, fee_rate=0.001):
    tickers = api.get_ticker()
    candidates = []
    for t in tickers:
        if t['market'].endswith('INR'):
            change = abs(float(t['change_24_hour']))
            price = float(t['last_price'])
            vol_rough = float(t.get('volume', 0)) / price
            expected_profit = (change / 100) * 0.5 - fee_rate * 2
            if expected_profit > 0.005 and vol_rough > 1:
                candidates.append((t, expected_profit, vol_rough))
    if candidates:
        candidates.sort(key=lambda x: x[1] * x[2], reverse=True)
        return candidates[0][0]['market']
    return None

# Keepalive thread
def keepalive():
    while True:
        try:
            requests.get(RENDER_URL)
        except:
            pass
        time.sleep(240)

# Bot loop (modified for global state)
def bot_loop(api):
    global bot_running, trades_df, last_scan, last_update, capital
    while bot_running:
        try:
            print(f"[DEBUG] Scanning at {datetime.now()} | Capital: ₹{capital}")
            best_pair = find_best_pair(api, capital)
            print(f"[DEBUG] Best pair: {best_pair}")
            if best_pair:
                candles = api.get_candles(best_pair)
                if candles:
                    signal, vol = get_signal(candles)
                    closes = [c[4] for c in candles]
                    rsi_val = calculate_rsi(closes)
                    print(f"[DEBUG] Signal: {signal} | Vol: {vol:.2%} | RSI: {rsi_val:.1f}")
                    if signal:
                        ticker = api.get_ticker_for_pair(best_pair)
                        price = float(ticker['last_price'])
                        quantity = min(capital * 0.05 / price, 10)
                        order = api.place_order(signal, 'market_order', best_pair, quantity)
                        if order:
                            new_trade = {'Time': datetime.now().strftime('%Y-%m-%d %H:%M:%S'), 'Pair': best_pair, 'Action': signal, 'PnL': 0.0}
                            trades_df = pd.concat([trades_df, pd.DataFrame([new_trade])], ignore_index=True)
                            last_update = datetime.now(ZoneInfo("Asia/Kolkata"))
                            if signal == 'buy':
                                capital -= quantity * price * 0.001
            last_scan = datetime.now(ZoneInfo("Asia/Kolkata"))
            time.sleep(60)
        except Exception as e:
            print(f"[DEBUG] Error: {e}")
            time.sleep(10)

api = CoinDCXAPI(API_KEY, API_SECRET, paper_trade)

@app.get("/", response_class=HTMLResponse)
async def dashboard(request: Request):
    ist_now = datetime.now(ZoneInfo("Asia/Kolkata"))
    balances = api.get_balance()
    portfolio = [{'Coin': k, 'Balance': v} for k, v in balances.items() if k != 'INR']
    status = "🟢 Running" if bot_running else "🔴 Stopped"
    last_scan_str = last_scan.strftime('%Y-%m-%d %H:%M:%S') if last_scan else "N/A"
    last_update_str = last_update.strftime('%Y-%m-%d %H:%M:%S') if last_update else "N/A"
    
    html_content = f"""
    <!DOCTYPE html>
    <html>
    <head><title>ENGINE_PRO_v2.5</title><meta charset="UTF-8"></head>
    <body style="font-family: Arial; background: #000; color: #fff; padding: 20px;">
        <h1>ENGINE_PRO_v2.5</h1>
        <p>Current Time (IST): {ist_now.strftime('%Y-%m-%d %H:%M:%S')}</p>
        <p>Bot Status: {status}</p>
        <p>Last Scan (IST): {last_scan_str} | Last Update (IST): {last_update_str}</p>
        
        <form method="post" action="/set_params">
            <label>Capital (INR): <input type="number" name="capital" value="{capital}" step="1000"></label><br>
            <label>Paper Mode: <input type="checkbox" name="paper_trade" {'checked' if paper_trade else ''}></label><br>
            <input type="submit" value="Update">
        </form>
        
        <h3>Portfolio</h3>
        <table border="1" style="color: #fff;">
            <tr><th>Coin</th><th>Balance</th></tr>
            {' '.join(f'<tr><td>{row["Coin"]}</td><td>{row["Balance"]}</td></tr>' for row in portfolio)}
        </table>
        
        <h3>Controls</h3>
        <form method="post" action="/start_bot"><input type="submit" value="Start Bot"></form>
        <form method="post" action="/stop_bot"><input type="submit" value="Stop Bot"></form>
        <form method="post" action="/keepalive"><input type="submit" value="Start Keepalive"></form>
        
        <h3>Trades</h3>
        <table border="1" style="color: #fff;">
            <tr><th>Time</th><th>Pair</th><th>Action</th><th>PnL</th></tr>
            {' '.join(f'<tr><td>{row["Time"]}</td><td>{row["Pair"]}</td><td>{row["Action"]}</td><td>{row["PnL"]}</td></tr>' for _, row in trades_df.iterrows()) or '<tr><td colspan="4">No trades yet</td></tr>'}
        </table>
        <p><a href="/">Refresh (every 30s manually)</a></p>
    </body>
    </html>
    """
    return HTMLResponse(content=html_content)

@app.post("/set_params")
async def set_params(capital: float = Form(...), paper_trade: bool = Form(False)):
    global capital_global, paper_trade_global  # Update globals
    capital = float(capital)
    paper_trade = paper_trade == "on"  # Checkbox hack
    api.paper_trade = paper_trade
    return {"status": "Updated", "capital": capital, "paper": paper_trade}

@app.post("/start_bot")
async def start_bot():
    global bot_running
    if not bot_running:
        bot_running = True
        threading.Thread(target=bot_loop, args=(api,), daemon=True).start()
    return {"status": "Bot started"}

@app.post("/stop_bot")
async def stop_bot():
    global bot_running
    bot_running = False
    return {"status": "Bot stopped"}

@app.post("/keepalive")
async def start_keepalive():
    threading.Thread(target=keepalive, daemon=True).start()
    return {"status": "Keepalive started"}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)