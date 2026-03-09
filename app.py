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

app = FastAPI()

# Env vars
API_KEY = os.getenv('API_KEY')
API_SECRET = os.getenv('API_SECRET')
RENDER_URL = 'https://coin-4k37.onrender.com/pi'

# Global state
bot_running = False
trades_df = pd.DataFrame(columns=['Time', 'Pair', 'Action', 'PnL'])
last_scan = None
last_update = None
capital = 10000.0
paper_trade = True

class CoinDCXAPI:
    def __init__(self, api_key, api_secret, paper_trade=False):
        self.base_url = 'https://api.coindcx.com'
        self.api_key = api_key
        self.api_secret = api_secret
        self.paper_trade = paper_trade
        self.balances = {'INR': 10000.0 if paper_trade else 0.0}
        self.orders = []
        self.trades = []

    def _generate_signature(self, body):
        timestamp = int(time.time() * 1000)
        body['timestamp'] = timestamp
        json_body = json.dumps(body, separators=(',', ':'))
        signature = hmac.new(self.api_secret.encode(), json_body.encode(), hashlib.sha256).hexdigest()
        return json_body, signature

    def get_markets(self):
        if self.paper_trade:
            return ['BTCINR', 'ETHINR', 'XRPINR']
        try:
            response = requests.get(f'{self.base_url}/exchange/v1/markets')
            markets = response.json()
            return [m for m in markets if m.endswith('INR')]
        except Exception as e:
            print(f"Error fetching markets: {e}")
            return []

    def get_ticker(self):
        if self.paper_trade:
            return [
                {'market': 'BTCINR', 'last_price': 5000000, 'change_24_hour': '2.5', 'volume': 1000000},
                {'market': 'ETHINR', 'last_price': 300000, 'change_24_hour': '-1.2', 'volume': 500000}
            ]
        try:
            response = requests.get(f'{self.base_url}/exchange/ticker')
            return response.json()
        except Exception as e:
            print(f"Error fetching ticker: {e}")
            return []

    def get_candles(self, pair, interval='1m', limit=100):
        if self.paper_trade:
            np.random.seed(42)
            closes = np.cumsum(np.random.randn(limit)) + 100
            return [[int(time.time()*1000 - i*60*1000), closes[i], closes[i]+0.1, closes[i]-0.1, closes[i], 10] for i in range(limit-1, -1, -1)]
        try:
            response = requests.get(f'{self.base_url}/market_data/candles?pair={pair}&interval={interval}')
            return response.json()
        except Exception as e:
            print(f"Error fetching candles for {pair}: {e}")
            return []

    def get_balance(self):
        if self.paper_trade:
            return self.balances
        try:
            body = {}
            json_body, signature = self._generate_signature(body)
            headers = {
                'Content-Type': 'application/json',
                'X-AUTH-APIKEY': self.api_key,
                'X-AUTH-SIGNATURE': signature
            }
            response = requests.post(f'{self.base_url}/exchange/v1/users/balances', data=json_body, headers=headers)
            data = response.json()
            balances = {item['currency']: float(item['balance']) for item in data}
            self.balances.update(balances)
            return balances
        except Exception as e:
            print(f"Error fetching balance: {e}")
            return {}

    def place_order(self, side, order_type, market, total_quantity, price_per_unit=None):
        if self.paper_trade:
            order_id = len(self.orders) + 1
            ticker = self.get_ticker_for_pair(market)
            price = price_per_unit or float(ticker['last_price']) if ticker else 100
            order = {
                'id': order_id, 'side': side, 'type': order_type, 'market': market,
                'quantity': total_quantity, 'price': price, 'status': 'filled',
                'timestamp': int(time.time() * 1000)
            }
            self.orders.append(order)
            if side == 'buy':
                cost = total_quantity * price * 1.001
                self.balances['INR'] -= cost
                coin = market[:-3]
                self.balances[coin] = self.balances.get(coin, 0) + total_quantity
            else:
                proceeds = total_quantity * price * 0.999
                self.balances['INR'] += proceeds
                coin = market[:-3]
                self.balances[coin] = self.balances.get(coin, 0) - total_quantity
            self.trades.append({'time': datetime.now().strftime('%Y-%m-%d %H:%M:%S'), 'pair': market, 'action': side, 'pnl': 0})
            return order
        try:
            body = {'side': side, 'order_type': order_type, 'market': market, 'total_quantity': total_quantity}
            if price_per_unit:
                body['price_per_unit'] = price_per_unit
            json_body, signature = self._generate_signature(body)
            headers = {'Content-Type': 'application/json', 'X-AUTH-APIKEY': self.api_key, 'X-AUTH-SIGNATURE': signature}
            response = requests.post(f'{self.base_url}/exchange/v1/orders/create', data=json_body, headers=headers)
            return response.json()
        except Exception as e:
            print(f"Error placing order: {e}")
            return None

    def get_ticker_for_pair(self, pair):
        tickers = self.get_ticker()
        for t in tickers:
            if t['market'] == pair:
                return t
        return None

# Strategy (unchanged)
def calculate_rsi(closes, period=14):
    df = pd.DataFrame({'close': closes})
    df['rsi'] = ta.rsi(df['close'], length=period)
    return df['rsi'].iloc[-1]

def select_interval(volatility):
    if volatility > 0.05: return '1m'
    elif volatility > 0.02: return '5m'
    else: return '15m'

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

# Threads
def keepalive():
    while True:
        try:
            requests.get(RENDER_URL)
        except:
            pass
        time.sleep(240)

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
    portfolio_rows = ''.join(f'<tr><td>{row["Coin"]}</td><td>{row["Balance"]:.4f}</td></tr>' for row in portfolio) or '<tr><td colspan="2">No holdings</td></tr>'
    trades_rows = ''.join(f'<tr><td>{row["Time"]}</td><td>{row["Pair"]}</td><td>{row["Action"]}</td><td>{row["PnL"]}</td></tr>' for _, row in trades_df.iterrows()) or '<tr><td colspan="4">No trades yet</td></tr>'
    
    html_content = f"""
    <!DOCTYPE html>
    <html>
    <head>
        <title>ENGINE_PRO_v2.5</title>
        <meta charset="UTF-8">
        <meta http-equiv="refresh" content="30">  <!-- Auto-refresh every 30s -->
        <style>body {{ font-family: Arial; background: #000; color: #fff; padding: 20px; }} table {{ border-collapse: collapse; width: 100%; }} th, td {{ border: 1px solid #fff; padding: 8px; text-align: left; }} form {{ margin: 10px 0; }}</style>
    </head>
    <body>
        <h1>ENGINE_PRO_v2.5</h1>
        <p>Current Time (IST): {ist_now.strftime('%Y-%m-%d %H:%M:%S')}</p>
        <p>Bot Status: {status} | Last Scan: {last_scan_str} | Last Update: {last_update_str}</p>
        
        <form method="post" action="/set_params">
            Capital (INR): <input type="number" name="capital" value="{capital}" step="1000"><br>
            Paper Mode: <input type="checkbox" name="paper_trade" {'checked' if paper_trade else ''} value="on"><br>
            <input type="submit" value="Update">
        </form>
        
        <h3>Portfolio</h3>
        <table><tr><th>Coin</th><th>Balance</th></tr>{portfolio_rows}</table>
        
        <h3>Controls</h3>
        <form method="post" action="/start_bot"><input type="submit" value="Start Bot"></form>
        <form method="post" action="/stop_bot"><input type="submit" value="Stop Bot"></form>
        <form method="post" action="/keepalive"><input type="submit" value="Start Keepalive"></form>
        
        <h3>Trades</h3>
        <table><tr><th>Time</th><th>Pair</th><th>Action</th><th>PnL</th></tr>{trades_rows}</table>
    </body>
    </html>
    """
    return HTMLResponse(content=html_content)

@app.post("/set_params")
async def set_params(request: Request, capital: float = Form(10000.0), paper_trade: str = Form("off")):
    global capital, paper_trade
    capital = float(capital)
    paper_trade = paper_trade == "on"
    api.paper_trade = paper_trade
    if paper_trade:
        api.balances['INR'] = capital
    return HTMLResponse(content=f"<p>Updated: Capital ₹{capital}, Paper: {paper_trade}</p><a href='/'>Back</a>")

@app.post("/start_bot")
async def start_bot():
    global bot_running
    if not bot_running:
        bot_running = True
        threading.Thread(target=bot_loop, args=(api,), daemon=True).start()
    return HTMLResponse(content="<p>Bot started!</p><a href='/'>Back</a>")

@app.post("/stop_bot")
async def stop_bot():
    global bot_running
    bot_running = False
    return HTMLResponse(content="<p>Bot stopped!</p><a href='/'>Back</a>")

@app.post("/keepalive")
async def start_keepalive():
    threading.Thread(target=keepalive, daemon=True).start()
    return HTMLResponse(content="<p>Keepalive started!</p><a href='/'>Back</a>")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=int(os.getenv("PORT", 8000)))