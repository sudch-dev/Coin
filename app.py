import os
import json
import time
import hmac
import hashlib
import requests
import threading
import streamlit as st
import pandas as pd
import pandas_ta as ta  # pip install pandas_ta
from datetime import datetime
import numpy as np

# Environment variables
API_KEY = os.getenv('API_KEY')
API_SECRET = os.getenv('API_SECRET')
RENDER_URL = 'https://coin-4k37.onrender.com/pi'

class CoinDCXAPI:
    def __init__(self, api_key, api_secret, paper_trade=False):
        self.base_url = 'https://api.coindcx.com'
        self.api_key = api_key
        self.api_secret = api_secret
        self.paper_trade = paper_trade
        self.balances = {'INR': 0.0, 'BTC': 0.0, 'ETH': 0.0}  # Example, extend as needed
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
            return ['BTCINR', 'ETHINR', 'XRPINR']  # Simulated
        try:
            response = requests.get(f'{self.base_url}/exchange/v1/markets')
            markets = response.json()
            return [m for m in markets if m.endswith('INR')]
        except Exception as e:
            st.error(f"Error fetching markets: {e}")
            return []

    def get_ticker(self):
        if self.paper_trade:
            return [{'market': 'BTCINR', 'last_price': 5000000, 'change_24_hour': '2.5'},  # Simulated
                    {'market': 'ETHINR', 'last_price': 300000, 'change_24_hour': '-1.2'}]
        try:
            response = requests.get(f'{self.base_url}/exchange/ticker')
            return response.json()
        except Exception as e:
            st.error(f"Error fetching ticker: {e}")
            return []

    def get_candles(self, pair, interval='1m', limit=100):
        if self.paper_trade:
            # Simulated candles: [timestamp, open, high, low, close, volume]
            np.random.seed(42)  # For reproducibility in sim
            closes = np.cumsum(np.random.randn(limit)) + 100  # Random walk
            return [[int(time.time()*1000 - i*60*1000), closes[i], closes[i]+0.1, closes[i]-0.1, closes[i], 10] for i in range(limit-1, -1, -1)]
        try:
            response = requests.get(f'{self.base_url}/market_data/candles?pair={pair}&interval={interval}')
            return response.json()
        except Exception as e:
            st.error(f"Error fetching candles for {pair}: {e}")
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
            self.balances.update(balances)  # Sync
            return balances
        except Exception as e:
            st.error(f"Error fetching balance: {e}")
            return {}

    def place_order(self, side, order_type, market, total_quantity, price_per_unit=None):
        if self.paper_trade:
            # Simulate order
            order_id = len(self.orders) + 1
            order = {
                'id': order_id,
                'side': side,
                'type': order_type,
                'market': market,
                'quantity': total_quantity,
                'price': price_per_unit or self.get_ticker_for_pair(market)['last_price'],
                'status': 'filled',  # Assume instant fill for paper
                'timestamp': int(time.time() * 1000)
            }
            self.orders.append(order)
            # Update balance simulation
            price = order['price']
            if side == 'buy':
                cost = total_quantity * price * (1 + 0.001)  # Fee
                self.balances['INR'] -= cost
                self.balances[market[:-3]] += total_quantity
            else:
                proceeds = total_quantity * price * (1 - 0.001)
                self.balances['INR'] += proceeds
                self.balances[market[:-3]] -= total_quantity
            # Add to trades
            self.trades.append({
                'time': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                'pair': market,
                'action': side,
                'pnl': 0  # Calculate later
            })
            return order
        try:
            body = {
                'side': side,
                'order_type': order_type,
                'market': market,
                'total_quantity': total_quantity
            }
            if price_per_unit:
                body['price_per_unit'] = price_per_unit
            json_body, signature = self._generate_signature(body)
            headers = {
                'Content-Type': 'application/json',
                'X-AUTH-APIKEY': self.api_key,
                'X-AUTH-SIGNATURE': signature
            }
            response = requests.post(f'{self.base_url}/exchange/v1/orders/create', data=json_body, headers=headers)
            return response.json()
        except Exception as e:
            st.error(f"Error placing order: {e}")
            return None

    def get_ticker_for_pair(self, pair):
        tickers = self.get_ticker()
        for t in tickers:
            if t['market'] == pair:
                return t
        return None

# Strategy Functions
def calculate_rsi(closes, period=14):
    df = pd.DataFrame({'close': closes})
    df['rsi'] = ta.rsi(df['close'], length=period)
    return df['rsi'].iloc[-1]

def select_interval(volatility):
    # Volatility rough: std of returns
    if volatility > 0.05:  # High vol
        return '1m'
    elif volatility > 0.02:
        return '5m'
    else:
        return '15m'

def get_signal(candles):
    df = pd.DataFrame(candles, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
    df['close'] = df['close'].astype(float)
    volatility = df['close'].pct_change().std()
    interval = select_interval(volatility)
    # Re-fetch if needed, but assume current is ok

    rsi = calculate_rsi(df['close'].values)
    # Simple strategy: RSI oversold/overbought, with momentum
    if rsi < 30 and df['close'].iloc[-1] > df['close'].iloc[-3]:  # Buy signal: oversold + up momentum
        return 'buy', volatility
    elif rsi > 70 and df['close'].iloc[-1] < df['close'].iloc[-3]:  # Sell signal
        return 'sell', volatility
    return None, volatility

def find_best_pair(api, capital, fee_rate=0.001):
    tickers = api.get_ticker()
    candidates = []
    for t in tickers:
        if t['market'].endswith('INR'):
            change = abs(float(t['change_24_hour']))
            price = float(t['last_price'])
            vol_rough = float(t['volume']) / price  # Relative volume
            expected_profit = (change / 100) * 0.5 - fee_rate * 2  # Assume capture half move
            position_size = min(capital * 0.1 / price, 100)  # Risk 10%
            if expected_profit > 0.005 and vol_rough > 1:  # Min 0.5% profit, liquid
                candidates.append((t, expected_profit, vol_rough))
    if candidates:
        candidates.sort(key=lambda x: x[1] * x[2], reverse=True)  # Profit * liquidity
        return candidates[0][0]['market']
    return None

# Keepalive
def keepalive():
    while True:
        try:
            requests.get(RENDER_URL)
        except:
            pass
        time.sleep(240)

# def bot_loop(api, capital, running):
    trades_df = pd.DataFrame(columns=['Time', 'Pair', 'Action', 'PnL'])
    while running[0]:
        try:
            print(f"[DEBUG] Scanning at {datetime.now()} | Capital: ₹{capital}")  # Log timestamp
            best_pair = find_best_pair(api, capital)
            print(f"[DEBUG] Best pair selected: {best_pair}")  # Will show None if no candidates
            if best_pair:
                candles = api.get_candles(best_pair)
                if not candles:
                    print(f"[DEBUG] No candles for {best_pair}")
                    time.sleep(60)
                    continue
                signal, vol = get_signal(candles)
                print(f"[DEBUG] Signal for {best_pair}: {signal} | Vol: {vol:.2%} | RSI: {calculate_rsi([c[4] for c in candles]):.1f}")  # Log RSI too
                if signal:
                    ticker = api.get_ticker_for_pair(best_pair)
                    price = float(ticker['last_price'])
                    quantity = min(capital * 0.05 / price, 10)
                    print(f"[DEBUG] Executing {signal} {quantity} @ ₹{price} for {best_pair}")
                    order = api.place_order(signal, 'market_order', best_pair, quantity)
                    if order:
                        new_trade = {
                            'Time': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                            'Pair': best_pair,
                            'Action': signal,
                            'PnL': 0.0
                        }
                        trades_df = pd.concat([trades_df, pd.DataFrame([new_trade])], ignore_index=True)
                        st.session_state.trades_df = trades_df  # Update UI live
                        print(f"[DEBUG] Trade executed: {order}")
                        capital -= quantity * price * 0.001 if signal == 'buy' else 0  # Sim update
                else:
                    print(f"[DEBUG] No signal—skipping {best_pair}")
            else:
                print("[DEBUG] No qualifying pair—idling")
            time.sleep(60)
        except Exception as e:
            print(f"[DEBUG] Bot error: {e}")
            time.sleep(10)
    st.session_state.trades_df = trades_df
    return trades_df

# Streamlit UI
def main():
    st.set_page_config(page_title='CoinDCX Trading Bot', layout='wide')
    st.title('ENGINE_PRO_v2.5')
    if st.button('Start Keepalive Thread (for Render)'):
        threading.Thread(target=keepalive, daemon=True).start()
        st.success('Keepalive started')

    if not API_KEY or not API_SECRET:
        st.warning('Set API_KEY and API_SECRET env vars')
        st.stop()

    paper_trade = st.checkbox('Paper Trade Mode', value=True)
    api = CoinDCXAPI(API_KEY, API_SECRET, paper_trade)

    col1, col2 = st.columns(2)
    with col1:
        capital = st.number_input('Available Capital (INR)', value=10000.0, step=1000.0)
    with col2:
        balances = api.get_balance()
        st.metric('INR Balance', f"₹ {balances.get('INR', 0):,.2f}")

    portfolio_df = pd.DataFrame([
        {'Coin': k, 'Balance': v} for k, v in balances.items() if k != 'INR'
    ])
    if not portfolio_df.empty:
        st.subheader('Portfolio')
        st.dataframe(portfolio_df)

    col3, col4 = st.columns(2)
    with col3:
        if st.button('Start Bot'):
            running = [True]
            st.session_state.running = running
            thread = threading.Thread(target=bot_loop, args=(api, capital, running))
            thread.daemon = True
            thread.start()
            st.success('Bot started')
    with col4:
        if st.button('Stop Bot'):
            if 'running' in st.session_state:
                st.session_state.running[0] = False
            st.success('Bot stopped')

    st.subheader('Trades')
    if 'trades_df' in st.session_state:
        st.dataframe(st.session_state.trades_df)
    else:
        st.info('No trades yet')

if __name__ == '__main__':
    if 'trades_df' not in st.session_state:
        st.session_state.trades_df = pd.DataFrame()
    main()