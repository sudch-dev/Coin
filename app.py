import os
import time
import threading
import hmac
import hashlib
import requests
import json
from flask import Flask, render_template, jsonify
from datetime import datetime, timedelta
from pytz import timezone
from collections import deque, defaultdict  # needed for FIFO P&L

app = Flask(__name__)

API_KEY = os.environ.get("API_KEY")
API_SECRET = os.environ.get("API_SECRET").encode()
BASE_URL = "https://api.coindcx.com"
PAIRS = [
    "BTCUSDT", "ETHUSDT", "XRPUSDT", "SHIBUSDT", "SOLUSDT",
    "DOGEUSDT", "ADAUSDT", "AEROUSDT", "BNBUSDT", "LTCUSDT"
]

PAIR_RULES = {
    "BTCUSDT": {"precision": 4, "min_qty": 0.001},
    "ETHUSDT": {"precision": 6, "min_qty": 0.0001},
    "XRPUSDT": {"precision": 4, "min_qty": 0.1},
    "SHIBUSDT": {"precision": 4, "min_qty": 10000},
    "DOGEUSDT": {"precision": 4, "min_qty": .01},
    "SOLUSDT": {"precision": 4, "min_qty": 0.01},
    "AEROUSDT": {"precision": 2, "min_qty": .01},
    "ADAUSDT": {"precision": 2, "min_qty": 0.1},
    "LTCUSDT": {"precision": 4, "min_qty": 0.001},
    "BNBUSDT": {"precision": 4, "min_qty": 0.001}
}

IST = timezone('Asia/Kolkata')
def ist_now(): return datetime.now(IST).strftime('%Y-%m-%d %H:%M:%S')
def ist_date(): return datetime.now(IST).strftime('%Y-%m-%d')
def ist_yesterday(): return (datetime.now(IST) - timedelta(days=1)).strftime('%Y-%m-%d')

tick_logs, candle_logs = {p: [] for p in PAIRS}, {p: [] for p in PAIRS}
scan_log, trade_log, exit_orders = [], [], []
daily_profit, pair_precision = {}, {}
running = False
status = {"msg": "Idle", "last": ""}
error_message = ""

def hmac_signature(payload):
    return hmac.new(API_SECRET, payload.encode(), hashlib.sha256).hexdigest()

def fetch_pair_precisions():
    try:
        r = requests.get(f"{BASE_URL}/exchange/v1/markets_details", timeout=10)
        if r.ok:
            for item in r.json():
                if item.get("pair") in PAIRS:
                    pair_precision[item["pair"]] = int(item.get("target_currency_precision", 6))
    except:
        pass

def get_wallet_balances():
    payload = json.dumps({"timestamp": int(time.time() * 1000)})
    sig = hmac_signature(payload)
    headers = {"X-AUTH-APIKEY": API_KEY, "X-AUTH-SIGNATURE": sig, "Content-Type": "application/json"}
    balances = {}
    try:
        r = requests.post(f"{BASE_URL}/exchange/v1/users/balances", headers=headers, data=payload, timeout=10)
        if r.ok:
            for b in r.json():
                balances[b['currency']] = float(b['balance'])
    except:
        pass
    return balances

def fetch_all_prices():
    try:
        r = requests.get(f"{BASE_URL}/exchange/ticker", timeout=10)
        if r.ok:
            now = int(time.time())
            return {item["market"]: {"price": float(item["last_price"]), "ts": now}
                    for item in r.json() if item.get("market") in PAIRS}
    except:
        pass
    return {}

def aggregate_candles(pair, interval=60):
    ticks = tick_logs[pair]
    if not ticks: return
    candles, candle, last_window = [], None, None
    for ts, price in sorted(ticks, key=lambda x: x[0]):
        wstart = ts - (ts % interval)
        if last_window != wstart:
            if candle: candles.append(candle)
            candle = {"open": price, "high": price, "low": price, "close": price, "volume": 1, "start": wstart}
            last_window = wstart
        else:
            candle["high"] = max(candle["high"], price)
            candle["low"] = min(candle["low"], price)
            candle["close"] = price
            candle["volume"] += 1
    if candle: candles.append(candle)
    candle_logs[pair] = candles[-50:]

def pa_buy_sell_signal(pair):
    candles = candle_logs[pair]
    if len(candles) < 3:
        return None

    prev1, prev2, curr = candles[-3], candles[-2], candles[-1]

    # Relaxed BUY
    if curr["close"] > prev2["close"] and curr["high"] > max(prev1["high"], prev2["high"]):
        return {"side": "BUY", "entry": curr["close"], "msg": "PA BUY: close > prev close and high > last 2 highs"}

    # Relaxed SELL
    if curr["close"] < prev2["close"] and curr["low"] < min(prev1["low"], prev2["low"]):
        return {"side": "SELL", "entry": curr["close"], "msg": "PA SELL: close < prev close and low < last 2 lows"}

    return None

def place_order(pair, side, qty):
    payload = {"market": pair, "side": side.lower(), "order_type": "market_order", "total_quantity": str(qty),
               "timestamp": int(time.time() * 1000)}
    body = json.dumps(payload)
    sig = hmac_signature(body)
    headers = {"X-AUTH-APIKEY": API_KEY, "X-AUTH-SIGNATURE": sig, "Content-Type": "application/json"}
    try:
        r = requests.post(f"{BASE_URL}/exchange/v1/orders/create", headers=headers, data=body, timeout=10)
        return r.json()
    except Exception as e:
        return {"error": str(e)}

def monitor_exits(prices):
    global error_message
    to_remove = []
    for ex in exit_orders:
        pair, side, qty, tp, sl, entry = ex.values()
        price = prices.get(pair, {}).get("price")
        if not price: continue
        if side == "BUY" and (price >= tp or price <= sl):
            res = place_order(pair, "SELL", qty)
            scan_log.append(f"{ist_now()} | {pair} | EXIT SELL {qty} @ {price} | {res}")
            pl = (price - entry) * qty
            daily_profit[ist_date()] = daily_profit.get(ist_date(), 0) + pl
            if "error" in res: error_message = res["error"]
            to_remove.append(ex)
        elif side == "SELL" and (price <= tp or price >= sl):
            res = place_order(pair, "BUY", qty)
            scan_log.append(f"{ist_now()} | {pair} | EXIT BUY {qty} @ {price} | {res}")
            pl = (entry - price) * qty
            daily_profit[ist_date()] = daily_profit.get(ist_date(), 0) + pl
            if "error" in res: error_message = res["error"]
            to_remove.append(ex)
    for ex in to_remove: exit_orders.remove(ex)

def scan_loop():
    global running, error_message
    scan_log.clear()
    last_candle_ts = {p: 0 for p in PAIRS}
    interval = 60

    while running:
        prices = fetch_all_prices()
        now = int(time.time())
        monitor_exits(prices)
        balances = get_wallet_balances()

        for pair in PAIRS:
            if pair not in prices:
                continue

            price = prices[pair]["price"]
            tick_logs[pair].append((now, price))
            if len(tick_logs[pair]) > 1000:
                tick_logs[pair] = tick_logs[pair][-1000:]

            aggregate_candles(pair, interval)
            last_candle = candle_logs[pair][-1] if candle_logs[pair] else None

            if last_candle and last_candle["start"] != last_candle_ts[pair]:
                last_candle_ts[pair] = last_candle["start"]
                signal = pa_buy_sell_signal(pair)

                if signal:
                    error_message = ""
                    coin = pair[:-4]
                    qty = (0.3 * balances.get("USDT", 0)) / signal["entry"] if signal["side"] == "BUY" else balances.get(coin, 0)
                    qty = round(qty, pair_precision.get(pair, 6))

                    tp = round(signal['entry'] * 1.003, 6)
                    sl = round(signal['entry'] * 0.997, 6)

                    res = place_order(pair, signal["side"], qty)

                    # Apply precision and min_qty logic
                    rule = PAIR_RULES.get(pair, {"precision": 6, "min_qty": 0.0001})
                    qty = max(qty, rule["min_qty"])
                    qty = round(qty, rule["precision"])

                    # Log result
                    scan_log.append(f"{ist_now()} | {pair} | {signal['side']} @ {signal['entry']} | {res}")
                    trade_log.append({
                        "time": ist_now(), "pair": pair, "side": signal["side"], "entry": signal["entry"],
                        "msg": signal["msg"], "tp": tp, "sl": sl, "qty": qty, "order_result": res
                    })
                    exit_orders.append({
                        "pair": pair, "side": signal["side"], "qty": qty,
                        "tp": tp, "sl": sl, "entry": signal["entry"]
                    })

                    if "error" in res:
                        error_message = res["error"]
                else:
                    # ✅ Even if no signal, log the scan
                    scan_log.append(f"{ist_now()} | {pair} | No Signal")

        status["msg"], status["last"] = "Running", ist_now()
        time.sleep(5)

    status["msg"] = "Idle"

# ---------- Executed-trade P&L (fetch + robust parse) ----------

def get_account_trades(from_ts_ms: int, to_ts_ms: int, symbol=None, limit=5000):
    """
    Try the documented trade-history endpoints. Use time-windowed one first,
    then fall back to the 'users/me/orders/trades' style and filter locally.
    """
    def signed_post(url, body):
        payload = json.dumps(body, separators=(',', ':'))
        sig = hmac_signature(payload)
        headers = {"X-AUTH-APIKEY": API_KEY, "X-AUTH-SIGNATURE": sig, "Content-Type": "application/json"}
        try:
            r = requests.post(url, headers=headers, data=payload, timeout=12)
            if r.ok:
                return r.json()
        except:
            pass
        return []

    # 1) Primary: time-windowed trade history (fields may be 'market', 'timestamp', 'price', 'quantity', 'side', 'fee_amount')
    body1 = {
        "timestamp": int(time.time() * 1000),
        "from_timestamp": int(from_ts_ms),
        "to_timestamp": int(to_ts_ms),
        "limit": limit
    }
    if symbol:
        body1["market"] = symbol
    trades = signed_post(f"{BASE_URL}/exchange/v1/orders/trade_history", body1)
    if isinstance(trades, list) and trades:
        return trades

    # 2) Fallback: broader account trades; filter window locally
    body2 = {
        "timestamp": int(time.time() * 1000),
        "limit": limit
    }
    if symbol:
        body2["market"] = symbol
    trades2 = signed_post(f"{BASE_URL}/exchange/v1/users/me/orders/trades", body2)
    return trades2 if isinstance(trades2, list) else []

def _norm_trade(t):
    """
    Normalize trade dict -> {market, side, price, qty, ts_ms, fee_base}
    Handles variations like market/symbol, price/price_per_unit/p, quantity/q, timestamp/T/created_at.
    """
    market = t.get("market") or t.get("symbol") or t.get("pair") or t.get("s")
    side   = (t.get("side") or "").lower()

    # price
    price = t.get("price", None)
    if price is None: price = t.get("price_per_unit", None)
    if price is None: price = t.get("p", 0)
    try:
        price = float(price)
    except:
        price = 0.0

    # quantity
    qty = t.get("quantity", None)
    if qty is None: qty = t.get("q", None)
    if qty is None: qty = t.get("total_quantity", 0)
    try:
        qty = float(qty)
    except:
        qty = 0.0

    # timestamp -> ms
    ts = t.get("timestamp", None)
    if ts is None: ts = t.get("T", None)
    if ts is None: ts = t.get("created_at", None)
    ts_ms = 0
    try:
        ts_ms = int(ts)
        if ts_ms < 10**12:  # seconds -> ms
            ts_ms = ts_ms * 1000
    except:
        ts_ms = 0  # if not numeric, ignore in-window filter later

    # fee amount (CoinDCX docs: fee_amount = absolute amount in BASE)
    fee_base = t.get("fee_amount", 0)  # may be missing
    try:
        fee_base = float(fee_base)
    except:
        fee_base = 0.0

    return {
        "market": market, "side": side, "price": price, "qty": qty, "ts_ms": ts_ms, "fee_base": fee_base
    }

def compute_realized_pnl_today():
    """
    Realized P&L for today (IST midnight -> now) from executed fills ONLY.
    FIFO in quote (USDT) terms. Fees assumed in BASE per CoinDCX docs; converted via trade price.
    """
    start_dt = datetime.now(IST).replace(hour=0, minute=0, second=0, microsecond=0)
    from_ts = int(start_dt.timestamp() * 1000)
    to_ts   = int(datetime.now(IST).timestamp() * 1000)

    raw = get_account_trades(from_ts, to_ts, symbol=None, limit=5000)
    if not isinstance(raw, list):
        return 0.0

    trades = [_norm_trade(t) for t in raw if isinstance(t, dict)]

    # Filter to the window if fallback returned a broader range
    trades = [t for t in trades if t["ts_ms"] == 0 or (from_ts <= t["ts_ms"] <= to_ts)]

    # Group by market, FIFO lots in BASE with cost in QUOTE
    by_market = defaultdict(list)
    for t in trades:
        if t["market"] and t["price"] > 0 and t["qty"] > 0 and t["side"] in ("buy", "sell"):
            by_market[t["market"]].append(t)

    realized_quote_pnl = 0.0
    inventory = defaultdict(deque)  # {market: deque([ [qty_base, cost_quote_per_base], ... ])}

    for market, lst in by_market.items():
        lst.sort(key=lambda x: x["ts_ms"])
        for tr in lst:
            px, q, fee_b = tr["price"], tr["qty"], tr["fee_base"]
            fee_quote = fee_b * px  # fee in BASE -> convert to QUOTE via trade price
            if tr["side"] == "buy":
                inventory[market].append([q, px])
                realized_quote_pnl -= fee_quote
            else:  # sell
                sell_qty = q
                while sell_qty > 1e-18 and inventory[market]:
                    lot_qty, lot_px = inventory[market][0]
                    used = min(sell_qty, lot_qty)
                    realized_quote_pnl += (px - lot_px) * used
                    lot_qty -= used
                    sell_qty -= used
                    if lot_qty <= 1e-18:
                        inventory[market].popleft()
                    else:
                        inventory[market][0][0] = lot_qty
                realized_quote_pnl -= fee_quote

    return round(realized_quote_pnl, 6)

# -------------------------------------------------------------

@app.route("/")
def index():
    return render_template("index.html")

@app.route("/start", methods=["POST"])
def start():
    global running
    if not running:
        running = True
        t = threading.Thread(target=scan_loop)
        t.daemon = True
        t.start()
    return jsonify({"status": "started"})

@app.route("/stop", methods=["POST"])
def stop():
    global running
    running = False
    return jsonify({"status": "stopped"})

@app.route("/status")
def get_status():
    balances = get_wallet_balances()
    coins = {pair[:-4]: balances.get(pair[:-4], 0.0) for pair in PAIRS}

    # Executed-only P&L (USDT terms for XXXUSDT pairs)
    profit_today = compute_realized_pnl_today()

    return jsonify({
        "status": status["msg"], "last": status["last"],
        "usdt": balances.get("USDT", 0.0),
        "profit_today": profit_today,
        "profit_yesterday": round(daily_profit.get(ist_yesterday(), 0), 4),  # keep as-is unless you want executed-only window calc too
        "coins": coins, "trades": trade_log[-10:][::-1], "scans": scan_log[-30:][::-1],
        "error": error_message
    })

@app.route("/ping")
def ping(): return "pong"

if __name__ == "__main__":
    fetch_pair_precisions()
    app.run(host="0.0.0.0", port=10000)
