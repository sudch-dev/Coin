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
from collections import deque, defaultdict
from statistics import median

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
    "DOGEUSDT": {"precision": 4, "min_qty": 0.01},
    "SOLUSDT": {"precision": 4, "min_qty": 0.01},
    "AEROUSDT": {"precision": 2, "min_qty": 0.01},
    "ADAUSDT": {"precision": 2, "min_qty": 0.1},
    "LTCUSDT": {"precision": 4, "min_qty": 0.001},
    "BNBUSDT": {"precision": 4, "min_qty": 0.001}
}

# --- Tunables ---
CANDLE_INTERVAL = 30                # seconds (was 60). 20–30 is snappier.
TRADE_COOLDOWN_SEC = 300            # cooldown after an exit; can lower to 120 if you want.

IST = timezone('Asia/Kolkata')
def ist_now(): return datetime.now(IST).strftime('%Y-%m-%d %H:%M:%S')
def ist_date(): return datetime.now(IST).strftime('%Y-%m-%d')
def ist_yesterday(): return (datetime.now(IST) - timedelta(days=1)).strftime('%Y-%m-%d')

tick_logs, candle_logs = {p: [] for p in PAIRS}, {p: [] for p in PAIRS}
scan_log, trade_log, exit_orders = [], [], []
daily_profit, pair_precision = {}, {}
running = False
status = {"msg": "Idle", "last": ""}
status_epoch = 0         # heartbeat for the UI watchdog
error_message = ""

# ===== Persistent P&L state (confirmed fills only) =====
PROFIT_STATE_FILE = "profit_state.json"
profit_state = {
    "cumulative_pnl": 0.0,
    "daily": {},
    "inventory": {},
    "processed_orders": []
}
pair_cooldown_until = {p: 0 for p in PAIRS}

def load_profit_state():
    global profit_state
    try:
        with open(PROFIT_STATE_FILE, "r") as f:
            data = json.load(f)
        profit_state["cumulative_pnl"] = float(data.get("cumulative_pnl", 0.0))
        profit_state["daily"] = dict(data.get("daily", {}))
        profit_state["inventory"] = data.get("inventory", {})
        profit_state["processed_orders"] = list(data.get("processed_orders", []))
    except:
        pass

def save_profit_state():
    tmp = {
        "cumulative_pnl": round(profit_state.get("cumulative_pnl", 0.0), 6),
        "daily": {k: round(v, 6) for k, v in profit_state.get("daily", {}).items()},
        "inventory": profit_state.get("inventory", {}),
        "processed_orders": profit_state.get("processed_orders", [])
    }
    try:
        with open(PROFIT_STATE_FILE, "w") as f:
            json.dump(tmp, f)
    except:
        pass

def _get_inventory_deque(market):
    inv = profit_state["inventory"].get(market, [])
    dq = deque()
    for lot in inv:
        try:
            q, c = float(lot[0]), float(lot[1])
            if q > 0 and c > 0:
                dq.append([q, c])
        except:
            continue
    return dq

def _set_inventory_from_deque(market, dq):
    profit_state["inventory"][market] = [[float(q), float(c)] for (q, c) in dq]

def apply_fill_update(market, side, price, qty, ts_ms, order_id):
    if not order_id: return
    if order_id in profit_state["processed_orders"]: return
    try:
        price = float(price); qty = float(qty)
    except:
        return
    if price <= 0 or qty <= 0: return

    inv = _get_inventory_deque(market)
    realized = 0.0

    if side.lower() == "buy":
        inv.append([qty, price])
    else:
        sell_q = qty
        while sell_q > 1e-18 and inv:
            lot_q, lot_px = inv[0]
            used = min(sell_q, lot_q)
            realized += (price - lot_px) * used
            lot_q -= used
            sell_q -= used
            if lot_q <= 1e-18:
                inv.popleft()
            else:
                inv[0][0] = lot_q

    _set_inventory_from_deque(market, inv)
    profit_state["processed_orders"].append(order_id)
    profit_state["cumulative_pnl"] = float(profit_state.get("cumulative_pnl", 0.0) + realized)
    dkey = ist_date()
    profit_state["daily"][dkey] = float(profit_state["daily"].get(dkey, 0.0) + realized)
    save_profit_state()

# ===== end persistent P&L helpers =====

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

def aggregate_candles(pair, interval=CANDLE_INTERVAL):
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
            candle["close"] = price   # <-- fixed (no stray ')')
            candle["volume"] += 1
    if candle: candles.append(candle)
    candle_logs[pair] = candles[-50:]

# ========= indicators =========
def _compute_ema(values, n):
    if len(values) < n: return None
    sma = sum(values[:n]) / n
    k = 2 / (n + 1)
    ema = sma
    for v in values[n:]:
        ema = v * k + ema * (1 - k)
    return ema

def _atr_14(candles):
    if len(candles) < 15: return None
    trs = []
    prev_close = candles[-15]["close"]
    for c in candles[-14:]:
        tr = max(c["high"] - c["low"], abs(c["high"] - prev_close), abs(c["low"] - prev_close))
        trs.append(tr)
        prev_close = c["close"]
    return sum(trs) / len(trs) if trs else None

# ========= responsive signal (uses live price) =========
def pa_buy_sell_signal(pair, live_price=None):
    """
    Responsive signal:
    - Trend: EMAfast > EMAslow (5/13) built on completed closes + live price
    - Breakout: price crosses Donchian channel of last N completed candles
    - ATR(14) from completed candles for sizing
    """
    candles = candle_logs[pair]
    if len(candles) < 25:
        return None

    # completed candles (exclude the currently forming one)
    completed = candles[:-1] if len(candles) >= 2 else candles
    if len(completed) < 20:
        return None

    closes = [c["close"] for c in completed]
    curr_price = float(live_price) if live_price else candles[-1]["close"]

    N = 5
    recent = completed[-N:]
    don_high = max(c["high"] for c in recent)
    don_low  = min(c["low"]  for c in recent)

    closes_plus_live = closes[-30:] + [curr_price]
    ema_fast = _compute_ema(closes_plus_live, 5)
    ema_slow = _compute_ema(closes_plus_live, 13)

    atr14 = _atr_14(completed)
    if ema_fast is None or ema_slow is None or atr14 is None:
        return None

    if curr_price > don_high and ema_fast > ema_slow:
        return {
            "side": "BUY",
            "entry": curr_price,
            "atr": atr14,
            "msg": f"BUY: live breakout > Donchian({N}) & EMA5>EMA13"
        }
    if curr_price < don_low and ema_fast < ema_slow:
        return {
            "side": "SELL",
            "entry": curr_price,
            "atr": atr14,
            "msg": f"SELL: live breakdown < Donchian({N}) & EMA5<EMA13"
        }
    return None

def _signed_post(url, body):
    payload = json.dumps(body, separators=(',', ':'))
    sig = hmac_signature(payload)
    headers = {"X-AUTH-APIKEY": API_KEY, "X-AUTH-SIGNATURE": sig, "Content-Type": "application/json"}
    try:
        r = requests.post(url, headers=headers, data=payload, timeout=12)
        if r.ok:
            return r.json()
    except:
        pass
    return {}

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

def get_order_status(order_id=None, client_order_id=None):
    body = {"timestamp": int(time.time() * 1000)}
    if order_id: body["id"] = order_id
    if client_order_id: body["client_order_id"] = client_order_id
    res = _signed_post(f"{BASE_URL}/exchange/v1/orders/status", body)
    return res if isinstance(res, dict) else {}

def _record_fill_from_status(market, side, st, order_id):
    try:
        total_q = float(st.get("total_quantity", 0))
        remain_q = float(st.get("remaining_quantity", 0))
        filled = max(0.0, total_q - remain_q)
        avg_px = float(st.get("avg_price", 0))
    except:
        filled, avg_px = 0.0, 0.0

    if filled > 0 and avg_px > 0:
        ts_field = st.get("updated_at") or st.get("created_at") or st.get("timestamp") or int(time.time()*1000)
        try:
            ts_ms = int(ts_field)
            if ts_ms < 10**12: ts_ms *= 1000
        except:
            ts_ms = int(time.time() * 1000)
        apply_fill_update(market, side, avg_px, filled, ts_ms, order_id)

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
            try:
                order_id = (res.get("orders") or [{}])[0].get("id")
            except:
                order_id = None
            if order_id:
                st = get_order_status(order_id=order_id)
                _record_fill_from_status(pair, "SELL", st, order_id)
            if "error" in res: error_message = res["error"]
            to_remove.append(ex)
            pair_cooldown_until[pair] = int(time.time()) + TRADE_COOLDOWN_SEC
        elif side == "SELL" and (price <= tp or price >= sl):
            res = place_order(pair, "BUY", qty)
            scan_log.append(f"{ist_now()} | {pair} | EXIT BUY {qty} @ {price} | {res}")
            try:
                order_id = (res.get("orders") or [{}])[0].get("id")
            except:
                order_id = None
            if order_id:
                st = get_order_status(order_id=order_id)
                _record_fill_from_status(pair, "BUY", st, order_id)
            if "error" in res: error_message = res["error"]
            to_remove.append(ex)
            pair_cooldown_until[pair] = int(time.time()) + TRADE_COOLDOWN_SEC
    for ex in to_remove: exit_orders.remove(ex)

def _has_open_exit_for(pair):
    for ex in exit_orders:
        if ex.get("pair") == pair:
            return True
    return False

def scan_loop():
    global running, error_message, status_epoch
    scan_log.clear()
    last_candle_ts = {p: 0 for p in PAIRS}
    interval = CANDLE_INTERVAL

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

            # build/refresh candles
            aggregate_candles(pair, interval)
            last_candle = candle_logs[pair][-1] if candle_logs[pair] else None

            if last_candle:
                # Evaluate signal EVERY LOOP using live price
                # Respect cooldown & pending exits first
                if int(time.time()) < pair_cooldown_until.get(pair, 0) or _has_open_exit_for(pair):
                    scan_log.append(f"{ist_now()} | {pair} | Cooldown/Exit pending — skip")
                else:
                    signal = pa_buy_sell_signal(pair, price)
                    if signal:
                        error_message = ""
                        entry = signal["entry"]
                        atr = signal.get("atr", None)

                        usdt_bal = balances.get("USDT", 0.0)
                        risk_amt = 0.005 * usdt_bal        # 0.5% risk per trade
                        min_tick_risk = entry * 0.0015     # tighter for responsiveness
                        risk_unit = max((0.6 * atr) if atr else 0, min_tick_risk)

                        if signal["side"] == "BUY":
                            sl = round(entry - risk_unit, 6)
                            tp = round(entry + 1.8 * risk_unit, 6)
                            risk_per_unit = max(entry - sl, 1e-9)
                            qty_risk = risk_amt / risk_per_unit
                            qty_cap = (0.3 * usdt_bal) / entry
                            qty = min(qty_risk, qty_cap)
                        else:
                            sl = round(entry + risk_unit, 6)
                            tp = round(entry - 1.8 * risk_unit, 6)
                            coin = pair[:-4]
                            qty = balances.get(coin, 0.0)

                        # Precision & min qty
                        qty = round(qty, pair_precision.get(pair, 6))
                        rule = PAIR_RULES.get(pair, {"precision": 6, "min_qty": 0.0001})
                        qty = max(qty, rule["min_qty"])
                        qty = round(qty, rule["precision"])

                        if qty <= 0:
                            scan_log.append(f"{ist_now()} | {pair} | Signal {signal['side']} but qty too small.")
                        else:
                            res = place_order(pair, signal["side"], qty)

                            scan_log.append(f"{ist_now()} | {pair} | {signal['side']} @ {entry} | SL {sl} | TP {tp} | {res}")
                            trade_log.append({
                                "time": ist_now(), "pair": pair, "side": signal["side"], "entry": entry,
                                "msg": signal["msg"], "tp": tp, "sl": sl, "qty": qty, "order_result": res
                            })
                            exit_orders.append({
                                "pair": pair, "side": signal["side"], "qty": qty,
                                "tp": tp, "sl": sl, "entry": entry
                            })

                            # Confirm from order success -> update P&L
                            try:
                                order_id = (res.get("orders") or [{}])[0].get("id")
                            except:
                                order_id = None
                            if order_id:
                                st = get_order_status(order_id=order_id)
                                _record_fill_from_status(pair, signal["side"], st, order_id)

                            if "error" in res:
                                error_message = res["error"]
                    else:
                        scan_log.append(f"{ist_now()} | {pair} | No Signal")

        status["msg"], status["last"] = "Running", ist_now()
        status_epoch = int(time.time())  # heartbeat for watchdog
        time.sleep(5)

    status["msg"] = "Idle"

def compute_realized_pnl_today():
    return round(profit_state["daily"].get(ist_date(), 0.0), 6)

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
    profit_today = compute_realized_pnl_today()
    profit_yesterday = round(profit_state["daily"].get(ist_yesterday(), 0.0), 6)
    cumulative_pnl = round(profit_state.get("cumulative_pnl", 0.0), 6)

    return jsonify({
        "status": status["msg"],
        "last": status["last"],
        "status_epoch": status_epoch,  # for frontend watchdog
        "usdt": balances.get("USDT", 0.0),
        "profit_today": profit_today,
        "profit_yesterday": profit_yesterday,
        "pnl_cumulative": cumulative_pnl,
        "coins": coins,
        "trades": trade_log[-10:][::-1],
        "scans": scan_log[-30:][::-1],
        "error": error_message
    })

@app.route("/ping")
def ping(): return "pong"

if __name__ == "__main__":
    load_profit_state()
    fetch_pair_precisions()
    app.run(host="0.0.0.0", port=10000)
