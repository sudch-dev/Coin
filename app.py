# app.py — fixed precision & min-qty (per user table), TP-only, UI-settable Candle Interval + TP%,
# keep-alive self-ping, P&L persistence, precision-flooring + error-learning

import os, re, time, json, hmac, hashlib, threading, requests
from flask import Flask, render_template, jsonify, request
from datetime import datetime, timedelta
from pytz import timezone
from collections import deque

app = Flask(__name__)

# ====== Creds / Host ======
API_KEY = os.environ.get("API_KEY", "")
API_SECRET = (os.environ.get("API_SECRET", "") or "").encode()
BASE_URL = "https://api.coindcx.com"

# Keep-alive (self-ping /ping)
APP_BASE_URL  = os.environ.get("APP_BASE_URL", "").rstrip("/")
KEEPALIVE_SEC = int(os.environ.get("KEEPALIVE_SEC", "240"))
_last_keepalive = 0
def _keepalive_ping():
    if not APP_BASE_URL: return
    try: requests.get(f"{APP_BASE_URL}/ping", timeout=5)
    except Exception: pass

# ====== Pairs & FIXED rules (as requested) ======
PAIRS = [
    "BTCUSDT", "ETHUSDT", "XRPUSDT", "SHIBUSDT", "SOLUSDT",
    "DOGEUSDT", "ADAUSDT", "AEROUSDT", "BNBUSDT", "LTCUSDT"
]

# Using "precision" for BOTH price & qty precision; min_notional=0 by default
PAIR_RULES = {
    "BTCUSDT": {"precision": 2, "min_qty": 0.001},
    "ETHUSDT": {"precision": 6, "min_qty": 0.0001},
    "XRPUSDT": {"precision": 4, "min_qty": 0.1},
    "SHIBUSDT": {"precision": 4, "min_qty": 10000},
    "DOGEUSDT": {"precision": 4, "min_qty": 0.01},
    "SOLUSDT": {"precision": 4, "min_qty": 0.01},
    "AEROUSDT": {"precision": 2, "min_qty": 0.01},
    "ADAUSDT": {"precision": 2, "min_qty": 2},
    "LTCUSDT": {"precision": 2, "min_qty": 0.001},
    "BNBUSDT": {"precision": 4, "min_qty": 0.001}
}

# ====== Settings (UI editable) ======
# Candle interval default 15 min; TP default = 1% (0.01)
SETTINGS = {
    "candle_interval_sec": 15 * 60,
    "tp_pct": 0.01
}

# ====== Time & State ======
TRADE_COOLDOWN_SEC = 300
IST = timezone('Asia/Kolkata')
def ist_now(): return datetime.now(IST).strftime('%Y-%m-%d %H:%M:%S')
def ist_date(): return datetime.now(IST).strftime('%Y-%m-%d')
def ist_yesterday(): return (datetime.now(IST) - timedelta(days=1)).strftime('%Y-%m-%d')

tick_logs = {p: [] for p in PAIRS}
candle_logs = {p: [] for p in PAIRS}
scan_log, trade_log, exit_orders = [], [], []  # TP-only (no SL)
running = False
status = {"msg": "Idle", "last": ""}
status_epoch, error_message = 0, ""
pair_cooldown_until = {p: 0 for p in PAIRS}

# ====== P&L persistence ======
PROFIT_STATE_FILE = "profit_state.json"
profit_state = {"cumulative_pnl": 0.0, "daily": {}, "inventory": {}, "processed_orders": []}

def load_profit_state():
    try:
        with open(PROFIT_STATE_FILE, "r") as f:
            data = json.load(f)
        profit_state.update({
            "cumulative_pnl": float(data.get("cumulative_pnl", 0.0)),
            "daily": dict(data.get("daily", {})),
            "inventory": data.get("inventory", {}),
            "processed_orders": list(data.get("processed_orders", [])),
        })
    except: pass

def save_profit_state():
    out = {
        "cumulative_pnl": round(profit_state.get("cumulative_pnl", 0.0), 6),
        "daily": {k: round(v, 6) for k, v in profit_state.get("daily", {}).items()},
        "inventory": profit_state.get("inventory", {}),
        "processed_orders": profit_state.get("processed_orders", []),
    }
    try:
        with open(PROFIT_STATE_FILE, "w") as f: json.dump(out, f)
    except: pass

def _get_inventory_deque(market):
    dq = deque()
    for lot in profit_state["inventory"].get(market, []):
        try:
            q, c = float(lot[0]), float(lot[1])
            if q > 0 and c > 0: dq.append([q, c])
        except: continue
    return dq

def _set_inventory_from_deque(market, dq):
    profit_state["inventory"][market] = [[float(q), float(c)] for (q,c) in dq]

def apply_fill_update(market, side, price, qty, ts_ms, order_id):
    if not order_id or order_id in profit_state["processed_orders"]: return
    try: price = float(price); qty = float(qty)
    except: return
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
            lot_q -= used; sell_q -= used
            if lot_q <= 1e-18: inv.popleft()
            else: inv[0][0] = lot_q
    _set_inventory_from_deque(market, inv)
    profit_state["processed_orders"].append(order_id)
    profit_state["cumulative_pnl"] = float(profit_state.get("cumulative_pnl", 0.0) + realized)
    dkey = ist_date()
    profit_state["daily"][dkey] = float(profit_state["daily"].get(dkey, 0.0) + realized)
    save_profit_state()

def compute_realized_pnl_today(): return round(profit_state["daily"].get(ist_date(), 0.0), 6)

# ====== Rule helpers ======
def _qp(pair): return int(PAIR_RULES.get(pair, {}).get("precision", 6))
def _pp(pair): return int(PAIR_RULES.get(pair, {}).get("precision", 6))
def _min_qty(pair): return float(PAIR_RULES.get(pair, {}).get("min_qty", 0.0))
def _min_notional(pair): return float(PAIR_RULES.get(pair, {}).get("min_notional", 0.0))  # default 0

def fmt_price(pair, px): return float(f"{float(px):.{_pp(pair)}f}")

def fmt_qty_floor(pair, qty):
    qp = _qp(pair)
    step = 10 ** (-qp)
    q = max(0.0, float(qty or 0.0))
    q = int(q / step) * step
    return float(f"{q:.{qp}f}")

# Learn from order error messages to correct precision / min_qty for next time
def learn_from_order_error(pair, res):
    try:
        if not isinstance(res, dict): return
        msg = res.get("message") or res.get("error") or ""
        if not isinstance(msg, str) or not msg: return

        m = re.search(r'precision\s+should\s+be\s+(\d+)', msg, re.I)
        if m:
            new_qp = int(m.group(1))
            old = _qp(pair)
            if new_qp != old and pair in PAIR_RULES:
                PAIR_RULES[pair]["precision"] = new_qp
                scan_log.append(f"{ist_now()} | learned precision {pair}: {old} -> {new_qp}")

        m2 = re.search(r'Quantity\s+should\s+be\s+greater\s+than\s+([0-9]*\.?[0-9]+)', msg, re.I)
        if m2:
            new_min = float(m2.group(1))
            old_min = _min_qty(pair)
            if new_min > old_min + 1e-18 and pair in PAIR_RULES:
                PAIR_RULES[pair]["min_qty"] = new_min
                scan_log.append(f"{ist_now()} | learned min_qty {pair}: {old_min} -> {new_min}")
    except Exception: pass

# ====== Exchange I/O ======
def hmac_signature(payload): return hmac.new(API_SECRET, payload.encode(), hashlib.sha256).hexdigest()

def get_wallet_balances():
    payload = json.dumps({"timestamp": int(time.time() * 1000)})
    headers = {"X-AUTH-APIKEY": API_KEY, "X-AUTH-SIGNATURE": hmac_signature(payload), "Content-Type":"application/json"}
    balances = {}
    try:
        r = requests.post(f"{BASE_URL}/exchange/v1/users/balances", headers=headers, data=payload, timeout=10)
        if r.ok:
            for b in r.json(): balances[b['currency']] = float(b['balance'])
    except: pass
    return balances

def fetch_all_prices():
    try:
        r = requests.get(f"{BASE_URL}/exchange/ticker", timeout=10)
        if r.ok:
            now = int(time.time())
            return {it["market"]: {"price": float(it["last_price"]), "ts": now}
                    for it in r.json() if it.get("market") in PAIRS}
    except: pass
    return {}

def _signed_post(url, body):
    payload = json.dumps(body, separators=(',', ':'))
    headers = {"X-AUTH-APIKEY": API_KEY, "X-AUTH-SIGNATURE": hmac_signature(payload), "Content-Type":"application/json"}
    try:
        r = requests.post(url, headers=headers, data=payload, timeout=12)
        if r.ok: return r.json()
    except: pass
    return {}

def place_order(pair, side, qty):
    qty = fmt_qty_floor(pair, qty)
    if qty <= 0:
        scan_log.append(f"{ist_now()} | {pair} | ABORT {side}: qty too small")
        return {"status":"error","code":400,"message":"qty <= 0 after precision floor"}
    payload = {"market": pair, "side": side.lower(), "order_type": "market_order",
               "total_quantity": str(qty), "timestamp": int(time.time()*1000)}
    try:
        body = json.dumps(payload, separators=(',', ':'))
        r = requests.post(f"{BASE_URL}/exchange/v1/orders/create",
                          headers={"X-AUTH-APIKEY": API_KEY, "X-AUTH-SIGNATURE": hmac_signature(body),
                                   "Content-Type":"application/json"},
                          data=body, timeout=10)
        res = r.json() if r is not None else {}
        if isinstance(res, dict) and (res.get("status")=="error" or res.get("code")==400 or "message" in res):
            learn_from_order_error(pair, res)
        return res
    except Exception as e:
        err = {"status":"error","message":str(e)}
        learn_from_order_error(pair, err)
        return err

def get_order_status(order_id=None, client_order_id=None):
    body = {"timestamp": int(time.time()*1000)}
    if order_id: body["id"] = order_id
    if client_order_id: body["client_order_id"] = client_order_id
    res = _signed_post(f"{BASE_URL}/exchange/v1/orders/status", body)
    return res if isinstance(res, dict) else {}

def _record_fill_from_status(market, side, st, order_id):
    try:
        total_q  = float(st.get("total_quantity", st.get("quantity", 0)))
        remain_q = float(st.get("remaining_quantity", st.get("remaining_qty", 0)))
        exec_q   = float(st.get("executed_quantity", st.get("filled_qty", 0)))
        filled   = exec_q if exec_q > 0 else max(0.0, total_q - remain_q)
        avg_px   = float(st.get("avg_price", st.get("average_price", st.get("avg_execution_price", st.get("price", 0)))))
    except: filled, avg_px = 0.0, 0.0
    if filled > 0 and avg_px > 0:
        ts_field = st.get("updated_at") or st.get("created_at") or st.get("timestamp") or int(time.time()*1000)
        try:
            ts_ms = int(ts_field);  ts_ms = ts_ms*1000 if ts_ms < 10**12 else ts_ms
        except: ts_ms = int(time.time()*1000)
        apply_fill_update(market, side, avg_px, filled, ts_ms, order_id)

# ====== Candles / Indicators ======
def aggregate_candles(pair, interval_sec):
    t = tick_logs[pair]
    if not t: return
    candles, candle, lastw = [], None, None
    for ts, px in sorted(t, key=lambda x: x[0]):
        w = ts - (ts % interval_sec)
        if w != lastw:
            if candle: candles.append(candle)
            candle = {"open": px, "high": px, "low": px, "close": px, "volume": 1, "start": w}
            lastw = w
        else:
            candle["high"] = max(candle["high"], px); candle["low"] = min(candle["low"], px)
            candle["close"] = px; candle["volume"] += 1
    if candle: candles.append(candle)
    candle_logs[pair] = candles[-100:]  # keep last 100

def _compute_ema(values, n):
    if len(values) < n: return None
    sma = sum(values[:n]) / n
    k = 2 / (n + 1)
    ema = sma
    for v in values[n:]: ema = v * k + ema * (1 - k)
    return ema

def pa_buy_sell_signal(pair, live_price=None):
    # Donchian(5) breakout + EMA 5/13 trend; fixed TP logic handled later
    candles = candle_logs[pair]
    if len(candles) < 25: return None
    completed = candles[:-1] if len(candles) >= 2 else candles
    if len(completed) < 20: return None

    closes = [c["close"] for c in completed]
    curr_price = float(live_price) if live_price else candles[-1]["close"]

    N = 5
    recent = completed[-N:]
    don_high = max(c["high"] for c in recent)
    don_low  = min(c["low"]  for c in recent)

    closes_plus_live = closes[-30:] + [curr_price]
    ema_fast = _compute_ema(closes_plus_live, 5)
    ema_slow = _compute_ema(closes_plus_live, 13)
    if ema_fast is None or ema_slow is None: return None

    if curr_price > don_high and ema_fast > ema_slow:
        return {"side":"BUY","entry":curr_price, "msg":f"BUY: breakout > Donchian({N}) & EMA5>EMA13"}
    if curr_price < don_low and ema_fast < ema_slow:
        return {"side":"SELL","entry":curr_price, "msg":f"SELL: breakdown < Donchian({N}) & EMA5<EMA13"}
    return None

# ====== Exits (TP-only) ======
def monitor_exits(prices):
    global error_message
    to_remove = []
    for ex in exit_orders:
        pair, side, qty, tp = ex.get("pair"), ex.get("side"), ex.get("qty"), ex.get("tp")
        price = prices.get(pair, {}).get("price")
        if price is None: continue
        qx = fmt_qty_floor(pair, qty)

        if side == "BUY" and price >= tp:
            res = place_order(pair, "SELL", qx)
            scan_log.append(f"{ist_now()} | {pair} | EXIT SELL {qx} @ {price} | {res}")
            if isinstance(res, dict) and (res.get("status")=="error" or "message" in res): learn_from_order_error(pair, res)
            try: order_id = (res.get("orders") or [{}])[0].get("id")
            except: order_id = None
            if order_id: _record_fill_from_status(pair, "SELL", get_order_status(order_id=order_id), order_id)
            if isinstance(res, dict) and "error" in res: error_message = res["error"]
            to_remove.append(ex); pair_cooldown_until[pair] = int(time.time()) + TRADE_COOLDOWN_SEC

        elif side == "SELL" and price <= tp:
            res = place_order(pair, "BUY", qx)
            scan_log.append(f"{ist_now()} | {pair} | EXIT BUY {qx} @ {price} | {res}")
            if isinstance(res, dict) and (res.get("status")=="error" or "message" in res): learn_from_order_error(pair, res)
            try: order_id = (res.get("orders") or [{}])[0].get("id")
            except: order_id = None
            if order_id: _record_fill_from_status(pair, "BUY", get_order_status(order_id=order_id), order_id)
            if isinstance(res, dict) and "error" in res: error_message = res["error"]
            to_remove.append(ex); pair_cooldown_until[pair] = int(time.time()) + TRADE_COOLDOWN_SEC

    for ex in to_remove: exit_orders.remove(ex)

def _has_open_exit_for(pair):
    return any(ex.get("pair")==pair for ex in exit_orders)

# ====== Main loop ======
def scan_loop():
    global running, error_message, status_epoch, _last_keepalive
    scan_log.clear()
    while running:
        # self keep-alive
        if APP_BASE_URL and (time.time() - _last_keepalive) >= KEEPALIVE_SEC:
            _keepalive_ping(); _last_keepalive = time.time()

        prices = fetch_all_prices()
        now = int(time.time())
        monitor_exits(prices)
        balances = get_wallet_balances()

        for pair in PAIRS:
            if pair not in prices: continue
            price = prices[pair]["price"]
            tick_logs[pair].append((now, price))
            if len(tick_logs[pair]) > 5000: tick_logs[pair] = tick_logs[pair][-5000:]

            # build/refresh candles (interval from SETTINGS)
            aggregate_candles(pair, SETTINGS["candle_interval_sec"])
            last_candle = candle_logs[pair][-1] if candle_logs[pair] else None

            if last_candle:
                if int(time.time()) < pair_cooldown_until.get(pair, 0) or _has_open_exit_for(pair):
                    scan_log.append(f"{ist_now()} | {pair} | Cooldown/Exit pending — skip")
                else:
                    signal = pa_buy_sell_signal(pair, price)
                    if signal:
                        error_message = ""
                        entry = float(signal["entry"])
                        usdt_bal = float(balances.get("USDT", 0.0))

                        # TP fixed at +/- TP%
                        tp_pct = float(SETTINGS["tp_pct"])
                        if signal["side"] == "BUY":
                            tp = fmt_price(pair, entry * (1.0 + tp_pct))
                            pre_qty = (0.3 * usdt_bal) / entry
                        else:
                            tp = fmt_price(pair, entry * (1.0 - tp_pct))
                            coin = pair[:-4]
                            pre_qty = float(balances.get(coin, 0.0))  # inventory-based shorts only

                        # exchange constraints
                        qty = pre_qty
                        # ensure min_qty & min_notional
                        min_q = _min_qty(pair); min_notional = _min_notional(pair)
                        if min_notional > 0:
                            qty = max(qty, min_notional / entry)
                        qty = max(qty, min_q)
                        qty = fmt_qty_floor(pair, qty)

                        if qty <= 0:
                            scan_log.append(f"{ist_now()} | {pair} | {signal['side']} skipped: cannot meet min rules/funds")
                        else:
                            res = place_order(pair, signal["side"], qty)
                            scan_log.append(f"{ist_now()} | {pair} | {signal['side']} @ {entry} | TP {tp} | {res}")
                            if isinstance(res, dict) and (res.get("status")=="error" or "message" in res):
                                learn_from_order_error(pair, res)

                            trade_log.append({
                                "time": ist_now(), "pair": pair, "side": signal["side"], "entry": entry,
                                "msg": signal["msg"], "tp": tp, "qty": qty, "order_result": res
                            })
                            exit_orders.append({"pair": pair, "side": signal["side"], "qty": qty, "tp": tp, "entry": entry})

                            # if we have an order id, record fills into P&L
                            try: order_id = (res.get("orders") or [{}])[0].get("id")
                            except: order_id = None
                            if order_id:
                                st = get_order_status(order_id=order_id)
                                _record_fill_from_status(pair, signal["side"], st, order_id)

                            if isinstance(res, dict) and "error" in res:
                                error_message = res["error"]
                    else:
                        scan_log.append(f"{ist_now()} | {pair} | No Signal")

        status["msg"], status["last"] = "Running", ist_now()
        status_epoch = int(time.time())
        time.sleep(5)

    status["msg"] = "Idle"

# ====== Routes ======
@app.route("/")
def index(): return render_template("index.html")

@app.route("/start", methods=["POST"])
def start():
    global running
    if not running:
        running = True
        threading.Thread(target=scan_loop, daemon=True).start()
    return jsonify({"status": "started"})

@app.route("/stop", methods=["POST"])
def stop():
    global running; running = False
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
        "status_epoch": status_epoch,
        "usdt": balances.get("USDT", 0.0),
        "profit_today": profit_today,
        "profit_yesterday": profit_yesterday,
        "pnl_cumulative": cumulative_pnl,
        "coins": coins,
        "trades": trade_log[-12:][::-1],
        "scans": scan_log[-60:][::-1],
        "error": error_message,
        "settings": {
            "candle_interval_sec": SETTINGS["candle_interval_sec"],
            "tp_pct": SETTINGS["tp_pct"]
        }
    })

@app.route("/config", methods=["GET", "POST"])
def config():
    if request.method == "GET":
        return jsonify({"ok": True, **SETTINGS})
    # POST: accept {"candle_interval_sec": int, "tp_pct": float}
    try:
        data = request.get_json(force=True) or {}
        ci = int(data.get("candle_interval_sec", SETTINGS["candle_interval_sec"]))
        tp = float(data.get("tp_pct", SETTINGS["tp_pct"]))
        # sane bounds
        if ci < 15: ci = 15
        if ci > 3600*6: ci = 3600*6
        if tp < 0.001: tp = 0.001     # 0.1% min
        if tp > 0.05: tp = 0.05       # 5% max
        SETTINGS["candle_interval_sec"] = ci
        SETTINGS["tp_pct"] = tp
        scan_log.append(f"{ist_now()} | SETTINGS updated: candle={ci}s, tp_pct={tp}")
        return jsonify({"ok": True, **SETTINGS})
    except Exception as e:
        return jsonify({"ok": False, "error": str(e)}), 400

@app.route("/ping")
def ping(): return "pong"

# ====== Boot ======
if __name__ == "__main__":
    load_profit_state()
    app.run(host="0.0.0.0", port=int(os.environ.get("PORT", "10000")))
