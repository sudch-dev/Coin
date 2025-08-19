import os
import time
import threading
import hmac
import hashlib
import requests
import json
import traceback
from flask import Flask, render_template, jsonify
from datetime import datetime, timedelta
from pytz import timezone
from collections import deque
from werkzeug.middleware.proxy_fix import ProxyFix

app = Flask(__name__)
# Trust Render's proxy headers so Flask knows it's behind HTTPS
app.wsgi_app = ProxyFix(app.wsgi_app, x_for=1, x_proto=1, x_host=1, x_port=1, x_prefix=1)

# If you host at a different domain later, set APP_BASE_URL in Render env
APP_BASE_URL = os.environ.get("APP_BASE_URL", "https://coin-4k37.onrender.com")

API_KEY = os.environ.get("API_KEY")
API_SECRET_RAW = os.environ.get("API_SECRET", "")
API_SECRET = API_SECRET_RAW.encode() if isinstance(API_SECRET_RAW, str) else API_SECRET_RAW
BASE_URL = "https://api.coindcx.com"

PAIRS = [
    "BTCUSDT", "ETHUSDT", "XRPUSDT", "SHIBUSDT", "SOLUSDT",
    "DOGEUSDT", "ADAUSDT", "AEROUSDT", "BNBUSDT", "LTCUSDT"
]

# You can refine these with live market details; we also fetch precision on boot
PAIR_RULES = {
    "BTCUSDT": {"precision": 6, "min_qty": 0.0005},
    "ETHUSDT": {"precision": 6, "min_qty": 0.0001},
    "XRPUSDT": {"precision": 1, "min_qty": 1},
    "SHIBUSDT": {"precision": 0, "min_qty": 10000},
    "DOGEUSDT": {"precision": 1, "min_qty": 5},
    "SOLUSDT": {"precision": 4, "min_qty": 0.01},
    "AEROUSDT": {"precision": 2, "min_qty": 1},
    "ADAUSDT": {"precision": 1, "min_qty": 2},
    "LTCUSDT": {"precision": 4, "min_qty": 0.01},
    "BNBUSDT": {"precision": 4, "min_qty": 0.01}
}

# ======= MODE =======
MODE = "maker"        # "maker" (HFT-style quoting)

# ======= HFT / Maker Tunables =======
CANDLE_INTERVAL = 10            # 10s candles for light trend context
POLL_SEC = 1.0                  # quote management cadence
QUOTE_TTL_SEC = 12              # replace quotes after this age
DRIFT_REQUOTE_PCT = 0.0008      # reprice if last moves > 0.08% from our quote
FEE_PCT_PER_SIDE = 0.0010       # 0.10% fee assumption (set to your actual tier)
TP_BUFFER_PCT = 0.0003          # profit buffer above round-trip fees
SPREAD_OFFSET_PCT = None        # if None, auto = (2*FEE + TP_BUFFER)/2 per side
EMA_FAST = 5
EMA_SLOW = 20

# Inventory & sizing
MAX_PER_PAIR_USDT = 50.0        # hard cap per pair in USDT
QUOTE_USDT = 20.0               # size per quote in USDT (pre-precision/min filters)
INVENTORY_USDT_CAP = 200.0      # total gross inventory cap across pairs
INVENTORY_REDUCE_BIAS = 1.6     # multiplier on quote side that reduces inventory

# Time stop for orphan inventory (no SL/TP; last resort flatten if stuck too long)
ORPHAN_MAX_SEC = 240

# Self-keepalive ping cadence (prevents Render sleep)
KEEPALIVE_SEC = 240

IST = timezone('Asia/Kolkata')
def ist_now(): return datetime.now(IST).strftime('%Y-%m-%d %H:%M:%S')
def ist_date(): return datetime.now(IST).strftime('%Y-%m-%d')
def ist_yesterday(): return (datetime.now(IST) - timedelta(days=1)).strftime('%Y-%m-%d')

tick_logs, candle_logs = {p: [] for p in PAIRS}, {p: [] for p in PAIRS}
scan_log, trade_log = [], []
daily_profit, pair_precision = {}, {}
running = False
status = {"msg": "Idle", "last": ""}
status_epoch = 0
error_message = ""

# ===== Persistent P&L (FIFO spot; fills only) =====
PROFIT_STATE_FILE = "profit_state.json"
profit_state = {
    "cumulative_pnl": 0.0,    # USDT
    "daily": {},
    "inventory": {},          # market -> [[qty, avg_price], ...]
    "processed_orders": []    # order ids applied
}

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

# ===== HTTP helpers & protocol-safe client =====
def hmac_signature(payload):
    return hmac.new(API_SECRET, payload.encode(), hashlib.sha256).hexdigest()

def _log_http_issue(prefix, r):
    try:
        ct = r.headers.get("content-type", "")
        body = r.text[:240] if hasattr(r, "text") else ""
        scan_log.append(f"{ist_now()} | {prefix} HTTP {r.status_code} | {body}")
    except Exception as e:
        scan_log.append(f"{ist_now()} | {prefix} log-fail: {e}")

def _signed_post(url, body):
    payload = json.dumps(body, separators=(',', ':'))
    sig = hmac_signature(payload)
    headers = {
        "X-AUTH-APIKEY": API_KEY or "",
        "X-AUTH-SIGNATURE": sig,
        "Content-Type": "application/json"
    }
    try:
        r = requests.post(url, headers=headers, data=payload, timeout=12)  # HTTPS endpoint
        if not r.ok:
            _log_http_issue(f"POST {url}", r)
        if r.headers.get("content-type", "").startswith("application/json"):
            return r.json()
        return {}
    except Exception as e:
        scan_log.append(f"{ist_now()} | POST fail {url} | {e.__class__.__name__}: {e}")
        tb = traceback.format_exc().splitlines()[-1]
        scan_log.append(tb)
        return {}

def _safe_get(url, timeout=10):
    try:
        r = requests.get(url, timeout=timeout)
        if not r.ok:
            _log_http_issue(f"GET {url}", r)
        if r.headers.get("content-type", "").startswith("application/json"):
            return r.json()
        return r
    except Exception as e:
        scan_log.append(f"{ist_now()} | GET fail {url} | {e.__class__.__name__}: {e}")
        return {}

def _keepalive_ping():
    try:
        requests.get(f"{APP_BASE_URL}/ping", timeout=5)
    except Exception:
        pass

def fetch_pair_precisions():
    try:
        r = requests.get(f"{BASE_URL}/exchange/v1/markets_details", timeout=10)
        if r.ok:
            for item in r.json():
                if item.get("pair") in PAIRS:
                    pair_precision[item["pair"]] = int(item.get("target_currency_precision", 6))
    except Exception as e:
        scan_log.append(f"{ist_now()} | markets_details fail: {e}")

def get_wallet_balances():
    payload = json.dumps({"timestamp": int(time.time() * 1000)})
    sig = hmac_signature(payload)
    headers = {"X-AUTH-APIKEY": API_KEY or "", "X-AUTH-SIGNATURE": sig, "Content-Type": "application/json"}
    balances = {}
    try:
        r = requests.post(f"{BASE_URL}/exchange/v1/users/balances", headers=headers, data=payload, timeout=10)
        if r.ok:
            for b in r.json():
                balances[b['currency']] = float(b['balance'])
        else:
            _log_http_issue("balances", r)
    except Exception as e:
        scan_log.append(f"{ist_now()} | balances fail: {e}")
    return balances

def fetch_all_prices():
    try:
        r = requests.get(f"{BASE_URL}/exchange/ticker", timeout=10)
        if r.ok:
            now = int(time.time())
            return {item["market"]: {"price": float(item["last_price"]), "ts": now}
                    for item in r.json() if item.get("market") in PAIRS}
        else:
            _log_http_issue("ticker", r)
    except Exception as e:
        scan_log.append(f"{ist_now()} | ticker fail: {e}")
    return {}

# ===== Candle builder for soft trend context =====
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
            candle["close"] = price
            candle["volume"] += 1
    if candle: candles.append(candle)
    candle_logs[pair] = candles[-240:]  # ~40m of 10s bars

def _ema(vals, n):
    if len(vals) < n: return None
    k = 2/(n+1)
    ema = sum(vals[:n])/n
    for v in vals[n:]:
        ema = v*k + ema*(1-k)
    return ema

# ====== Maker Quote Engine ======
active_quotes = {p: {"bid": None, "ask": None} for p in PAIRS}  # {"id","px","qty","ts"}
inventory_timer = {}  # market -> ts when long inventory first observed

def place_order(pair, side, qty):
    payload = {
        "market": pair,
        "side": side.lower(),
        "order_type": "market_order",
        "total_quantity": str(qty),
        "timestamp": int(time.time() * 1000)
    }
    return _signed_post(f"{BASE_URL}/exchange/v1/orders/create", payload) or {}

def place_limit_order(pair, side, qty, price):
    payload = {
        "market": pair,
        "side": side.lower(),
        "order_type": "limit_order",
        "price_per_unit": str(price),
        "total_quantity": str(qty),
        "timestamp": int(time.time() * 1000)
        # If supported by CoinDCX, add post-only to avoid taker fills:
        # "time_in_force": "PO"  OR  "post_only": True
    }
    return _signed_post(f"{BASE_URL}/exchange/v1/orders/create", payload) or {}

def cancel_order(order_id=None, client_order_id=None):
    body = {"timestamp": int(time.time() * 1000)}
    if order_id: body["id"] = order_id
    if client_order_id: body["client_order_id"] = client_order_id
    return _signed_post(f"{BASE_URL}/exchange/v1/orders/cancel", body) or {}

def get_order_status(order_id=None, client_order_id=None):
    body = {"timestamp": int(time.time() * 1000)}
    if order_id: body["id"] = order_id
    if client_order_id: body["client_order_id"] = client_order_id
    res = _signed_post(f"{BASE_URL}/exchange/v1/orders/status", body)
    return res if isinstance(res, dict) else {}

def _extract_order_id(res: dict):
    if not isinstance(res, dict): return None
    try:
        if isinstance(res.get("orders"), list) and res["orders"]:
            o = res["orders"][0]
            return str(o.get("id") or o.get("order_id") or o.get("client_order_id") or "")
    except:
        pass
    return str(res.get("id") or res.get("order_id") or res.get("client_order_id") or res.get("orderId") or "") or None

def _fnum(x, d=0.0):
    try: return float(x)
    except: return d

def _record_fill_from_status(market, side, st, order_id):
    if not isinstance(st, dict): return
    total_q  = _fnum(st.get("total_quantity", st.get("quantity", st.get("orig_qty", 0))))
    remain_q = _fnum(st.get("remaining_quantity", st.get("remaining_qty", st.get("leaves_qty", 0))))
    exec_q   = _fnum(st.get("executed_quantity", st.get("filled_qty", st.get("executedQty", 0))))
    filled   = exec_q if exec_q > 0 else max(0.0, total_q - remain_q)
    avg_px   = _fnum(st.get("avg_price", st.get("average_price", st.get("avg_execution_price", st.get("price", 0)))))
    if filled > 0 and avg_px > 0:
        ts_field = st.get("updated_at") or st.get("created_at") or st.get("timestamp") or int(time.time()*1000)
        try:
            ts_ms = int(ts_field)
            if ts_ms < 10**12: ts_ms *= 1000
        except:
            ts_ms = int(time.time() * 1000)
        apply_fill_update(market, side, avg_px, filled, ts_ms, order_id)

def _effective_half_spread_pct():
    base = (2*FEE_PCT_PER_SIDE + TP_BUFFER_PCT) / 2.0
    return max(base, (SPREAD_OFFSET_PCT or 0.0))

def _quote_prices(last):
    off = _effective_half_spread_pct()
    bid = round(last * (1.0 - off), 6)
    ask = round(last * (1.0 + off), 6)
    return bid, ask

def _qty_for_pair(pair, price, usdt_avail, coin_avail, side, trend_skew):
    q = max(1e-12, QUOTE_USDT) / max(price, 1e-9)
    q *= trend_skew
    rule = PAIR_RULES.get(pair, {"precision": 6, "min_qty": 0.0001})
    q = min(q, MAX_PER_PAIR_USDT / max(price, 1e-9))
    if side == "BUY":
        q = min(q, usdt_avail / max(price, 1e-9))
    else:
        q = min(q, coin_avail)
    q = max(q, rule["min_qty"])
    q = round(q, rule["precision"])
    return q

def _net_inventory_units(pair):
    dq = _get_inventory_deque(pair)
    return sum(q for q, _ in dq)

def _gross_inventory_usdt(prices):
    total = 0.0
    for p in PAIRS:
        q = abs(_net_inventory_units(p))
        px = prices.get(p, {}).get("price", 0.0)
        total += q * px
    return total

def _cancel_quote(pair, side):
    q = active_quotes[pair][side]
    if not q: return
    oid = q.get("id")
    if oid:
        cancel_order(order_id=oid)
        scan_log.append(f"{ist_now()} | {pair} | cancel {side} quote {oid}")
    active_quotes[pair][side] = None

def _place_quote(pair, side, price, qty):
    res = place_limit_order(pair, side, qty, price)
    oid = _extract_order_id(res)
    active_quotes[pair][side] = {"id": oid, "px": price, "qty": qty, "ts": int(time.time())}
    scan_log.append(f"{ist_now()} | {pair} | quote {side} {qty} @ {price} | id={oid} | res={res}")
    return oid

def _check_quote_fill(pair, side):
    q = active_quotes[pair][side]
    if not q or not q.get("id"): return False
    st = get_order_status(order_id=q["id"])
    rem = _fnum(st.get("remaining_quantity", st.get("remaining_qty", st.get("leaves_qty", 0))))
    status_txt = (st.get("status") or "").lower()
    filled = (rem == 0) or ("filled" in status_txt and "part" not in status_txt)
    if filled:
        _record_fill_from_status(pair, side, st, q["id"])
        active_quotes[pair][side] = None
        scan_log.append(f"{ist_now()} | {pair} | FILL {side} | st={st}")
        return True
    return False

def _manage_orphan_inventory(pair, now_ts, prices, balances):
    q_units = _net_inventory_units(pair)
    if q_units <= 0:
        inventory_timer.pop(pair, None)
        return
    if pair not in inventory_timer:
        inventory_timer[pair] = now_ts
        return
    if now_ts - inventory_timer[pair] < ORPHAN_MAX_SEC:
        return
    # Last resort: market-out partial inventory
    coin = pair[:-4]
    coin_bal = balances.get(coin, 0.0)
    qty = min(0.3 * q_units, coin_bal)
    rule = PAIR_RULES.get(pair, {"precision": 6, "min_qty": 0.0001})
    qty = max(qty, rule["min_qty"])
    qty = round(qty, rule["precision"])
    if qty <= 0: return
    res = place_order(pair, "SELL", qty)
    oid = _extract_order_id(res)
    if oid:
        st = get_order_status(order_id=oid)
        _record_fill_from_status(pair, "SELL", st, oid)
    scan_log.append(f"{ist_now()} | {pair} | ORPHAN TIMEOUT market SELL {qty} | res={res}")
    inventory_timer[pair] = now_ts  # reset timer for remaining

# ======== Main Loop ========
last_keepalive = 0

def scan_loop():
    global running, error_message, status_epoch, last_keepalive
    scan_log.clear()
    last_tick_ts = {p: 0 for p in PAIRS}
    running = True

    while running:
        # keepalive ping to prevent Render sleep
        now_real = time.time()
        if now_real - last_keepalive >= KEEPALIVE_SEC:
            _keepalive_ping()
            last_keepalive = now_real

        prices = fetch_all_prices()
        now = int(time.time())
        balances = get_wallet_balances()

        # Build ticks & candles
        for pair in PAIRS:
            if pair in prices:
                ts = prices[pair]["ts"]
                px = prices[pair]["price"]
                if ts != last_tick_ts[pair]:
                    last_tick_ts[pair] = ts
                tick_logs[pair].append((now, px))
                if len(tick_logs[pair]) > 4000:
                    tick_logs[pair] = tick_logs[pair][-4000:]
                aggregate_candles(pair, CANDLE_INTERVAL)

        # Total gross inventory guard
        gross_usdt = _gross_inventory_usdt(prices)
        if gross_usdt > INVENTORY_USDT_CAP:
            scan_log.append(f"{ist_now()} | RISK | Gross inventory {round(gross_usdt,2)} > cap {INVENTORY_USDT_CAP} — pausing new bids")

        # Quote management per pair
        for pair in PAIRS:
            if pair not in prices:
                continue

            last = prices[pair]["price"]
            bid_px, ask_px = _quote_prices(last)

            # Trend skew from EMAs (soft)
            closes = [c["close"] for c in (candle_logs.get(pair) or [])[:-1]]
            ema_fast = _ema(closes[-(EMA_SLOW+EMA_FAST):] + [last], EMA_FAST) if len(closes) >= EMA_SLOW else None
            ema_slow = _ema(closes[-(EMA_SLOW+EMA_FAST):] + [last], EMA_SLOW) if len(closes) >= EMA_SLOW else None
            bullish = (ema_fast is not None and ema_slow is not None and ema_fast >= ema_slow)

            # Balances & inventory
            usdt = balances.get("USDT", 0.0)
            coin = pair[:-4]
            coin_bal = balances.get(coin, 0.0)
            net_units = _net_inventory_units(pair)

            # Decide sizes with inventory bias
            reduce_bias_bid = INVENTORY_REDUCE_BIAS if net_units < 0 else 1.0  # (short on spot is unusual; kept for symmetry)
            reduce_bias_ask = INVENTORY_REDUCE_BIAS if net_units > 0 else 1.0  # if long, bias ask to reduce
            trend_bias_bid = 1.2 if bullish else 0.9
            trend_bias_ask = 1.2 if not bullish else 0.9

            qty_bid = _qty_for_pair(pair, bid_px, usdt, coin_bal, "BUY", reduce_bias_bid * trend_bias_bid)
            qty_ask = _qty_for_pair(pair, ask_px, usdt, coin_bal, "SELL", reduce_bias_ask * trend_bias_ask)

            # If gross cap breached, don't place new BUYs
            if gross_usdt > INVENTORY_USDT_CAP:
                qty_bid = 0.0

            # Manage existing quotes: cancel/refresh on TTL or drift
            for side, q in list(active_quotes[pair].items()):
                if not q:
                    continue
                age = now - int(q.get("ts", now))
                px = q.get("px", last)
                drift = abs(last - px) / max(px, 1e-9)
                if age >= QUOTE_TTL_SEC or drift >= DRIFT_REQUOTE_PCT:
                    _cancel_quote(pair, side)

            # Check fills on remaining quotes
            for side in ("bid", "ask"):
                _check_quote_fill(pair, "BUY" if side == "bid" else "SELL")

            # Place/replace quotes if empty
            if qty_bid > 0 and not active_quotes[pair]["bid"]:
                bpx = min(bid_px, round(last * (1 - 1e-6), 6))  # ensure not crossing
                _place_quote(pair, "BUY", bpx, qty_bid)

            if qty_ask > 0 and coin_bal >= PAIR_RULES.get(pair, {"min_qty": 0.0})["min_qty"]:
                if not active_quotes[pair]["ask"]:
                    apx = max(ask_px, round(last * (1 + 1e-6), 6))  # ensure not crossing
                    _place_quote(pair, "SELL", apx, qty_ask)

            # Orphan inventory protection
            _manage_orphan_inventory(pair, now, prices, balances)

        status["msg"], status["last"] = "Running", ist_now()
        status_epoch = int(time.time())

        # Loop heartbeat (helps detect if loop is alive in logs)
        print(f"[{ist_now()}] Loop active — quoting…")

        time.sleep(POLL_SEC)

    status["msg"] = "Idle"

def compute_realized_pnl_today():
    return round(profit_state["daily"].get(ist_date(), 0.0), 6)

# ===== Routes =====
@app.route("/")
def index():
    return render_template("index.html")

@app.route("/start", methods=["POST"])
def start():
    _start_loop_once()
    return jsonify({"status": "started"})

@app.route("/stop", methods=["POST"])
def stop():
    global running
    running = False
    # cancel outstanding quotes on stop
    try:
        for p in PAIRS:
            for s in ("bid", "ask"):
                if active_quotes[p][s] and active_quotes[p][s].get("id"):
                    cancel_order(order_id=active_quotes[p][s]["id"])
                    active_quotes[p][s] = None
    except:
        pass
    return jsonify({"status": "stopped"})

@app.route("/status")
def get_status():
    balances = get_wallet_balances()
    coins = {pair[:-4]: balances.get(pair[:-4], 0.0) for pair in PAIRS}
    profit_today = compute_realized_pnl_today()
    profit_yesterday = round(profit_state["daily"].get(ist_yesterday(), 0.0), 6)
    cumulative_pnl = round(profit_state.get("cumulative_pnl", 0.0), 6)

    visible_quotes = {}
    for p in PAIRS:
        v = {}
        for side in ("bid", "ask"):
            q = active_quotes[p][side]
            if q:
                v[side] = {"id": q.get("id"), "px": q.get("px"), "qty": q.get("qty"), "ts": q.get("ts")}
        if v:
            visible_quotes[p] = v

    return jsonify({
        "status": status["msg"],
        "last": status["last"],
        "status_epoch": status_epoch,
        "usdt": balances.get("USDT", 0.0),
        "profit_today": profit_today,
        "profit_yesterday": profit_yesterday,
        "pnl_cumulative": cumulative_pnl,
        "processed_orders": len(profit_state.get("processed_orders", [])),
        "inventory_markets": list(profit_state.get("inventory", {}).keys()),
        "quotes": visible_quotes,
        "coins": coins,
        "trades": trade_log[-10:][::-1],
        "scans": scan_log[-60:][::-1],
        "error": error_message
    })

@app.route("/ping")
def ping(): 
    return "pong"

# ===== Dual autostart (boot + first request) =====
def _start_loop_once():
    global running
    if not running:
        running = True
        t = threading.Thread(target=scan_loop, daemon=True)
        t.start()

@app.before_first_request
def _kick_on_first_request():
    _start_loop_once()

if __name__ == "__main__":
    load_profit_state()
    fetch_pair_precisions()
    _start_loop_once()  # start even before first web request
    app.run(host="0.0.0.0", port=10000)
