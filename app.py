import os
import time
import threading
import hmac
import hashlib
import requests
import json
import traceback
from flask import Flask, render_template, jsonify, request
from datetime import datetime, timedelta
from pytz import timezone
from collections import deque

# -------------------- Flask --------------------
app = Flask(__name__)

APP_BASE_URL   = os.environ.get("APP_BASE_URL", "")
KEEPALIVE_TOKEN= os.environ.get("KEEPALIVE_TOKEN", "")

# -------------------- API / Markets --------------------
API_KEY        = os.environ.get("API_KEY", "")
API_SECRET_RAW = os.environ.get("API_SECRET", "")
API_SECRET     = API_SECRET_RAW.encode() if isinstance(API_SECRET_RAW, str) else API_SECRET_RAW
BASE_URL       = "https://api.coindcx.com"

# Pairs (LTC removed)
PAIRS = ["BTCUSDT","ETHUSDT","XRPUSDT","SHIBUSDT","SOLUSDT","DOGEUSDT","ADAUSDT","AEROUSDT","BNBUSDT"]

# Fallback rules; will be refreshed from exchange at boot and periodically
PAIR_RULES = {
    "BTCUSDT":  {"price_precision": 1, "qty_precision": 4, "min_qty": 0.001,  "min_notional": 0.0},
    "ETHUSDT":  {"price_precision": 2, "qty_precision": 6, "min_qty": 0.0001, "min_notional": 0.0},
    "XRPUSDT":  {"price_precision": 4, "qty_precision": 4, "min_qty": 0.1,    "min_notional": 0.0},
    "SHIBUSDT": {"price_precision": 8, "qty_precision": 4, "min_qty": 10000,  "min_notional": 0.0},
    "DOGEUSDT": {"price_precision": 5, "qty_precision": 4, "min_qty": 0.001,  "min_notional": 0.0},
    "SOLUSDT":  {"price_precision": 2, "qty_precision": 4, "min_qty": 0.01,   "min_notional": 0.0},
    "AEROUSDT": {"price_precision": 3, "qty_precision": 2, "min_qty": 0.01,   "min_notional": 0.0},
    "ADAUSDT":  {"price_precision": 4, "qty_precision": 2, "min_qty": 0.1,    "min_notional": 0.0},
    "BNBUSDT":  {"price_precision": 3, "qty_precision": 4, "min_qty": 0.001,  "min_notional": 0.0},
}

# -------------------- Core settings --------------------
CANDLE_INTERVAL = 5           # seconds
POLL_SEC        = 1.0
KEEPALIVE_SEC   = 240

# fees & rounding
FEE_PCT_PER_SIDE= 0.0010      # 0.10% per side—adjust if your account differs
BUY_HEADROOM    = 1.0005      # tiny buffer so rounding+fees still fit

# signal windows
EMA_FAST        = 5
EMA_SLOW        = 20
MACD_FAST       = 12
MACD_SLOW       = 26
MACD_SIGNAL     = 9

# position sizing
BUY_USDT_FRAC   = 0.30        # buy 30% of free USDT when signal fires

# exits
TP_PCT          = 0.005       # +0.5% target
NO_SL           = True        # explicit: no stop loss

# rules refresh cadence
RULES_REFRESH_SEC = 1800

# -------------------- Time helpers --------------------
IST = timezone('Asia/Kolkata')
def ist_now():        return datetime.now(IST).strftime('%Y-%m-%d %H:%M:%S')
def ist_date():       return datetime.now(IST).strftime('%Y-%m-%d')
def ist_yesterday():  return (datetime.now(IST) - timedelta(days=1)).strftime('%Y-%m-%d')

# -------------------- State --------------------
tick_logs   = {p: [] for p in PAIRS}    # (ts, px)
candle_logs = {p: [] for p in PAIRS}    # list of dicts per pair
scan_log    = []                        # text lines for UI
trade_log   = []                        # optional manual additions
running     = False
status      = {"msg":"Idle","last":""}
status_epoch= 0
error_message = ""

# TP orders we opened: {pair: {"id":..., "px":..., "qty":..., "ts":...}}
open_tp = {p: None for p in PAIRS}

# profit state (FIFO)
PROFIT_STATE_FILE = "profit_state.json"
profit_state = {"cumulative_pnl":0.0,"daily":{},"inventory":{},"processed_orders":[]}

# keepalive tracker
last_keepalive = 0

# -------------------- PnL helpers (FIFO) --------------------
from collections import deque

def load_profit_state():
    global profit_state
    try:
        with open(PROFIT_STATE_FILE,"r") as f:
            data = json.load(f)
        profit_state["cumulative_pnl"] = float(data.get("cumulative_pnl",0.0))
        profit_state["daily"]          = dict(data.get("daily",{}))
        profit_state["inventory"]      = data.get("inventory",{})
        profit_state["processed_orders"]= list(data.get("processed_orders",[]))
    except:
        pass

def save_profit_state():
    try:
        with open(PROFIT_STATE_FILE,"w") as f:
            json.dump({
                "cumulative_pnl": round(profit_state.get("cumulative_pnl",0.0),6),
                "daily": {k:round(v,6) for k,v in profit_state.get("daily",{}).items()},
                "inventory": profit_state.get("inventory",{}),
                "processed_orders": profit_state.get("processed_orders",[])
            }, f)
    except:
        pass

def _get_inventory_deque(market):
    dq = deque()
    for lot in profit_state["inventory"].get(market, []):
        try:
            q,c = float(lot[0]), float(lot[1])
            if q>0 and c>0: dq.append([q,c])
        except:
            continue
    return dq

def _set_inventory_from_deque(market, dq):
    profit_state["inventory"][market] = [[float(q),float(c)] for q,c in dq]

def apply_fill_update(market, side, price, qty, ts_ms, order_id):
    # de-dup
    if (not order_id) or order_id in profit_state["processed_orders"]:
        return
    try:
        price = float(price); qty = float(qty)
        if price<=0 or qty<=0: return
    except: return

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
            if lot_q <= 1e-18: inv.popleft()
            else: inv[0][0] = lot_q

    _set_inventory_from_deque(market, inv)
    profit_state["processed_orders"].append(order_id)
    profit_state["cumulative_pnl"] = float(profit_state.get("cumulative_pnl",0.0) + realized)
    dkey = ist_date()
    profit_state["daily"][dkey] = float(profit_state["daily"].get(dkey,0.0) + realized)
    save_profit_state()

def compute_realized_pnl_today():
    return round(profit_state["daily"].get(ist_date(),0.0),6)

# -------------------- HTTP helpers --------------------
def hmac_signature(payload):
    return hmac.new(API_SECRET, payload.encode(), hashlib.sha256).hexdigest()

def _log_http_issue(prefix, r):
    try:
        body = r.text[:240] if hasattr(r,"text") else ""
        scan_log.append(f"{ist_now()} | {prefix} HTTP {r.status_code} | {body}")
    except Exception as e:
        scan_log.append(f"{ist_now()} | {prefix} log-fail: {e}")

def _signed_post(url, body):
    payload = json.dumps(body, separators=(',',':'))
    sig = hmac_signature(payload)
    headers = {"X-AUTH-APIKEY": API_KEY, "X-AUTH-SIGNATURE": sig, "Content-Type":"application/json"}
    try:
        r = requests.post(url, headers=headers, data=payload, timeout=12)
        if not r.ok: _log_http_issue(f"POST {url}", r)
        if r.headers.get("content-type","").startswith("application/json"):
            return r.json()
        return {}
    except Exception as e:
        scan_log.append(f"{ist_now()} | POST fail {url} | {e.__class__.__name__}: {e}")
        scan_log.append(traceback.format_exc().splitlines()[-1])
        return {}

def _keepalive_ping():
    try:
        if not APP_BASE_URL: return
        url = f"{APP_BASE_URL.rstrip('/')}/ping"
        headers = {}
        if KEEPALIVE_TOKEN:
            url = f"{url}?t={KEEPALIVE_TOKEN}"
            headers["X-Keepalive-Token"] = KEEPALIVE_TOKEN
        requests.get(url, headers=headers, timeout=5)
    except Exception:
        pass

# -------------------- Exchange helpers --------------------
def fetch_pair_precisions():
    try:
        r = requests.get(f"{BASE_URL}/exchange/v1/markets_details", timeout=12)
        if not r.ok:
            _log_http_issue("markets_details", r); return
        data = r.json()
        hit = 0
        for item in data:
            p = item.get("pair") or item.get("market") or item.get("coindcx_name")
            if p in PAIRS:
                PAIR_RULES[p] = {
                    "price_precision": int(item.get("target_currency_precision",6)),
                    "qty_precision":   int(item.get("base_currency_precision",6)),
                    "min_qty":         float(item.get("min_quantity",0.0) or 0.0),
                    "min_notional":    float(item.get("min_notional",0.0) or 0.0)
                }
                hit += 1
        scan_log.append(f"{ist_now()} | market rules refreshed ({hit} pairs)")
    except Exception as e:
        scan_log.append(f"{ist_now()} | markets_details fail: {e}")

def get_wallet_balances():
    payload = json.dumps({"timestamp": int(time.time()*1000)})
    sig = hmac_signature(payload)
    headers = {"X-AUTH-APIKEY": API_KEY, "X-AUTH-SIGNATURE": sig, "Content-Type":"application/json"}
    balances = {}
    try:
        r = requests.post(f"{BASE_URL}/exchange/v1/users/balances", headers=headers, data=payload, timeout=10)
        if r.ok:
            for b in r.json(): balances[b["currency"]] = float(b["balance"])
        else: _log_http_issue("balances", r)
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

def place_market(pair, side, qty):
    body = {
        "market": pair,
        "side": side.lower(),
        "order_type": "market_order",
        "total_quantity": f"{qty}",
        "timestamp": int(time.time()*1000)
    }
    scan_log.append(f"{ist_now()} | {pair} | PRE-ORDER {side} qty={qty} @ MKT")
    return _signed_post(f"{BASE_URL}/exchange/v1/orders/create", body) or {}

def place_limit(pair, side, qty, price):
    body = {
        "market": pair,
        "side": side.lower(),
        "order_type": "limit_order",
        "price_per_unit": f"{price}",
        "total_quantity": f"{qty}",
        "timestamp": int(time.time()*1000)
    }
    scan_log.append(f"{ist_now()} | {pair} | PRE-ORDER {side} qty={qty} @ {price}")
    return _signed_post(f"{BASE_URL}/exchange/v1/orders/create", body) or {}

def get_order_status(order_id=None, client_order_id=None):
    body = {"timestamp": int(time.time()*1000)}
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
    except: pass
    return str(res.get("id") or res.get("order_id") or res.get("client_order_id") or res.get("orderId") or "") or None

# -------------------- Precision / min qty --------------------
def _rules(pair):           return PAIR_RULES.get(pair, {})
def _pp(pair):              return int(_rules(pair).get("price_precision",6))
def _qp(pair):              return int(_rules(pair).get("qty_precision",6))
def _min_qty(pair):         return float(_rules(pair).get("min_qty",0.0) or 0.0)
def _min_notional(pair):    return float(_rules(pair).get("min_notional",0.0) or 0.0)

def fmt_price(pair, price):
    return float(f"{float(price):.{_pp(pair)}f}")

def fmt_qty(pair, qty):
    q = max(float(qty), _min_qty(pair))
    return float(f"{q:.{_qp(pair)}f}")

def _qty_step(pair): return 10**(-_qp(pair))

def _affordable_buy_qty(pair, price, usdt_free):
    if price <= 0: return 0.0
    eff_unit = price * (1.0 + FEE_PCT_PER_SIDE) * BUY_HEADROOM
    return max(0.0, usdt_free / eff_unit)

# -------------------- Indicators --------------------
def _ema(series, n):
    if len(series) < n: return None
    k = 2/(n+1)
    ema = sum(series[:n])/n
    for v in series[n:]: ema = v*k + ema*(1-k)
    return ema

def _macd(series, fast=MACD_FAST, slow=MACD_SLOW, sig=MACD_SIGNAL):
    if len(series) < slow+sig: return None, None, None
    ema_fast = _ema(series, fast)
    ema_slow = _ema(series, slow)
    if ema_fast is None or ema_slow is None: return None, None, None
    macd_line = ema_fast - ema_slow
    # build macd history to compute signal
    hist = []
    e_fast = sum(series[:fast])/fast
    e_slow = sum(series[:slow])/slow
    kf, ks = 2/(fast+1), 2/(slow+1)
    for v in series[slow:]:
        e_fast = v*kf + e_fast*(1-kf) if len(hist)>0 else ema_fast
        e_slow = v*ks + e_slow*(1-ks) if len(hist)>0 else ema_slow
        hist.append(e_fast - e_slow)
    if len(hist) < sig: return macd_line, None, None
    # signal ema of macd hist
    ksig = 2/(sig+1)
    s = sum(hist[:sig])/sig
    for v in hist[sig:]: s = v*ksig + s*(1-ksig)
    signal = s
    return macd_line, signal, macd_line - signal

# -------------------- Logic --------------------
_last_rules_refresh = 0
_autostart_lock = threading.Lock()

def _maybe_place_tp(pair, ref_avg_px=None):
    """Place a TP limit sell for FULL coin free balance at +TP_PCT.
       If a TP already open for this pair, skip."""
    if open_tp.get(pair): return
    balances = get_wallet_balances()
    free_coin = balances.get(pair[:-4], 0.0)
    if free_coin < _min_qty(pair): return
    # choose a reference: recent avg buy px if provided, else last close
    last_close = candle_logs.get(pair, [{}])[-1].get("close", 0.0)
    base_px = ref_avg_px or last_close
    if base_px <= 0: return
    tp_px  = fmt_price(pair, base_px * (1.0 + TP_PCT))
    qty    = fmt_qty(pair, free_coin)
    if qty < _min_qty(pair): return

    res = place_limit(pair, "SELL", qty, tp_px)
    oid = _extract_order_id(res)
    open_tp[pair] = {"id": oid, "px": tp_px, "qty": qty, "ts": int(time.time())}
    scan_log.append(f"{ist_now()} | {pair} | TP SELL placed qty={qty} @ {tp_px} | id={oid} | res={res}")

def _check_tp_fills(pair):
    """Poll TP order and record PnL when filled."""
    tp = open_tp.get(pair)
    if not tp or not tp.get("id"): return
    st = get_order_status(order_id=tp["id"])
    # consider filled when remaining is 0 or status contains 'filled'
    rem = float(st.get("remaining_quantity", st.get("remaining_qty", st.get("leaves_qty", 0))) or 0)
    status_txt = (st.get("status") or "").lower()
    filled = (rem == 0) or ("filled" in status_txt and "part" not in status_txt)
    if filled:
        # record fill
        total  = float(st.get("total_quantity", st.get("quantity", st.get("orig_qty", tp["qty"]))) or tp["qty"])
        avg_px = float(st.get("avg_price", st.get("average_price", st.get("price", tp["px"]))) or tp["px"])
        oid = tp["id"]
        apply_fill_update(pair, "SELL", avg_px, total, int(time.time()*1000), oid)
        scan_log.append(f"{ist_now()} | {pair} | TP FILLED qty={total} @ {avg_px} | st={st}")
        open_tp[pair] = None

def aggregate_candles(pair, interval=CANDLE_INTERVAL):
    ticks = tick_logs[pair]
    if not ticks: return
    candles, candle, last_window = [], None, None
    for ts, price in sorted(ticks, key=lambda x: x[0]):
        wstart = ts - (ts % interval)
        if last_window != wstart:
            if candle: candles.append(candle)
            candle = {"open": price,"high": price,"low": price,"close": price,"volume":1,"start": wstart}
            last_window = wstart
        else:
            candle["high"] = max(candle["high"], price)
            candle["low"]  = min(candle["low"], price)
            candle["close"]= price
            candle["volume"] += 1
    if candle: candles.append(candle)
    candle_logs[pair] = candles[-600:]  # ~50 minutes of 5s bars

def _calc_ema_state(pair):
    cs = candle_logs.get(pair) or []
    if len(cs) < max(EMA_SLOW, 3): return None, None, None
    closes = [c["close"] for c in cs]
    # prev and current
    prev_closes = closes[:-1]
    now_closes  = closes
    ema_fast_prev = _ema(prev_closes, EMA_FAST)
    ema_slow_prev = _ema(prev_closes, EMA_SLOW)
    ema_fast_now  = _ema(now_closes,  EMA_FAST)
    ema_slow_now  = _ema(now_closes,  EMA_SLOW)
    cross_up   = (ema_fast_prev is not None and ema_slow_prev is not None and
                  ema_fast_now  is not None and ema_slow_now  is not None and
                  ema_fast_prev <= ema_slow_prev and ema_fast_now > ema_slow_now)
    cross_down = (ema_fast_prev is not None and ema_slow_prev is not None and
                  ema_fast_now  is not None and ema_slow_now  is not None and
                  ema_fast_prev >= ema_slow_prev and ema_fast_now < ema_slow_now)
    state = "bull" if ema_fast_now is not None and ema_slow_now is not None and ema_fast_now >= ema_slow_now else "bear"
    return state, cross_up, cross_down

def _calc_macd_state(pair):
    cs = candle_logs.get(pair) or []
    closes = [c["close"] for c in cs]
    macd_line, signal, hist = _macd(closes)
    if macd_line is None or signal is None: return "none", None
    return ("bullish" if macd_line > signal else "bearish"), (macd_line, signal, hist)

def normalize_buy_qty(pair, price, usdt_free):
    # spend BUY_USDT_FRAC of free USDT, but obey min qty / notional and precision and fees/headroom
    spend = max(0.0, usdt_free * BUY_USDT_FRAC)
    max_units = _affordable_buy_qty(pair, price, spend)
    q = fmt_qty(pair, max_units)
    mq = _min_qty(pair)
    if q < mq: q = fmt_qty(pair, mq)
    if q < mq: return 0.0
    mn = _min_notional(pair)
    if mn > 0 and price*q < mn:
        need_q = mn/price
        q = fmt_qty(pair, max(q, need_q))
        if price*q*(1.0+FEE_PCT_PER_SIDE)*BUY_HEADROOM > spend + 1e-9:
            # cannot afford min notional
            return 0.0
    return q

def scan_once():
    global status_epoch, error_message
    try:
        # keepalive
        global last_keepalive
        now_real = time.time()
        if now_real - last_keepalive >= KEEPALIVE_SEC:
            _keepalive_ping()
            last_keepalive = now_real

        # rules refresh
        global _last_rules_refresh
        if (time.time() - _last_rules_refresh) >= RULES_REFRESH_SEC:
            fetch_pair_precisions()
            _last_rules_refresh = time.time()

        prices = fetch_all_prices()
        now_ts = int(time.time())

        # ticks -> candles
        for pair in PAIRS:
            if pair in prices:
                px = prices[pair]["price"]
                tick_logs[pair].append((now_ts, px))
                if len(tick_logs[pair]) > 6000: tick_logs[pair] = tick_logs[pair][-6000:]
                aggregate_candles(pair, CANDLE_INTERVAL)

        balances = get_wallet_balances()
        usdt_free = float(balances.get("USDT", 0.0))

        # per pair logic
        for pair in PAIRS:
            _check_tp_fills(pair)  # poll TP status first

            cs = candle_logs.get(pair) or []
            if len(cs) < 30:  # need a bit of history
                continue

            last = cs[-1]["close"]
            ema_state, cross_up, cross_down = _calc_ema_state(pair)
            macd_state, macd_vals = _calc_macd_state(pair)

            # Detailed log line (signals + gates; smc=open placeholder)
            scan_log.append(f"{ist_now()} | {pair} | ema={ema_state} cross_up={bool(cross_up)} macd={macd_state} smc=open")

            # BUY gate
            buy_signal = bool(cross_up) and (macd_state == "bullish")

            if buy_signal:
                q = normalize_buy_qty(pair, last, usdt_free)
                if q >= _min_qty(pair):
                    res = place_market(pair, "BUY", q)
                    oid = _extract_order_id(res)
                    if oid:
                        st = get_order_status(order_id=oid)
                        # record BUY fill for FIFO & place TP from avg fill
                        total  = float(st.get("total_quantity", st.get("quantity", st.get("orig_qty", q))) or q)
                        avg_px = float(st.get("avg_price", st.get("average_price", st.get("price", last))) or last)
                        apply_fill_update(pair, "BUY", avg_px, total, int(time.time()*1000), oid)
                        scan_log.append(f"{ist_now()} | {pair} | BUY filled qty={total} @ {avg_px} | id={oid}")
                        # TP for FULL free coin wallet (exit only on profit hit)
                        _maybe_place_tp(pair, ref_avg_px=avg_px)
                        # refresh balances snapshot
                        balances = get_wallet_balances()
                        usdt_free = float(balances.get("USDT", 0.0))
                    else:
                        scan_log.append(f"{ist_now()} | {pair} | BUY rejected | res={res}")
                else:
                    scan_log.append(f"{ist_now()} | {pair} | BUY skipped — qty<{_min_qty(pair)} or notional fail")

            # If we already hold coin (from older runs) and have no TP, set one at +0.5% from last close
            bal_coin = balances.get(pair[:-4], 0.0)
            if open_tp.get(pair) is None and bal_coin >= _min_qty(pair):
                _maybe_place_tp(pair, ref_avg_px=None)

        status["msg"], status["last"] = "Running", ist_now()
        status_epoch = int(time.time())

    except Exception as e:
        error_message = f"{e.__class__.__name__}: {e}"
        scan_log.append(f"{ist_now()} | ERROR | {error_message}")
        scan_log.append(traceback.format_exc().splitlines()[-1])

def scan_loop():
    scan_log.clear()
    while running:
        scan_once()
        time.sleep(POLL_SEC)

# -------------------- Routes --------------------
@app.route("/")
def index():
    return render_template("index.html")

@app.route("/start", methods=["POST"])
def start():
    _start_loop_once()
    return jsonify({"status":"started"})

@app.route("/stop", methods=["POST"])
def stop():
    global running
    running = False
    return jsonify({"status":"stopped"})

@app.route("/status")
def get_status():
    balances = get_wallet_balances()
    usdt_total = balances.get("USDT", 0.0)
    coins = {p[:-4]: balances.get(p[:-4], 0.0) for p in PAIRS}
    profit_today     = compute_realized_pnl_today()
    profit_yesterday = round(profit_state["daily"].get(ist_yesterday(), 0.0), 6)
    cumulative_pnl   = round(profit_state.get("cumulative_pnl", 0.0), 6)

    # visible TP orders
    tp_view = {}
    for p, o in open_tp.items():
        if o:
            tp_view[p] = {"id": o.get("id"), "px": o.get("px"), "qty": o.get("qty"), "ts": o.get("ts")}

    # keepalive meta for UI
    now_real = time.time()
    last_age = now_real - (last_keepalive or 0)
    keepalive_info = {
        "enabled": bool(KEEPALIVE_TOKEN),
        "interval_sec": KEEPALIVE_SEC,
        "last_ping_epoch": int(last_keepalive or 0),
        "last_ping_age_sec": (max(0, int(last_age)) if last_keepalive else None),
        "next_due_sec": (max(0, int(KEEPALIVE_SEC - last_age)) if last_keepalive else None),
        "app_base_url": APP_BASE_URL,
        "poll_sec": POLL_SEC,
    }

    return jsonify({
        "status": status["msg"],
        "last": status["last"],
        "status_epoch": status_epoch,
        "usdt": usdt_total,
        "coins": coins,
        "profit_today": profit_today,
        "profit_yesterday": profit_yesterday,
        "pnl_cumulative": cumulative_pnl,
        "tp_orders": tp_view,
        "trades": trade_log[-10:][::-1],
        "scans": scan_log[-120:][::-1],
        "error": error_message,
        "keepalive": keepalive_info
    })

@app.route("/ping", methods=["GET","HEAD"])
def ping():
    token = os.environ.get("KEEPALIVE_TOKEN","")
    provided = (request.args.get("t") or request.headers.get("X-Keepalive-Token") or "")
    if token and provided != token:
        print(f"[{ist_now()}] /ping forbidden (bad token) method={request.method}")
        return "forbidden", 403
    print(f"[{ist_now()}] /ping ok method={request.method}")
    if request.method == "HEAD":
        return ("", 200)
    return ("pong", 200)

# -------------------- Boot / Autostart --------------------
def _start_loop_once():
    global running
    with _autostart_lock:
        if not running:
            running = True
            t = threading.Thread(target=scan_loop, daemon=True)
            t.start()

def _boot_kick():
    try:
        load_profit_state()
        fetch_pair_precisions()
        time.sleep(1.0)
        _start_loop_once()
    except Exception as e:
        print("boot kick failed:", e)

if os.environ.get("AUTOSTART","1") == "1":
    threading.Thread(target=_boot_kick, daemon=True).start()

if __name__ == "__main__":
    load_profit_state()
    fetch_pair_precisions()
    _start_loop_once()
    port = int(os.environ.get("PORT","10000"))
    app.run(host="0.0.0.0", port=port)
import time
import threading
import hmac
import hashlib
import requests
import json
import traceback
from flask import Flask, render_template, jsonify, request
from datetime import datetime, timedelta
from pytz import timezone
from collections import deque

# -------------------- Flask --------------------
app = Flask(__name__)

APP_BASE_URL   = os.environ.get("APP_BASE_URL", "")
KEEPALIVE_TOKEN= os.environ.get("KEEPALIVE_TOKEN", "")

# -------------------- API / Markets --------------------
API_KEY        = os.environ.get("API_KEY", "")
API_SECRET_RAW = os.environ.get("API_SECRET", "")
API_SECRET     = API_SECRET_RAW.encode() if isinstance(API_SECRET_RAW, str) else API_SECRET_RAW
BASE_URL       = "https://api.coindcx.com"

# Pairs (LTC removed)
PAIRS = ["BTCUSDT","ETHUSDT","XRPUSDT","SHIBUSDT","SOLUSDT","DOGEUSDT","ADAUSDT","AEROUSDT","BNBUSDT"]

# Fallback rules; will be refreshed from exchange at boot and periodically
PAIR_RULES = {
    "BTCUSDT":  {"price_precision": 1, "qty_precision": 4, "min_qty": 0.001,  "min_notional": 0.0},
    "ETHUSDT":  {"price_precision": 2, "qty_precision": 6, "min_qty": 0.0001, "min_notional": 0.0},
    "XRPUSDT":  {"price_precision": 4, "qty_precision": 4, "min_qty": 0.1,    "min_notional": 0.0},
    "SHIBUSDT": {"price_precision": 8, "qty_precision": 4, "min_qty": 10000,  "min_notional": 0.0},
    "DOGEUSDT": {"price_precision": 5, "qty_precision": 4, "min_qty": 0.001,  "min_notional": 0.0},
    "SOLUSDT":  {"price_precision": 2, "qty_precision": 4, "min_qty": 0.01,   "min_notional": 0.0},
    "AEROUSDT": {"price_precision": 3, "qty_precision": 2, "min_qty": 0.01,   "min_notional": 0.0},
    "ADAUSDT":  {"price_precision": 4, "qty_precision": 2, "min_qty": 0.1,    "min_notional": 0.0},
    "BNBUSDT":  {"price_precision": 3, "qty_precision": 4, "min_qty": 0.001,  "min_notional": 0.0},
}

# -------------------- Core settings --------------------
CANDLE_INTERVAL = 5           # seconds
POLL_SEC        = 1.0
KEEPALIVE_SEC   = 240

# fees & rounding
FEE_PCT_PER_SIDE= 0.0010      # 0.10% per side—adjust if your account differs
BUY_HEADROOM    = 1.0005      # tiny buffer so rounding+fees still fit

# signal windows
EMA_FAST        = 5
EMA_SLOW        = 20
MACD_FAST       = 12
MACD_SLOW       = 26
MACD_SIGNAL     = 9

# position sizing
BUY_USDT_FRAC   = 0.30        # buy 30% of free USDT when signal fires

# exits
TP_PCT          = 0.02    # +0.5% target
NO_SL           = True        # explicit: no stop loss

# rules refresh cadence
RULES_REFRESH_SEC = 1800

# -------------------- Time helpers --------------------
IST = timezone('Asia/Kolkata')
def ist_now():        return datetime.now(IST).strftime('%Y-%m-%d %H:%M:%S')
def ist_date():       return datetime.now(IST).strftime('%Y-%m-%d')
def ist_yesterday():  return (datetime.now(IST) - timedelta(days=1)).strftime('%Y-%m-%d')

# -------------------- State --------------------
tick_logs   = {p: [] for p in PAIRS}    # (ts, px)
candle_logs = {p: [] for p in PAIRS}    # list of dicts per pair
scan_log    = []                        # text lines for UI
trade_log   = []                        # optional manual additions
running     = False
status      = {"msg":"Idle","last":""}
status_epoch= 0
error_message = ""

# TP orders we opened: {pair: {"id":..., "px":..., "qty":..., "ts":...}}
open_tp = {p: None for p in PAIRS}

# profit state (FIFO)
PROFIT_STATE_FILE = "profit_state.json"
profit_state = {"cumulative_pnl":0.0,"daily":{},"inventory":{},"processed_orders":[]}

# keepalive tracker
last_keepalive = 0

# -------------------- PnL helpers (FIFO) --------------------
from collections import deque

def load_profit_state():
    global profit_state
    try:
        with open(PROFIT_STATE_FILE,"r") as f:
            data = json.load(f)
        profit_state["cumulative_pnl"] = float(data.get("cumulative_pnl",0.0))
        profit_state["daily"]          = dict(data.get("daily",{}))
        profit_state["inventory"]      = data.get("inventory",{})
        profit_state["processed_orders"]= list(data.get("processed_orders",[]))
    except:
        pass

def save_profit_state():
    try:
        with open(PROFIT_STATE_FILE,"w") as f:
            json.dump({
                "cumulative_pnl": round(profit_state.get("cumulative_pnl",0.0),6),
                "daily": {k:round(v,6) for k,v in profit_state.get("daily",{}).items()},
                "inventory": profit_state.get("inventory",{}),
                "processed_orders": profit_state.get("processed_orders",[])
            }, f)
    except:
        pass

def _get_inventory_deque(market):
    dq = deque()
    for lot in profit_state["inventory"].get(market, []):
        try:
            q,c = float(lot[0]), float(lot[1])
            if q>0 and c>0: dq.append([q,c])
        except:
            continue
    return dq

def _set_inventory_from_deque(market, dq):
    profit_state["inventory"][market] = [[float(q),float(c)] for q,c in dq]

def apply_fill_update(market, side, price, qty, ts_ms, order_id):
    # de-dup
    if (not order_id) or order_id in profit_state["processed_orders"]:
        return
    try:
        price = float(price); qty = float(qty)
        if price<=0 or qty<=0: return
    except: return

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
            if lot_q <= 1e-18: inv.popleft()
            else: inv[0][0] = lot_q

    _set_inventory_from_deque(market, inv)
    profit_state["processed_orders"].append(order_id)
    profit_state["cumulative_pnl"] = float(profit_state.get("cumulative_pnl",0.0) + realized)
    dkey = ist_date()
    profit_state["daily"][dkey] = float(profit_state["daily"].get(dkey,0.0) + realized)
    save_profit_state()

def compute_realized_pnl_today():
    return round(profit_state["daily"].get(ist_date(),0.0),6)

# -------------------- HTTP helpers --------------------
def hmac_signature(payload):
    return hmac.new(API_SECRET, payload.encode(), hashlib.sha256).hexdigest()

def _log_http_issue(prefix, r):
    try:
        body = r.text[:240] if hasattr(r,"text") else ""
        scan_log.append(f"{ist_now()} | {prefix} HTTP {r.status_code} | {body}")
    except Exception as e:
        scan_log.append(f"{ist_now()} | {prefix} log-fail: {e}")

def _signed_post(url, body):
    payload = json.dumps(body, separators=(',',':'))
    sig = hmac_signature(payload)
    headers = {"X-AUTH-APIKEY": API_KEY, "X-AUTH-SIGNATURE": sig, "Content-Type":"application/json"}
    try:
        r = requests.post(url, headers=headers, data=payload, timeout=12)
        if not r.ok: _log_http_issue(f"POST {url}", r)
        if r.headers.get("content-type","").startswith("application/json"):
            return r.json()
        return {}
    except Exception as e:
        scan_log.append(f"{ist_now()} | POST fail {url} | {e.__class__.__name__}: {e}")
        scan_log.append(traceback.format_exc().splitlines()[-1])
        return {}

def _keepalive_ping():
    try:
        if not APP_BASE_URL: return
        url = f"{APP_BASE_URL.rstrip('/')}/ping"
        headers = {}
        if KEEPALIVE_TOKEN:
            url = f"{url}?t={KEEPALIVE_TOKEN}"
            headers["X-Keepalive-Token"] = KEEPALIVE_TOKEN
        requests.get(url, headers=headers, timeout=5)
    except Exception:
        pass

# -------------------- Exchange helpers --------------------
def fetch_pair_precisions():
    try:
        r = requests.get(f"{BASE_URL}/exchange/v1/markets_details", timeout=12)
        if not r.ok:
            _log_http_issue("markets_details", r); return
        data = r.json()
        hit = 0
        for item in data:
            p = item.get("pair") or item.get("market") or item.get("coindcx_name")
            if p in PAIRS:
                PAIR_RULES[p] = {
                    "price_precision": int(item.get("target_currency_precision",6)),
                    "qty_precision":   int(item.get("base_currency_precision",6)),
                    "min_qty":         float(item.get("min_quantity",0.0) or 0.0),
                    "min_notional":    float(item.get("min_notional",0.0) or 0.0)
                }
                hit += 1
        scan_log.append(f"{ist_now()} | market rules refreshed ({hit} pairs)")
    except Exception as e:
        scan_log.append(f"{ist_now()} | markets_details fail: {e}")

def get_wallet_balances():
    payload = json.dumps({"timestamp": int(time.time()*1000)})
    sig = hmac_signature(payload)
    headers = {"X-AUTH-APIKEY": API_KEY, "X-AUTH-SIGNATURE": sig, "Content-Type":"application/json"}
    balances = {}
    try:
        r = requests.post(f"{BASE_URL}/exchange/v1/users/balances", headers=headers, data=payload, timeout=10)
        if r.ok:
            for b in r.json(): balances[b["currency"]] = float(b["balance"])
        else: _log_http_issue("balances", r)
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

def place_market(pair, side, qty):
    body = {
        "market": pair,
        "side": side.lower(),
        "order_type": "market_order",
        "total_quantity": f"{qty}",
        "timestamp": int(time.time()*1000)
    }
    scan_log.append(f"{ist_now()} | {pair} | PRE-ORDER {side} qty={qty} @ MKT")
    return _signed_post(f"{BASE_URL}/exchange/v1/orders/create", body) or {}

def place_limit(pair, side, qty, price):
    body = {
        "market": pair,
        "side": side.lower(),
        "order_type": "limit_order",
        "price_per_unit": f"{price}",
        "total_quantity": f"{qty}",
        "timestamp": int(time.time()*1000)
    }
    scan_log.append(f"{ist_now()} | {pair} | PRE-ORDER {side} qty={qty} @ {price}")
    return _signed_post(f"{BASE_URL}/exchange/v1/orders/create", body) or {}

def get_order_status(order_id=None, client_order_id=None):
    body = {"timestamp": int(time.time()*1000)}
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
    except: pass
    return str(res.get("id") or res.get("order_id") or res.get("client_order_id") or res.get("orderId") or "") or None

# -------------------- Precision / min qty --------------------
def _rules(pair):           return PAIR_RULES.get(pair, {})
def _pp(pair):              return int(_rules(pair).get("price_precision",6))
def _qp(pair):              return int(_rules(pair).get("qty_precision",6))
def _min_qty(pair):         return float(_rules(pair).get("min_qty",0.0) or 0.0)
def _min_notional(pair):    return float(_rules(pair).get("min_notional",0.0) or 0.0)

def fmt_price(pair, price):
    return float(f"{float(price):.{_pp(pair)}f}")

def fmt_qty(pair, qty):
    q = max(float(qty), _min_qty(pair))
    return float(f"{q:.{_qp(pair)}f}")

def _qty_step(pair): return 10**(-_qp(pair))

def _affordable_buy_qty(pair, price, usdt_free):
    if price <= 0: return 0.0
    eff_unit = price * (1.0 + FEE_PCT_PER_SIDE) * BUY_HEADROOM
    return max(0.0, usdt_free / eff_unit)

# -------------------- Indicators --------------------
def _ema(series, n):
    if len(series) < n: return None
    k = 2/(n+1)
    ema = sum(series[:n])/n
    for v in series[n:]: ema = v*k + ema*(1-k)
    return ema

def _macd(series, fast=MACD_FAST, slow=MACD_SLOW, sig=MACD_SIGNAL):
    if len(series) < slow+sig: return None, None, None
    ema_fast = _ema(series, fast)
    ema_slow = _ema(series, slow)
    if ema_fast is None or ema_slow is None: return None, None, None
    macd_line = ema_fast - ema_slow
    # build macd history to compute signal
    hist = []
    e_fast = sum(series[:fast])/fast
    e_slow = sum(series[:slow])/slow
    kf, ks = 2/(fast+1), 2/(slow+1)
    for v in series[slow:]:
        e_fast = v*kf + e_fast*(1-kf) if len(hist)>0 else ema_fast
        e_slow = v*ks + e_slow*(1-ks) if len(hist)>0 else ema_slow
        hist.append(e_fast - e_slow)
    if len(hist) < sig: return macd_line, None, None
    # signal ema of macd hist
    ksig = 2/(sig+1)
    s = sum(hist[:sig])/sig
    for v in hist[sig:]: s = v*ksig + s*(1-ksig)
    signal = s
    return macd_line, signal, macd_line - signal

# -------------------- Logic --------------------
_last_rules_refresh = 0
_autostart_lock = threading.Lock()

def _maybe_place_tp(pair, ref_avg_px=None):
    """Place a TP limit sell for FULL coin free balance at +TP_PCT.
       If a TP already open for this pair, skip."""
    if open_tp.get(pair): return
    balances = get_wallet_balances()
    free_coin = balances.get(pair[:-4], 0.0)
    if free_coin < _min_qty(pair): return
    # choose a reference: recent avg buy px if provided, else last close
    last_close = candle_logs.get(pair, [{}])[-1].get("close", 0.0)
    base_px = ref_avg_px or last_close
    if base_px <= 0: return
    tp_px  = fmt_price(pair, base_px * (1.0 + TP_PCT))
    qty    = fmt_qty(pair, free_coin)
    if qty < _min_qty(pair): return

    res = place_limit(pair, "SELL", qty, tp_px)
    oid = _extract_order_id(res)
    open_tp[pair] = {"id": oid, "px": tp_px, "qty": qty, "ts": int(time.time())}
    scan_log.append(f"{ist_now()} | {pair} | TP SELL placed qty={qty} @ {tp_px} | id={oid} | res={res}")

def _check_tp_fills(pair):
    """Poll TP order and record PnL when filled."""
    tp = open_tp.get(pair)
    if not tp or not tp.get("id"): return
    st = get_order_status(order_id=tp["id"])
    # consider filled when remaining is 0 or status contains 'filled'
    rem = float(st.get("remaining_quantity", st.get("remaining_qty", st.get("leaves_qty", 0))) or 0)
    status_txt = (st.get("status") or "").lower()
    filled = (rem == 0) or ("filled" in status_txt and "part" not in status_txt)
    if filled:
        # record fill
        total  = float(st.get("total_quantity", st.get("quantity", st.get("orig_qty", tp["qty"]))) or tp["qty"])
        avg_px = float(st.get("avg_price", st.get("average_price", st.get("price", tp["px"]))) or tp["px"])
        oid = tp["id"]
        apply_fill_update(pair, "SELL", avg_px, total, int(time.time()*1000), oid)
        scan_log.append(f"{ist_now()} | {pair} | TP FILLED qty={total} @ {avg_px} | st={st}")
        open_tp[pair] = None

def aggregate_candles(pair, interval=CANDLE_INTERVAL):
    ticks = tick_logs[pair]
    if not ticks: return
    candles, candle, last_window = [], None, None
    for ts, price in sorted(ticks, key=lambda x: x[0]):
        wstart = ts - (ts % interval)
        if last_window != wstart:
            if candle: candles.append(candle)
            candle = {"open": price,"high": price,"low": price,"close": price,"volume":1,"start": wstart}
            last_window = wstart
        else:
            candle["high"] = max(candle["high"], price)
            candle["low"]  = min(candle["low"], price)
            candle["close"]= price
            candle["volume"] += 1
    if candle: candles.append(candle)
    candle_logs[pair] = candles[-600:]  # ~50 minutes of 5s bars

def _calc_ema_state(pair):
    cs = candle_logs.get(pair) or []
    if len(cs) < max(EMA_SLOW, 3): return None, None, None
    closes = [c["close"] for c in cs]
    # prev and current
    prev_closes = closes[:-1]
    now_closes  = closes
    ema_fast_prev = _ema(prev_closes, EMA_FAST)
    ema_slow_prev = _ema(prev_closes, EMA_SLOW)
    ema_fast_now  = _ema(now_closes,  EMA_FAST)
    ema_slow_now  = _ema(now_closes,  EMA_SLOW)
    cross_up   = (ema_fast_prev is not None and ema_slow_prev is not None and
                  ema_fast_now  is not None and ema_slow_now  is not None and
                  ema_fast_prev <= ema_slow_prev and ema_fast_now > ema_slow_now)
    cross_down = (ema_fast_prev is not None and ema_slow_prev is not None and
                  ema_fast_now  is not None and ema_slow_now  is not None and
                  ema_fast_prev >= ema_slow_prev and ema_fast_now < ema_slow_now)
    state = "bull" if ema_fast_now is not None and ema_slow_now is not None and ema_fast_now >= ema_slow_now else "bear"
    return state, cross_up, cross_down

def _calc_macd_state(pair):
    cs = candle_logs.get(pair) or []
    closes = [c["close"] for c in cs]
    macd_line, signal, hist = _macd(closes)
    if macd_line is None or signal is None: return "none", None
    return ("bullish" if macd_line > signal else "bearish"), (macd_line, signal, hist)

def normalize_buy_qty(pair, price, usdt_free):
    # spend BUY_USDT_FRAC of free USDT, but obey min qty / notional and precision and fees/headroom
    spend = max(0.0, usdt_free * BUY_USDT_FRAC)
    max_units = _affordable_buy_qty(pair, price, spend)
    q = fmt_qty(pair, max_units)
    mq = _min_qty(pair)
    if q < mq: q = fmt_qty(pair, mq)
    if q < mq: return 0.0
    mn = _min_notional(pair)
    if mn > 0 and price*q < mn:
        need_q = mn/price
        q = fmt_qty(pair, max(q, need_q))
        if price*q*(1.0+FEE_PCT_PER_SIDE)*BUY_HEADROOM > spend + 1e-9:
            # cannot afford min notional
            return 0.0
    return q

def scan_once():
    global status_epoch, error_message
    try:
        # keepalive
        global last_keepalive
        now_real = time.time()
        if now_real - last_keepalive >= KEEPALIVE_SEC:
            _keepalive_ping()
            last_keepalive = now_real

        # rules refresh
        global _last_rules_refresh
        if (time.time() - _last_rules_refresh) >= RULES_REFRESH_SEC:
            fetch_pair_precisions()
            _last_rules_refresh = time.time()

        prices = fetch_all_prices()
        now_ts = int(time.time())

        # ticks -> candles
        for pair in PAIRS:
            if pair in prices:
                px = prices[pair]["price"]
                tick_logs[pair].append((now_ts, px))
                if len(tick_logs[pair]) > 6000: tick_logs[pair] = tick_logs[pair][-6000:]
                aggregate_candles(pair, CANDLE_INTERVAL)

        balances = get_wallet_balances()
        usdt_free = float(balances.get("USDT", 0.0))

        # per pair logic
        for pair in PAIRS:
            _check_tp_fills(pair)  # poll TP status first

            cs = candle_logs.get(pair) or []
            if len(cs) < 30:  # need a bit of history
                continue

            last = cs[-1]["close"]
            ema_state, cross_up, cross_down = _calc_ema_state(pair)
            macd_state, macd_vals = _calc_macd_state(pair)

            # Detailed log line (signals + gates; smc=open placeholder)
            scan_log.append(f"{ist_now()} | {pair} | ema={ema_state} cross_up={bool(cross_up)} macd={macd_state} smc=open")

            # BUY gate
            buy_signal = bool(cross_up) and (macd_state == "bullish")

            if buy_signal:
                q = normalize_buy_qty(pair, last, usdt_free)
                if q >= _min_qty(pair):
                    res = place_market(pair, "BUY", q)
                    oid = _extract_order_id(res)
                    if oid:
                        st = get_order_status(order_id=oid)
                        # record BUY fill for FIFO & place TP from avg fill
                        total  = float(st.get("total_quantity", st.get("quantity", st.get("orig_qty", q))) or q)
                        avg_px = float(st.get("avg_price", st.get("average_price", st.get("price", last))) or last)
                        apply_fill_update(pair, "BUY", avg_px, total, int(time.time()*1000), oid)
                        scan_log.append(f"{ist_now()} | {pair} | BUY filled qty={total} @ {avg_px} | id={oid}")
                        # TP for FULL free coin wallet (exit only on profit hit)
                        _maybe_place_tp(pair, ref_avg_px=avg_px)
                        # refresh balances snapshot
                        balances = get_wallet_balances()
                        usdt_free = float(balances.get("USDT", 0.0))
                    else:
                        scan_log.append(f"{ist_now()} | {pair} | BUY rejected | res={res}")
                else:
                    scan_log.append(f"{ist_now()} | {pair} | BUY skipped — qty<{_min_qty(pair)} or notional fail")

            # If we already hold coin (from older runs) and have no TP, set one at +0.5% from last close
            bal_coin = balances.get(pair[:-4], 0.0)
            if open_tp.get(pair) is None and bal_coin >= _min_qty(pair):
                _maybe_place_tp(pair, ref_avg_px=None)

        status["msg"], status["last"] = "Running", ist_now()
        status_epoch = int(time.time())

    except Exception as e:
        error_message = f"{e.__class__.__name__}: {e}"
        scan_log.append(f"{ist_now()} | ERROR | {error_message}")
        scan_log.append(traceback.format_exc().splitlines()[-1])

def scan_loop():
    scan_log.clear()
    while running:
        scan_once()
        time.sleep(POLL_SEC)

# -------------------- Routes --------------------
@app.route("/")
def index():
    return render_template("index.html")

@app.route("/start", methods=["POST"])
def start():
    _start_loop_once()
    return jsonify({"status":"started"})

@app.route("/stop", methods=["POST"])
def stop():
    global running
    running = False
    return jsonify({"status":"stopped"})

@app.route("/status")
def get_status():
    balances = get_wallet_balances()
    usdt_total = balances.get("USDT", 0.0)
    coins = {p[:-4]: balances.get(p[:-4], 0.0) for p in PAIRS}
    profit_today     = compute_realized_pnl_today()
    profit_yesterday = round(profit_state["daily"].get(ist_yesterday(), 0.0), 6)
    cumulative_pnl   = round(profit_state.get("cumulative_pnl", 0.0), 6)

    # visible TP orders
    tp_view = {}
    for p, o in open_tp.items():
        if o:
            tp_view[p] = {"id": o.get("id"), "px": o.get("px"), "qty": o.get("qty"), "ts": o.get("ts")}

    # keepalive meta for UI
    now_real = time.time()
    last_age = now_real - (last_keepalive or 0)
    keepalive_info = {
        "enabled": bool(KEEPALIVE_TOKEN),
        "interval_sec": KEEPALIVE_SEC,
        "last_ping_epoch": int(last_keepalive or 0),
        "last_ping_age_sec": (max(0, int(last_age)) if last_keepalive else None),
        "next_due_sec": (max(0, int(KEEPALIVE_SEC - last_age)) if last_keepalive else None),
        "app_base_url": APP_BASE_URL,
        "poll_sec": POLL_SEC,
    }

    return jsonify({
        "status": status["msg"],
        "last": status["last"],
        "status_epoch": status_epoch,
        "usdt": usdt_total,
        "coins": coins,
        "profit_today": profit_today,
        "profit_yesterday": profit_yesterday,
        "pnl_cumulative": cumulative_pnl,
        "tp_orders": tp_view,
        "trades": trade_log[-10:][::-1],
        "scans": scan_log[-120:][::-1],
        "error": error_message,
        "keepalive": keepalive_info
    })

@app.route("/ping", methods=["GET","HEAD"])
def ping():
    token = os.environ.get("KEEPALIVE_TOKEN","")
    provided = (request.args.get("t") or request.headers.get("X-Keepalive-Token") or "")
    if token and provided != token:
        print(f"[{ist_now()}] /ping forbidden (bad token) method={request.method}")
        return "forbidden", 403
    print(f"[{ist_now()}] /ping ok method={request.method}")
    if request.method == "HEAD":
        return ("", 200)
    return ("pong", 200)

# -------------------- Boot / Autostart --------------------
def _start_loop_once():
    global running
    with _autostart_lock:
        if not running:
            running = True
            t = threading.Thread(target=scan_loop, daemon=True)
            t.start()

def _boot_kick():
    try:
        load_profit_state()
        fetch_pair_precisions()
        time.sleep(1.0)
        _start_loop_once()
    except Exception as e:
        print("boot kick failed:", e)

if os.environ.get("AUTOSTART","1") == "1":
    threading.Thread(target=_boot_kick, daemon=True).start()

if __name__ == "__main__":
    load_profit_state()
    fetch_pair_precisions()
    _start_loop_once()
    port = int(os.environ.get("PORT","10000"))
    app.run(host="0.0.0.0", port=port)
