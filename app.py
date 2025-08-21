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
KEEPALIVE_TOKEN = os.environ.get("KEEPALIVE_TOKEN", "")

# -------------------- API / Markets --------------------
API_KEY = os.environ.get("API_KEY", "")
API_SECRET_RAW = os.environ.get("API_SECRET", "")
API_SECRET = API_SECRET_RAW.encode() if isinstance(API_SECRET_RAW, str) else API_SECRET_RAW
BASE_URL = "https://api.coindcx.com"

# Pairs (LTC removed)
PAIRS = [
    "BTCUSDT", "ETHUSDT", "XRPUSDT", "SHIBUSDT",
    "SOLUSDT", "DOGEUSDT", "ADAUSDT", "AEROUSDT", "BNBUSDT"
]

# Local fallbacks; live rules fetched later
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

# -------------------- Strategy knobs (Simple EMA+MACD) --------------------
EMA_FAST, EMA_SLOW = 5, 20          # EMA crossover
MACD_FAST, MACD_SLOW, MACD_SIG = 12, 26, 9

TP_PCT = 0.005                      # 0.5% take profit
USE_STOPLOSS = False                # No SL (as requested)

BUY_USDT_FRACTION = 0.30            # invest 30% of free USDT
SELL_COIN_FRACTION = 1.00           # sell full wallet balance on sell signal

CANDLE_INTERVAL = 5                 # seconds per bar
POLL_SEC = 1.0

# Fees/formatting helpers
FEE_PCT_PER_SIDE = 0.0010           # 0.10% side
BUY_HEADROOM = 1.0005               # make sure rounding still fits
KEEPALIVE_SEC = 240

# -------------------- Time helpers --------------------
IST = timezone('Asia/Kolkata')
def ist_now(): return datetime.now(IST).strftime('%Y-%m-%d %H:%M:%S')
def ist_date(): return datetime.now(IST).strftime('%Y-%m-%d')
def ist_yesterday(): return (datetime.now(IST) - timedelta(days=1)).strftime('%Y-%m-%d')

# -------------------- State --------------------
tick_logs = {p: [] for p in PAIRS}         # (ts, price)
candle_logs = {p: [] for p in PAIRS}       # list of OHLCV dicts
scan_log, trade_log = [], []
running = False
status = {"msg": "Idle", "last": ""}
status_epoch = 0
error_message = ""
last_keepalive = 0

# Orders / quotes we opened for TP
open_tp_orders = []   # [{pair, side, qty, limit_px, since_ts}]

# P&L persistence (FIFO)
PROFIT_STATE_FILE = "profit_state.json"
profit_state = {"cumulative_pnl": 0.0, "daily": {}, "inventory": {}, "processed_orders": []}

def load_profit_state():
    try:
        with open(PROFIT_STATE_FILE, "r") as f:
            d = json.load(f)
        profit_state.update(d)
    except: pass

def save_profit_state():
    try:
        out = {
            "cumulative_pnl": float(profit_state.get("cumulative_pnl", 0.0)),
            "daily": {k: float(v) for k, v in profit_state.get("daily", {}).items()},
            "inventory": profit_state.get("inventory", {}),
            "processed_orders": profit_state.get("processed_orders", [])
        }
        with open(PROFIT_STATE_FILE, "w") as f:
            json.dump(out, f)
    except: pass

from collections import deque
def _inv_deque(market):
    dq = deque()
    for lot in profit_state["inventory"].get(market, []):
        try:
            q, c = float(lot[0]), float(lot[1])
            if q > 0 and c > 0: dq.append([q, c])
        except: pass
    return dq
def _set_inv(market, dq):
    profit_state["inventory"][market] = [[float(q), float(c)] for q, c in dq]

def apply_fill_update(market, side, price, qty, ts_ms, order_id):
    if not order_id or order_id in profit_state["processed_orders"]:
        return
    price, qty = float(price), float(qty)
    if price <= 0 or qty <= 0: return

    dq = _inv_deque(market)
    realized = 0.0
    if side.lower() == "buy":
        dq.append([qty, price])
    else:
        sell = qty
        while sell > 1e-18 and dq:
            q0, c0 = dq[0]
            used = min(sell, q0)
            realized += (price - c0) * used
            q0 -= used; sell -= used
            if q0 <= 1e-18: dq.popleft()
            else: dq[0][0] = q0

    _set_inv(market, dq)
    profit_state["processed_orders"].append(order_id)
    profit_state["cumulative_pnl"] = float(profit_state.get("cumulative_pnl", 0.0) + realized)
    dkey = ist_date()
    profit_state["daily"][dkey] = float(profit_state["daily"].get(dkey, 0.0) + realized)
    save_profit_state()

# -------------------- HTTP helpers --------------------
def hmac_signature(payload):
    return hmac.new(API_SECRET, payload.encode(), hashlib.sha256).hexdigest()

def _log_http_issue(prefix, r):
    try:
        body = r.text[:240] if hasattr(r, "text") else ""
        scan_log.append(f"{ist_now()} | {prefix} HTTP {r.status_code} | {body}")
    except Exception as e:
        scan_log.append(f"{ist_now()} | {prefix} log-fail: {e}")

def _signed_post(url, body):
    payload = json.dumps(body, separators=(',', ':'))
    sig = hmac_signature(payload)
    headers = {"X-AUTH-APIKEY": API_KEY, "X-AUTH-SIGNATURE": sig, "Content-Type": "application/json"}
    try:
        r = requests.post(url, headers=headers, data=payload, timeout=12)
        if not r.ok: _log_http_issue(f"POST {url}", r)
        if r.headers.get("content-type", "").startswith("application/json"):
            return r.json()
        return {}
    except Exception as e:
        scan_log.append(f"{ist_now()} | POST fail {url} | {e.__class__.__name__}: {e}")
        scan_log.append(traceback.format_exc().splitlines()[-1])
        return {}

def _keepalive_ping():
    if not APP_BASE_URL: return
    try:
        url = f"{APP_BASE_URL.rstrip('/')}/ping"
        headers = {}
        if KEEPALIVE_TOKEN:
            url += f"?t={KEEPALIVE_TOKEN}"
            headers["X-Keepalive-Token"] = KEEPALIVE_TOKEN
        requests.head(url, headers=headers, timeout=5)
    except: pass

# -------------------- Market rules --------------------
def fetch_pair_precisions():
    try:
        r = requests.get(f"{BASE_URL}/exchange/v1/markets_details", timeout=12)
        if not r.ok:
            _log_http_issue("markets_details", r); return
        for item in r.json():
            p = item.get("pair") or item.get("market") or item.get("coindcx_name")
            if p in PAIRS:
                PAIR_RULES[p] = {
                    "price_precision": int(item.get("target_currency_precision", 6)),
                    "qty_precision":   int(item.get("base_currency_precision", 6)),
                    "min_qty":         float(item.get("min_quantity", 0.0) or 0.0),
                    "min_notional":    float(item.get("min_notional", 0.0) or 0.0)
                }
        scan_log.append(f"{ist_now()} | market rules refreshed")
    except Exception as e:
        scan_log.append(f"{ist_now()} | markets_details fail: {e}")

# -------------------- Wallet / Prices --------------------
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
            return {d["market"]: {"price": float(d["last_price"]), "ts": now}
                    for d in r.json() if d.get("market") in PAIRS}
        else:
            _log_http_issue("ticker", r)
    except Exception as e:
        scan_log.append(f"{ist_now()} | ticker fail: {e}")
    return {}

# -------------------- Precision helpers --------------------
def _rules(pair): return PAIR_RULES.get(pair, {})
def _min_qty(pair): return float(_rules(pair).get("min_qty", 0.0) or 0.0)
def _qty_prec(pair): return int(_rules(pair).get("qty_precision", 6))
def _min_notional(pair): return float(_rules(pair).get("min_notional", 0.0) or 0.0)

def fmt_price(pair, price):
    pp = int(_rules(pair).get("price_precision", 6))
    return float(f"{float(price):.{pp}f}")

def fmt_qty(pair, qty):
    qp = _qty_prec(pair)
    mq = _min_qty(pair)
    q = max(float(qty), mq)
    q = float(f"{q:.{qp}f}")
    if q < mq: q = float(f"{mq:.{qp}f}")
    return q

def _qty_step(pair): return 10 ** (-_qty_prec(pair))
def _fee_mult(side): return 1.0 + FEE_PCT_PER_SIDE if side.upper() == "BUY" else 1.0 - FEE_PCT_PER_SIDE

def normalize_qty_for_side(pair, side, price, qty, usdt_avail, coin_avail):
    q = fmt_qty(pair, qty)
    mq = _min_qty(pair)
    if q < mq: q = fmt_qty(pair, mq)

    if side.upper() == "BUY":
        denom = max(price * _fee_mult("BUY") * BUY_HEADROOM, 1e-12)
        max_buy = usdt_avail / denom
        if q > max_buy:
            q = fmt_qty(pair, max_buy)
        step = _qty_step(pair)
        while q >= mq and (price * q * _fee_mult("BUY") * BUY_HEADROOM) > usdt_avail + 1e-12:
            q = fmt_qty(pair, max(0.0, q - step))
    else:
        if q > coin_avail:
            q = fmt_qty(pair, coin_avail)

    mn = _min_notional(pair)
    if q < mq or q <= 0: return 0.0
    if mn > 0 and (price * q) < mn:
        need = mn / max(price, 1e-9)
        q = fmt_qty(pair, max(q, need))
        if side.upper() == "BUY":
            denom = max(price * _fee_mult("BUY") * BUY_HEADROOM, 1e-12)
            max_buy = usdt_avail / denom
            if q > max_buy: q = fmt_qty(pair, max_buy)
        else:
            if q > coin_avail: q = fmt_qty(pair, coin_avail)
        if q < mq or (price*q) < mn: return 0.0
    return q

# -------------------- Candles & Indicators --------------------
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
            candle["low"]  = min(candle["low"], price)
            candle["close"] = price
            candle["volume"] += 1
    if candle: candles.append(candle)
    candle_logs[pair] = candles[-600:]  # keep last ~50 mins of 5s bars

def _ema(vals, n):
    if len(vals) < n: return None
    k = 2/(n+1)
    ema = sum(vals[:n])/n
    for v in vals[n:]:
        ema = v*k + ema*(1-k)
    return ema

def _macd(vals):
    if len(vals) < MACD_SLOW + MACD_SIG: return None, None, None
    # classic MACD with EMAs
    def ema_series(values, n):
        k = 2/(n+1)
        e = []
        ema = sum(values[:n])/n
        e.extend([None]*(n-1))
        e.append(ema)
        for v in values[n:]:
            ema = v*k + ema*(1-k)
            e.append(ema)
        return e
    e12 = ema_series(vals, MACD_FAST)
    e26 = ema_series(vals, MACD_SLOW)
    macd_line = []
    for i in range(len(vals)):
        if e12[i] is None or e26[i] is None: macd_line.append(None)
        else: macd_line.append(e12[i] - e26[i])
    # signal
    valid = [x for x in macd_line if x is not None]
    if len(valid) < MACD_SIG: return None, None, None
    sig_series = ema_series(valid, MACD_SIG)
    # align signal with macd_line length
    sig_full = [None]*(len(macd_line)-len(sig_series)) + sig_series
    hist = [None if (macd_line[i] is None or sig_full[i] is None) else macd_line[i]-sig_full[i] for i in range(len(macd_line))]
    return macd_line[-1], sig_full[-1], hist[-1]

# --- simple SMC "gate" (trend bias using last two closes) ---
def _smc_gate(pair):
    cs = candle_logs.get(pair) or []
    if len(cs) < 3: return "closed"
    c1, c2 = cs[-3], cs[-2]
    if c2["close"] > c1["high"]:  # impulsive up close
        return "buy-open"
    if c2["close"] < c1["low"]:   # impulsive down close
        return "sell-open"
    return "closed"

# -------------------- Exchange order helpers --------------------
def place_market(pair, side, qty):
    payload = {
        "market": pair,
        "side": side.lower(),
        "order_type": "market_order",
        "total_quantity": f"{fmt_qty(pair, qty)}",
        "timestamp": int(time.time()*1000)
    }
    scan_log.append(f"{ist_now()} | {pair} | PRE-ORDER {side} qty={payload['total_quantity']} @ MKT")
    res = _signed_post(f"{BASE_URL}/exchange/v1/orders/create", payload) or {}
    return res

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

def _fnum(x, d=0.0):
    try: return float(x)
    except: return d

def _record_fill_from_status(market, side, st, order_id):
    if not isinstance(st, dict): return 0.0, 0.0
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
            ts_ms = int(time.time()*1000)
        apply_fill_update(market, side, avg_px, filled, ts_ms, order_id)
    return filled, avg_px

# -------------------- Strategy loop --------------------
_autostart_lock = threading.Lock()

def _ema_state(fast, slow):
    if fast is None or slow is None: return "None"
    if abs(fast - slow) / max(slow, 1e-9) < 0.001: return "flat"
    return "bull" if fast > slow else "bear"

def scan_loop():
    global running, error_message, status_epoch, last_keepalive
    scan_log.clear()
    running = True

    while running:
        # keepalive
        now_real = time.time()
        if now_real - last_keepalive >= KEEPALIVE_SEC:
            _keepalive_ping()
            last_keepalive = now_real

        prices = fetch_all_prices()
        now_ts = int(time.time())

        # ticks -> candles
        for pair in PAIRS:
            if pair in prices:
                px = prices[pair]["price"]
                tick_logs[pair].append((now_ts, px))
                if len(tick_logs[pair]) > 4000:
                    tick_logs[pair] = tick_logs[pair][-4000:]
                aggregate_candles(pair, CANDLE_INTERVAL)

        balances = get_wallet_balances()
        free_usdt = float(balances.get("USDT", 0.0))

        # per pair scan
        for pair in PAIRS:
            cs = candle_logs.get(pair) or []
            if len(cs) < max(EMA_SLOW, MACD_SLOW) + 3:
                continue

            closes = [c["close"] for c in cs[:-1]] + [cs[-1]["close"]]
            ema_fast = _ema(closes, EMA_FAST)
            ema_slow = _ema(closes, EMA_SLOW)
            ema_state = _ema_state(ema_fast, ema_slow)

            macd_line, macd_sig, macd_hist = _macd(closes)
            if macd_line is None or macd_sig is None:
                macd_state = "neutral"
            else:
                if abs(macd_line - macd_sig) < 1e-9:
                    macd_state = "neutral"
                else:
                    macd_state = "bull" if macd_line > macd_sig else "bear"

            smc_gate = _smc_gate(pair)

            last_price = closes[-1]
            decision = "HOLD"
            reason = []

            # BUY: EMA bull cross + MACD bull
            if ema_state == "bull" and macd_state == "bull":
                # qty from 30% USDT
                usdt_to_use = free_usdt * BUY_USDT_FRACTION
                if usdt_to_use > 1e-6:
                    qty_raw = usdt_to_use / max(last_price, 1e-9)
                    qty = normalize_qty_for_side(pair, "BUY", last_price, qty_raw,
                                                 usdt_avail=free_usdt, coin_avail=0.0)
                    if qty > 0:
                        res = place_market(pair, "BUY", qty)
                        oid = _extract_order_id(res)
                        st = get_order_status(order_id=oid) if oid else {}
                        filled, avg_px = _record_fill_from_status(pair, "BUY", st, oid)
                        if filled > 0 and avg_px > 0:
                            # open TP limit
                            tp_px = fmt_price(pair, avg_px * (1.0 + TP_PCT))
                            open_tp_orders.append({
                                "pair": pair, "side": "SELL", "qty": fmt_qty(pair, filled),
                                "limit_px": tp_px, "since_ts": int(time.time())
                            })
                            trade_log.append({
                                "ts": ist_now(), "pair": pair, "side": "BUY",
                                "qty": filled, "price": avg_px, "realized": 0.0
                            })
                            decision = "BUY"
                            reason.append(f"qty={filled} avg={avg_px} tp={tp_px}")
                        else:
                            reason.append("buy failed/zero fill")
                    else:
                        reason.append("qty<min or notional fail")
                else:
                    reason.append("USDT low")

            # SELL: EMA bear + MACD bear — sell coin balance
            elif ema_state == "bear" and macd_state == "bear":
                coin = pair[:-4]
                coin_bal = float(balances.get(coin, 0.0))
                qty_raw = coin_bal * SELL_COIN_FRACTION
                qty = normalize_qty_for_side(pair, "SELL", last_price, qty_raw,
                                             usdt_avail=free_usdt, coin_avail=coin_bal)
                if qty > 0:
                    res = place_market(pair, "SELL", qty)
                    oid = _extract_order_id(res)
                    st = get_order_status(order_id=oid) if oid else {}
                    filled, avg_px = _record_fill_from_status(pair, "SELL", st, oid)
                    if filled > 0 and avg_px > 0:
                        trade_log.append({
                            "ts": ist_now(), "pair": pair, "side": "SELL",
                            "qty": filled, "price": avg_px, "realized": 0.0
                        })
                        decision = "SELL"
                        reason.append(f"qty={filled} avg={avg_px}")
                    else:
                        reason.append("sell failed/zero fill")
                else:
                    reason.append("no coin/min fail")

            # Log the scan line with gates + decision
            scan_log.append(
                f"{ist_now()} | {pair} | ema={ema_state} macd={macd_state} smc={smc_gate} "
                f"→ {decision}{(' [' + ', '.join(reason) + ']') if reason else ''}"
            )

        status["msg"], status["last"] = "Running", ist_now()
        status_epoch = int(time.time())
        time.sleep(POLL_SEC)

    status["msg"] = "Idle"

def compute_realized_pnl_today():
    return round(profit_state["daily"].get(ist_date(), 0.0), 6)

# -------------------- Routes --------------------
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
    return jsonify({"status": "stopped"})

@app.route("/status")
def get_status():
    balances = get_wallet_balances()
    usdt_total = balances.get("USDT", 0.0)
    coins = {pair[:-4]: balances.get(pair[:-4], 0.0) for pair in PAIRS}

    # keepalive info for UI
    now_real = time.time()
    last_age = now_real - (last_keepalive or 0)
    keepalive_info = {
        "enabled": bool(KEEPALIVE_TOKEN),
        "interval_sec": KEEPALIVE_SEC,
        "last_ping_epoch": int(last_keepalive or 0),
        "last_ping_age_sec": (max(0, int(last_age)) if last_keepalive else None),
        "next_due_sec": (max(0, int(KEEPALIVE_SEC - last_age)) if last_keepalive else None),
        "app_base_url": APP_BASE_URL,
    }

    return jsonify({
        "status": status["msg"],
        "last": status["last"],
        "status_epoch": status_epoch,
        "usdt": usdt_total,
        "coins": coins,
        "profit_today": compute_realized_pnl_today(),
        "profit_yesterday": round(profit_state["daily"].get(ist_yesterday(), 0.0), 6),
        "pnl_cumulative": round(profit_state.get("cumulative_pnl", 0.0), 6),
        "trades": trade_log[-12:][::-1],
        "scans": scan_log[-120:][::-1],           # show more signal lines
        "open_tp": open_tp_orders[-20:],
        "keepalive": keepalive_info,
        "error": error_message or "-"
    })

@app.route("/ping", methods=["GET", "HEAD"])
def ping():
    token = os.environ.get("KEEPALIVE_TOKEN", "")
    provided = (request.args.get("t") or request.headers.get("X-Keepalive-Token") or "")
    if token and provided != token:
        print(f"[{ist_now()}] /ping forbidden (bad token) method={request.method}")
        return "forbidden", 403
    print(f"[{ist_now()}] /ping ok method={request.method}")
    if request.method == "HEAD":
        return ("", 200)
    return ("pong", 200)

# -------------------- Safe autostart --------------------
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

if os.environ.get("AUTOSTART", "1") == "1":
    threading.Thread(target=_boot_kick, daemon=True).start()

if __name__ == "__main__":
    load_profit_state()
    fetch_pair_precisions()
    _start_loop_once()
    port = int(os.environ.get("PORT", "10000"))
    app.run(host="0.0.0.0", port=port)
