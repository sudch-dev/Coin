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

APP_BASE_URL = os.environ.get("APP_BASE_URL", "https://coin-4k37.onrender.com")
KEEPALIVE_TOKEN = os.environ.get("KEEPALIVE_TOKEN", "")

# -------------------- API / Markets --------------------
API_KEY = os.environ.get("API_KEY")
API_SECRET_RAW = os.environ.get("API_SECRET", "")
API_SECRET = API_SECRET_RAW.encode() if isinstance(API_SECRET_RAW, str) else API_SECRET_RAW
BASE_URL = "https://api.coindcx.com"

# No LTC per your request
PAIRS = [
    "BTCUSDT", "ETHUSDT", "XRPUSDT", "SHIBUSDT", "SOLUSDT",
    "DOGEUSDT", "ADAUSDT", "AEROUSDT", "BNBUSDT"
]

# Local fallbacks; overwritten by live fetch at boot and periodically thereafter
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

# -------------------- Strategy settings (simple trend) --------------------
CANDLE_INTERVAL = 5      # 5-second bars
POLL_SEC = 1.0
KEEPALIVE_SEC = 240

# Fees & formatting
FEE_PCT_PER_SIDE = 0.0010
BUY_HEADROOM = 1.0005    # cushion so rounded qty still fits balance+fee

# Indicators
EMA_FAST_N = 5
EMA_SLOW_N = 20

# MACD(12,26,9)
MACD_FAST = 12
MACD_SLOW = 26
MACD_SIGNAL = 9

# Divergence lookback
DIV_LOOKBACK = 60

# Risk / sizing
BUY_USDT_FRAC = 0.30   # 30% of free USDT for entry
TP_PROFIT_PCT = 0.010  # 1.0% TP

# -------------------- Time helpers --------------------
IST = timezone('Asia/Kolkata')
def ist_now(): return datetime.now(IST).strftime('%Y-%m-%d %H:%M:%S')
def ist_date(): return datetime.now(IST).strftime('%Y-%m-%d')
def ist_yesterday(): return (datetime.now(IST) - timedelta(days=1)).strftime('%Y-%m-%d')

# -------------------- State --------------------
tick_logs, candle_logs = {p: [] for p in PAIRS}, {p: [] for p in PAIRS}
scan_log, trade_log = [], []
running = False
status = {"msg": "Idle", "last": ""}
status_epoch = 0
error_message = ""

# --- Order book (for UI) ---
order_book = {}  # id -> {id,pair,side,type,px,qty,status,created,updated,reason}
ORDER_BOOK_MAX = 200

# Track per-pair TP order id
tp_orders = {p: None for p in PAIRS}

# rules refresh cadence
_last_rules_refresh = 0
RULES_REFRESH_SEC = 1800  # 30 mins

# -------------------- Persistent P&L (FIFO) --------------------
PROFIT_STATE_FILE = "profit_state.json"
profit_state = {
    "cumulative_pnl": 0.0,
    "daily": {},
    "inventory": {},
    "processed_orders": []
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

def _avg_entry_price(market):
    dq = _get_inventory_deque(market)
    tot_q = sum(q for q, _ in dq)
    if tot_q <= 0: return 0.0
    tot_cost = sum(q * c for q, c in dq)
    return tot_cost / tot_q if tot_q > 0 else 0.0

def apply_fill_update(market, side, price, qty, ts_ms, order_id):
    if not order_id or order_id in profit_state["processed_orders"]:
        return
    try:
        price = float(price); qty = float(qty)
    except:
        return
    if price <= 0 or qty <= 0:
        return

    inv = _get_inventory_deque(market)
    realized = 0.0

    if side.lower() == "buy":
        inv.append([qty, price])
        trade_log.append(f"{ist_now()} | BUY FILL {market} {round(qty,8)} @ {price}")
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
        trade_log.append(f"{ist_now()} | SELL FILL {market} {round(qty,8)} @ {price}")

    _set_inventory_from_deque(market, inv)
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

def _order_book_upsert(oid, **fields):
    if not oid:
        return
    ob = order_book.get(oid, {})
    ob.update(fields)
    now = int(time.time())
    if "created" not in ob:
        ob["created"] = now
    ob["updated"] = now
    order_book[oid] = ob
    if len(order_book) > ORDER_BOOK_MAX:
        oldest = sorted(order_book.items(), key=lambda kv: kv[1].get("updated", 0))[:20]
        for k, _ in oldest:
            order_book.pop(k, None)

def _signed_post(url, body):
    payload = json.dumps(body, separators=(',', ':'))
    sig = hmac_signature(payload)
    headers = {"X-AUTH-APIKEY": API_KEY or "", "X-AUTH-SIGNATURE": sig, "Content-Type": "application/json"}
    try:
        if "/exchange/v1/orders/" in url:
            kind = "CREATE" if url.endswith("/create") else "CANCEL" if url.endswith("/cancel") else "STATUS"
            safe = {
                "kind": kind,
                "market": body.get("market"),
                "side": body.get("side"),
                "type": body.get("order_type"),
                "qty": body.get("total_quantity"),
                "price": body.get("price_per_unit"),
                "id": body.get("id") or body.get("client_order_id")
            }
            scan_log.append(f"{ist_now()} | POST->EXCHANGE {kind} | {url} | {safe}")

        r = requests.post(url, headers=headers, data=payload, timeout=12)
        if not r.ok:
            _log_http_issue(f"POST {url}", r)
        if r.headers.get("content-type", "").startswith("application/json"):
            return r.json()
        return {}
    except Exception as e:
        scan_log.append(f"{ist_now()} | POST fail {url} | {e.__class__.__name__}: {e}")
        scan_log.append(traceback.format_exc().splitlines()[-1])
        return {}

def _keepalive_ping():
    try:
        if not APP_BASE_URL:
            return
        url = f"{APP_BASE_URL}/ping"
        if KEEPALIVE_TOKEN:
            url = f"{url}?t={KEEPALIVE_TOKEN}"
            headers = {"X-Keepalive-Token": KEEPALIVE_TOKEN}
        else:
            headers = {}
        requests.get(url, headers=headers, timeout=5)
    except Exception:
        pass

# -------------------- Market rules (precision / min qty / notional) --------------------
def fetch_pair_precisions():
    try:
        r = requests.get(f"{BASE_URL}/exchange/v1/markets_details", timeout=12)
        if not r.ok:
            _log_http_issue("markets_details", r)
            return
        data = r.json()
        for item in data:
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

# -------------------- Balances / Prices --------------------
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

# -------------------- Precision, fees & min-qty helpers --------------------
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
    if q < mq:
        q = float(f"{mq:.{qp}f}")
    return q

def _qty_step(pair):
    return 10 ** (-_qty_prec(pair))

def meets_min_notional(pair, price, qty):
    mn = _min_notional(pair)
    return (price * qty) >= mn

def _fee_multiplier(side):
    return 1.0 + FEE_PCT_PER_SIDE if side.upper() == "BUY" else 1.0 - FEE_PCT_PER_SIDE

def _affordable_buy_qty(pair, price, usdt_avail):
    if price <= 0:
        return 0.0
    eff = usdt_avail / (price * _fee_multiplier("BUY") * BUY_HEADROOM)
    return max(0.0, eff)

def _learn_precision_from_error(pair, msg_lower):
    if not isinstance(msg_lower, str):
        return False
    import re
    m = re.search(r'([A-Z]{3,5})\s+precision\s+should\s+be\s+(\d+)', msg_lower.upper())
    if not m:
        return False
    cur = m.group(1)
    pval = int(m.group(2))
    base = pair[:-4]
    quote = pair[-4:]
    if cur == base:
        PAIR_RULES.setdefault(pair, {})["qty_precision"] = pval
        return True
    if cur == quote:
        PAIR_RULES.setdefault(pair, {})["price_precision"] = pval
        return True
    return False

def normalize_qty_for_side(pair, side, price, qty, usdt_avail, coin_avail):
    """
    Enforce precision, min_qty, wallet caps (fee+headroom for BUY), min_notional.
    """
    q = fmt_qty(pair, qty)
    mq = _min_qty(pair)
    if q < mq:
        q = fmt_qty(pair, mq)

    if side.upper() == "BUY":
        denom = max(price * _fee_multiplier("BUY") * BUY_HEADROOM, 1e-12)
        max_buyable = usdt_avail / denom
        if q > max_buyable:
            q = fmt_qty(pair, max_buyable)
        step = _qty_step(pair)
        while q >= mq and (price * q * _fee_multiplier("BUY") * BUY_HEADROOM) > usdt_avail + 1e-12:
            q = fmt_qty(pair, max(0.0, q - step))
    else:
        if q > coin_avail:
            q = fmt_qty(pair, coin_avail)

    if q < mq or q <= 0:
        return 0.0

    mn = _min_notional(pair)
    if mn > 0 and (price * q) < mn:
        needed_q = mn / max(price, 1e-9)
        q = fmt_qty(pair, max(q, needed_q))
        if side.upper() == "BUY":
            denom = max(price * _fee_multiplier("BUY") * BUY_HEADROOM, 1e-12)
            max_buyable = usdt_avail / denom
            if q > max_buyable:
                q = fmt_qty(pair, max_buyable)
            step = _qty_step(pair)
            while q >= mq and (price * q * _fee_multiplier("BUY") * BUY_HEADROOM) > usdt_avail + 1e-12:
                q = fmt_qty(pair, max(0.0, q - step))
        else:
            if q > coin_avail:
                q = fmt_qty(pair, coin_avail)
        if q < mq or (mn > 0 and (price * q) < mn):
            return 0.0
    return q

# -------------------- Candles / Indicators --------------------
def aggregate_candles(pair, interval=CANDLE_INTERVAL):
    ticks = tick_logs[pair]
    if not ticks:
        return
    candles, candle, last_window = [], None, None
    for ts, price in sorted(ticks, key=lambda x: x[0]):
        wstart = ts - (ts % interval)
        if last_window != wstart:
            if candle:
                candles.append(candle)
            candle = {"open": price, "high": price, "low": price, "close": price, "volume": 1, "start": wstart}
            last_window = wstart
        else:
            candle["high"] = max(candle["high"], price)
            candle["low"] = min(candle["low"], price)
            candle["close"] = price
            candle["volume"] += 1
    if candle:
        candles.append(candle)
    candle_logs[pair] = candles[-1500:]  # keep enough (~2 hours of 5s bars)

def _ema_series(vals, n):
    if len(vals) < 1:
        return []
    k = 2 / (n + 1)
    out = []
    ema = None
    for i, v in enumerate(vals):
        if ema is None:
            # seed with SMA of first n when possible
            if i+1 >= n:
                ema = sum(vals[i+1-n:i+1]) / n
            else:
                out.append(None); continue
        else:
            ema = v * k + ema * (1 - k)
        out.append(ema)
    return out

def _ema(vals, n):
    s = _ema_series(vals, n)
    return s[-1] if s else None

def _macd(vals, fast=MACD_FAST, slow=MACD_SLOW, signal=MACD_SIGNAL):
    if len(vals) < slow + signal + 2:
        return None, None, None, None, None
    ema_fast = _ema_series(vals, fast)
    ema_slow = _ema_series(vals, slow)
    macd_line = []
    for i in range(len(vals)):
        if ema_fast[i] is None or ema_slow[i] is None:
            macd_line.append(None)
        else:
            macd_line.append(ema_fast[i] - ema_slow[i])
    sig_line = _ema_series([x for x in macd_line], signal)
    hist = []
    for i in range(len(vals)):
        if macd_line[i] is None or sig_line[i] is None:
            hist.append(None)
        else:
            hist.append(macd_line[i] - sig_line[i])
    # latest and previous
    return macd_line[-1], sig_line[-1], hist[-1], macd_line[-2], sig_line[-2]

def _divergence_price_macd(closes, macd_line, lookback=DIV_LOOKBACK):
    # very light detection: last two swing lows/highs via local extrema search
    n = len(closes)
    if n < lookback + 10:
        return None
    start = max(0, n - lookback)
    # collect local highs/lows in region
    highs, lows = [], []
    for i in range(start+2, n-2):
        if closes[i] is None or macd_line[i] is None:
            continue
        if closes[i] >= closes[i-1] and closes[i] >= closes[i-2] and closes[i] >= closes[i+1] and closes[i] >= closes[i+2]:
            highs.append(i)
        if closes[i] <= closes[i-1] and closes[i] <= closes[i-2] and closes[i] <= closes[i+1] and closes[i] <= closes[i+2]:
            lows.append(i)
    if len(lows) >= 2:
        i1, i2 = lows[-2], lows[-1]  # previous low, recent low
        if closes[i2] < closes[i1] and macd_line[i2] > macd_line[i1]:
            return "bullish"
    if len(highs) >= 2:
        j1, j2 = highs[-2], highs[-1]
        if closes[j2] > closes[j1] and macd_line[j2] < macd_line[j1]:
            return "bearish"
    return None

# -------------------- Exchange calls --------------------
def place_order(pair, side, qty, price_hint=None, balances=None):
    usdt_avail = (balances or {}).get("USDT", 0.0)
    coin_avail = (balances or {}).get(pair[:-4], 0.0)
    price = float(price_hint or 0.0)

    q_pre = fmt_qty(pair, qty)
    q = normalize_qty_for_side(pair, side, price if price > 0 else 1.0, q_pre, usdt_avail, coin_avail)
    if q <= 0:
        scan_log.append(f"{ist_now()} | {pair} | SKIP market {side} — min/fees/wallet/notional fail")
        return {}
    payload = {
        "market": pair,
        "side": side.lower(),
        "order_type": "market_order",
        "total_quantity": f"{q}",
        "timestamp": int(time.time() * 1000)
    }
    scan_log.append(f"{ist_now()} | {pair} | PRE-ORDER {side} qty={payload['total_quantity']} @ MKT")
    res = _signed_post(f"{BASE_URL}/exchange/v1/orders/create", payload) or {}
    oid = _extract_order_id(res)

    if isinstance(res, dict) and res.get("status") == "error":
        _order_book_upsert(
            oid or f"rej-{int(time.time()*1000)}",
            pair=pair, side=side.lower(), type="market",
            px=price, qty=q, status="rejected", reason=res.get("message", "error")
        )
        return res

    if oid:
        _order_book_upsert(oid, pair=pair, side=side.lower(), type="market",
                           px=price, qty=q, status="open")
        st = get_order_status(order_id=oid)
        filled, avg_px = _record_fill_from_status(pair, side.upper(), st, oid)
        if filled > 0:
            _order_book_upsert(oid, status="filled", px=avg_px, qty=filled)
    return res

def place_limit_order(pair, side, qty, price, balances=None):
    """
    Returns (res, q_used, p_used).
    """
    p = fmt_price(pair, price)
    usdt_avail = (balances or {}).get("USDT", 0.0)
    coin_avail = (balances or {}).get(pair[:-4], 0.0)

    q = normalize_qty_for_side(pair, side, p, fmt_qty(pair, qty), usdt_avail, coin_avail)
    if q <= 0:
        scan_log.append(f"{ist_now()} | {pair} | SKIP limit {side} {p} — min/fees/wallet/notional fail")
        return {}, 0.0, p

    payload = {
        "market": pair,
        "side": side.lower(),
        "order_type": "limit_order",
        "price_per_unit": f"{p}",
        "total_quantity": f"{q}",
        "timestamp": int(time.time() * 1000)
    }
    scan_log.append(f"{ist_now()} | {pair} | PRE-ORDER {side} qty={payload['total_quantity']} @ {payload['price_per_unit']}")

    res = _signed_post(f"{BASE_URL}/exchange/v1/orders/create", payload) or {}
    oid = _extract_order_id(res)
    if isinstance(res, dict) and res.get("status") == "error":
        _order_book_upsert(
            oid or f"rej-{int(time.time()*1000)}",
            pair=pair, side=side.lower(), type="limit",
            px=p, qty=q, status="rejected", reason=res.get("message", "error")
        )
        return res, q, p

    if oid:
        _order_book_upsert(oid, pair=pair, side=side.lower(), type="limit",
                           px=p, qty=q, status="open")
    return res, q, p

def cancel_order(order_id=None, client_order_id=None):
    body = {"timestamp": int(time.time() * 1000)}
    if order_id: body["id"] = order_id
    if client_order_id: body["client_order_id"] = client_order_id
    res = _signed_post(f"{BASE_URL}/exchange/v1/orders/cancel", body) or {}
    return res

def get_order_status(order_id=None, client_order_id=None):
    body = {"timestamp": int(time.time() * 1000)}
    if order_id: body["id"] = order_id
    if client_order_id: body["client_order_id"] = client_order_id
    res = _signed_post(f"{BASE_URL}/exchange/v1/orders/status", body)
    return res if isinstance(res, dict) else {}

def _extract_order_id(res: dict):
    if not isinstance(res, dict):
        return None
    try:
        if isinstance(res.get("orders"), list) and res["orders"]:
            o = res["orders"][0]
            return str(o.get("id") or o.get("order_id") or o.get("client_order_id") or "")
    except:
        pass
    return str(res.get("id") or res.get("order_id") or res.get("client_order_id") or res.get("orderId") or "") or None

def _fnum(x, d=0.0):
    try:
        return float(x)
    except:
        return d

def _record_fill_from_status(market, side, st, order_id):
    if not isinstance(st, dict): return 0.0, 0.0
    total_q  = _fnum(st.get("total_quantity", st.get("quantity", st.get("orig_qty", 0))))
    remain_q = _fnum(st.get("remaining_quantity", st.get("remaining_qty", st.get("leaves_qty", 0))))
    exec_q   = _fnum(st.get("executed_quantity", st.get("filled_qty", st.get("executedQty", 0))))
    filled   = exec_q if exec_q > 0 else max(0.0, total_q - remain_q)
    avg_px   = _fnum(st.get("avg_price", st.get("average_price", st.get("avg_execution_price", st.get("price", 0)))))
    if filled > 0 and avg_px > 0:
        ts_field = st.get("updated_at") or st.get("created_at") or st.get("timestamp") or int(time.time() * 1000)
        try:
            ts_ms = int(ts_field)
            if ts_ms < 10 ** 12: ts_ms *= 1000
        except:
            ts_ms = int(time.time() * 1000)
        apply_fill_update(market, side, avg_px, filled, ts_ms, order_id)
    return filled, avg_px

# -------------------- Strategy core --------------------
def _ema_cross(prev_fast, prev_slow, cur_fast, cur_slow):
    cross_up = False
    cross_dn = False
    if prev_fast is not None and prev_slow is not None and cur_fast is not None and cur_slow is not None:
        cross_up = (cur_fast >= cur_slow) and (prev_fast < prev_slow)
        cross_dn = (cur_fast <= cur_slow) and (prev_fast > prev_slow)
    return cross_up, cross_dn

def _net_inventory_units(market):
    dq = _get_inventory_deque(market)
    return sum(q for q, _ in dq)

def _position_entry_avg(market):
    return _avg_entry_price(market)

def _ensure_tp_order(pair, last_px, balances):
    """
    If we have a long position and no TP limit working, place at +1%.
    """
    pos = _net_inventory_units(pair)
    if pos <= 0:
        return
    if tp_orders.get(pair):
        return
    entry = _position_entry_avg(pair)
    if entry <= 0:
        return
    tp_px = fmt_price(pair, entry * (1.0 + TP_PROFIT_PCT))
    res, q_used, p_used = place_limit_order(pair, "SELL", pos, tp_px, balances=balances)
    oid = _extract_order_id(res)
    if oid:
        tp_orders[pair] = oid
        scan_log.append(f"{ist_now()} | {pair} | TP LIMIT placed {q_used} @ {p_used} (target +{round(TP_PROFIT_PCT*100,2)}%)")

def _cancel_tp(pair):
    oid = tp_orders.get(pair)
    if not oid:
        return
    res = cancel_order(order_id=oid)
    _order_book_upsert(oid, status="cancelled", reason="tp_replaced_or_signal")
    tp_orders[pair] = None
    scan_log.append(f"{ist_now()} | {pair} | TP LIMIT cancel id={oid} | res={res}")

def _check_tp_filled(pair):
    oid = tp_orders.get(pair)
    if not oid:
        return False
    st = get_order_status(order_id=oid)
    status_txt = (st.get("status") or "").lower()
    rem = _fnum(st.get("remaining_quantity", st.get("remaining_qty", st.get("leaves_qty", 0))))
    filled = (rem == 0) or ("filled" in status_txt and "part" not in status_txt)
    if filled:
        _order_book_upsert(oid, status="filled")
        tp_orders[pair] = None
        scan_log.append(f"{ist_now()} | {pair} | TP LIMIT filled | st={st}")
        return True
    return False

# -------------------- Main loop --------------------
last_keepalive = 0
_autostart_lock = threading.Lock()

def scan_loop():
    global running, error_message, status_epoch, last_keepalive, _last_rules_refresh
    scan_log.clear()
    running = True

    while running:
        now_real = time.time()
        if now_real - last_keepalive >= KEEPALIVE_SEC:
            _keepalive_ping()
            last_keepalive = now_real

        prices = fetch_all_prices()

        if (time.time() - _last_rules_refresh) >= RULES_REFRESH_SEC:
            fetch_pair_precisions()
            _last_rules_refresh = time.time()

        now_ts = int(time.time())
        balances = get_wallet_balances()

        # ticks & candles
        for pair in PAIRS:
            if pair in prices:
                px = prices[pair]["price"]
                tick_logs[pair].append((now_ts, px))
                if len(tick_logs[pair]) > 6000:
                    tick_logs[pair] = tick_logs[pair][-6000:]
                aggregate_candles(pair, CANDLE_INTERVAL)

        # per-pair strategy
        for pair in PAIRS:
            if pair not in prices:
                continue

            last = prices[pair]["price"]

            cs = candle_logs.get(pair) or []
            if len(cs) < max(EMA_SLOW_N, MACD_SLOW) + MACD_SIGNAL + 5:
                scan_log.append(f"{ist_now()} | {pair} | WARMUP bars={len(cs)}")
                continue

            closes = [c["close"] for c in cs]
            # Use completed candle close and include last price as most recent
            series = [c["close"] for c in cs[:-1]] + [last]

            ema_fast_series = _ema_series(series, EMA_FAST_N)
            ema_slow_series = _ema_series(series, EMA_SLOW_N)
            ema_fast = ema_fast_series[-1]
            ema_slow = ema_slow_series[-1]
            prev_fast = ema_fast_series[-2]
            prev_slow = ema_slow_series[-2]
            cross_up, cross_dn = _ema_cross(prev_fast, prev_slow, ema_fast, ema_slow)

            macd_now, macd_sig, macd_hist, macd_prev, sig_prev = _macd(series, MACD_FAST, MACD_SLOW, MACD_SIGNAL)
            macd_bull = (macd_now is not None and macd_sig is not None and macd_now > macd_sig and (macd_prev is not None and sig_prev is not None and macd_now >= macd_prev))
            macd_bear = (macd_now is not None and macd_sig is not None and macd_now < macd_sig and (macd_prev is not None and sig_prev is not None and macd_now <= macd_prev))

            # divergence detection using MACD line vs price
            # recompute macd line series quickly for divergence
            # (reuse ema series to compute macd list)
            _macd_line_series = []
            ef = _ema_series(series, MACD_FAST)
            es = _ema_series(series, MACD_SLOW)
            for i in range(len(series)):
                if ef[i] is None or es[i] is None:
                    _macd_line_series.append(None)
                else:
                    _macd_line_series.append(ef[i] - es[i])

            div = _divergence_price_macd(series, _macd_line_series, DIV_LOOKBACK)

            # gates
            allow_buy = (cross_up and macd_bull) or (div == "bullish" and ema_fast >= ema_slow)
            allow_sell_signal = (cross_dn and macd_bear) or (div == "bearish" and ema_fast <= ema_slow)

            # position & entry avg
            pos_units = _net_inventory_units(pair)
            entry_avg = _position_entry_avg(pair)
            roi_pct = ((last / entry_avg) - 1.0) if (entry_avg > 0 and pos_units > 0) else 0.0

            # log signals
            scan_log.append(
                f"{ist_now()} | {pair} | EMA5={round(ema_fast,8)} EMA20={round(ema_slow,8)} "
                f"| xUp={cross_up} xDn={cross_dn} | MACD={None if macd_now is None else round(macd_now,8)} "
                f"SIG={None if macd_sig is None else round(macd_sig,8)} HIST={None if macd_hist is None else round(macd_hist,8)} "
                f"| macdBull={macd_bull} macdBear={macd_bear} | div={div} "
                f"| allowBUY={allow_buy} allowSELLsig={allow_sell_signal} "
                f"| pos={round(pos_units,8)} entry={round(entry_avg,8)} roi%={round(roi_pct*100,3)}"
            )

            # 1) ENTRY (only if flat)
            if pos_units <= 0 and allow_buy:
                usdt_free = balances.get("USDT", 0.0)
                use_usdt = max(0.0, usdt_free * BUY_USDT_FRAC)
                qty = use_usdt / max(last, 1e-9)
                qty = normalize_qty_for_side(pair, "BUY", last, qty, usdt_free, 0.0)
                if qty > 0:
                    res = place_order(pair, "BUY", qty, price_hint=last, balances=balances)
                    oid = _extract_order_id(res)
                    if oid:
                        # After market fill, place TP if we still hold
                        pos_units = _net_inventory_units(pair)
                        if pos_units > 0:
                            _ensure_tp_order(pair, last, balances)

            # 2) EXIT (whichever occurs first: sell signal, or +1% TP)
            elif pos_units > 0:
                # if TP is already working, check if filled, else keep it
                if tp_orders.get(pair):
                    _check_tp_filled(pair)

                # SELL signal: exit immediately (market), cancel any TP
                if allow_sell_signal:
                    if tp_orders.get(pair):
                        _cancel_tp(pair)
                    qty = fmt_qty(pair, pos_units)
                    if qty > 0:
                        res = place_order(pair, "SELL", qty, price_hint=last, balances=balances)
                        # no SL; this is only on signal
                        tp_orders[pair] = None
                        continue

                # If ROI >= +1% and no TP order, (re)place TP
                if roi_pct >= TP_PROFIT_PCT:
                    if not tp_orders.get(pair):
                        _ensure_tp_order(pair, last, balances)

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
    # cancel any TP orders on stop
    try:
        for p in PAIRS:
            if tp_orders.get(p):
                try:
                    cancel_order(order_id=tp_orders[p])
                except:
                    pass
                tp_orders[p] = None
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

    # order book snapshot
    ob_items = sorted(order_book.values(), key=lambda x: x.get("updated", 0), reverse=True)
    ob_open = [o for o in ob_items if o.get("status") == "open"][:50]
    ob_filled = [o for o in ob_items if o.get("status") == "filled"][:50]
    ob_cancelled = [o for o in ob_items if o.get("status") == "cancelled"][:50]
    ob_rejected = [o for o in ob_items if o.get("status") == "rejected"][:50]

    # keepalive block for UI
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

    # per-pair position summary
    positions = {}
    for p in PAIRS:
        positions[p] = {
            "units": round(_net_inventory_units(p), 8),
            "avg_entry": round(_position_entry_avg(p), 8),
            "tp_order": tp_orders.get(p)
        }

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
        "positions": positions,
        "coins": coins,
        "order_book": {
            "open": ob_open,
            "filled": ob_filled,
            "cancelled": ob_cancelled,
            "rejected": ob_rejected
        },
        "trades": trade_log[-15:][::-1],
        "scans": scan_log[-120:][::-1],
        "error": error_message,
        "keepalive": keepalive_info
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
_autostart_lock = threading.Lock()
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

# Also autostart when running directly
if __name__ == "__main__":
    load_profit_state()
    fetch_pair_precisions()
    _start_loop_once()
    port = int(os.environ.get("PORT", "10000"))
    app.run(host="0.0.0.0", port=port)
