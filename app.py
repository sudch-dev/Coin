import os
import time
import threading
import hmac
import hashlib
import requests
import json
import traceback
import re
from flask import Flask, render_template, jsonify
from datetime import datetime, timedelta
from pytz import timezone
from collections import deque
from decimal import Decimal, ROUND_DOWN, getcontext

# High precision math for quantization
getcontext().prec = 28

# -------------------- Flask --------------------
app = Flask(__name__)
APP_BASE_URL = os.environ.get("APP_BASE_URL", "https://coin-4k37.onrender.com")

# -------------------- API / Markets --------------------
API_KEY = os.environ.get("API_KEY") or ""
API_SECRET_RAW = os.environ.get("API_SECRET", "")
API_SECRET = API_SECRET_RAW.encode() if isinstance(API_SECRET_RAW, str) else (API_SECRET_RAW or b"")
BASE_URL = "https://api.coindcx.com"

PAIRS = [
    "BTCUSDT", "ETHUSDT", "XRPUSDT", "SHIBUSDT", "SOLUSDT",
    "DOGEUSDT", "ADAUSDT", "AEROUSDT", "BNBUSDT", "LTCUSDT"
]

# Local fallbacks (overridden by markets_details when available)
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

# Populated from markets_details & error learning
# PAIR_META[pair] = {"qty_step","qty_min","qty_prec","px_tick","px_prec","min_price","max_price"}
PAIR_META = {p: {} for p in PAIRS}

# Seed px_prec from your error logs (helps before first learn)
SEED_PX_PREC = {
    "BTCUSDT": 1,
    "LTCUSDT": 2,
    "SOLUSDT": 2,
    "BNBUSDT": 3,
    "AEROUSDT": 3,
    "ADAUSDT": 4,
    "DOGEUSDT": 5
}
for _p, _n in SEED_PX_PREC.items():
    if _p in PAIR_META:
        PAIR_META[_p]["px_prec"] = _n
        PAIR_META[_p]["px_tick"] = 10 ** (-_n)

# -------------------- Engine settings (maker / spread capture) --------------------
CANDLE_INTERVAL = 10
POLL_SEC = 1.0
QUOTE_TTL_SEC = 4
DRIFT_REQUOTE_PCT = 0.0003

FEE_PCT_PER_SIDE = 0.0010
TP_BUFFER_PCT = 0.0006
SPREAD_OFFSET_PCT = None

ATR_WINDOW = 18
ATR_SPREAD_MULT = 1.5
MAX_ATR_PCT = 0.0020
MAX_SLOPE_PCT = 0.0020
EMA_FAST, EMA_SLOW = 5, 20

MAX_PER_PAIR_USDT = 40.0
QUOTE_USDT = 12.0
INVENTORY_USDT_CAP = 120.0
INVENTORY_REDUCE_BIAS = 2.5

FAST_MOVE_PCT = 0.0015
FAST_MOVE_WINDOW_SEC = 5
FREEZE_SEC_AFTER_FAST_MOVE = 6

KILL_SWITCH_PCT = 0.0030
KILL_SWITCH_SELL_FRAC = 0.6
KILL_SWITCH_COOLDOWN_SEC = 20

ORPHAN_MAX_SEC = 900
KEEPALIVE_SEC = 240

# -------------------- Time helpers --------------------
IST = timezone('Asia/Kolkata')
def ist_now(): return datetime.now(IST).strftime('%Y-%m-%d %H:%M:%S')
def ist_date(): return datetime.now(IST).strftime('%Y-%m-%d')
def ist_yesterday(): return (datetime.now(IST) - timedelta(days=1)).strftime('%Y-%m-%d')

# -------------------- State --------------------
tick_logs, candle_logs = {p: [] for p in PAIRS}, {p: [] for p in PAIRS}
scan_log, trade_log = [], []
daily_profit, pair_precision = {}, {}
running = False
status = {"msg": "Idle", "last": ""}
status_epoch = 0
error_message = ""

active_quotes = {p: {"bid": None, "ask": None} for p in PAIRS}
inventory_timer = {}
last_price_state = {p: {"last": None, "ts": 0} for p in PAIRS}
pair_freeze_until = {p: 0 for p in PAIRS}
kill_cooldown_until = {p: 0 for p in PAIRS}
last_buy_fill = {p: {"entry": None, "qty": 0.0, "ts": 0} for p in PAIRS}

# -------------------- Persistent P&L (FIFO) --------------------
PROFIT_STATE_FILE = "profit_state.json"
profit_state = {
    "cumulative_pnl": 0.0,  # USDT
    "daily": {},
    "inventory": {},        # market -> [[qty, price], ...]
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

# -------------------- HTTP + signing helpers --------------------
def hmac_signature(payload: str) -> str:
    return hmac.new(API_SECRET, payload.encode(), hashlib.sha256).hexdigest()

def _log_http_issue(prefix, r):
    try:
        body = r.text[:240] if hasattr(r, "text") else ""
        scan_log.append(f"{ist_now()} | {prefix} HTTP {r.status_code} | {body}")
    except Exception as e:
        scan_log.append(f"{ist_now()} | {prefix} log-fail: {e}")

def _signed_post(url, body_dict):
    """
    CoinDCX private endpoints accept JSON body with:
      headers: X-AUTH-APIKEY / X-AUTH-SIGNATURE
      signature = HMAC_SHA256(API_SECRET, json.dumps(body))
    """
    payload = json.dumps(body_dict, separators=(',', ':'))
    sig = hmac_signature(payload)
    headers = {"X-AUTH-APIKEY": API_KEY, "X-AUTH-SIGNATURE": sig, "Content-Type": "application/json"}
    try:
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
        requests.get(f"{APP_BASE_URL}/ping", timeout=5)
    except Exception:
        pass

# -------------------- Balance & meta --------------------
def get_wallet_balances():
    """
    Returns dict like {"USDT": 123.45, "BTC": 0.01, ...}
    """
    body = {"timestamp": int(time.time() * 1000)}
    res = _signed_post(f"{BASE_URL}/exchange/v1/users/balances", body)
    balances = {}
    try:
        if isinstance(res, list):
            for b in res:
                balances[b["currency"]] = float(b.get("balance", 0.0))
    except Exception as e:
        scan_log.append(f"{ist_now()} | balances parse fail: {e}")
    return balances

def _safe_float(x, default=None):
    try:
        if x is None: return default
        return float(x)
    except:
        return default

def _safe_int(x, default=None):
    try:
        if x is None: return default
        return int(x)
    except:
        return default

def fetch_pair_precisions():
    """
    Populate PAIR_META with tick/step/min/precision using markets_details.
    Falls back to PAIR_RULES if fields are missing.
    """
    try:
        r = requests.get(f"{BASE_URL}/exchange/v1/markets_details", timeout=12)
        if not r.ok:
            _log_http_issue("markets_details", r); return
        data = r.json()
    except Exception as e:
        scan_log.append(f"{ist_now()} | markets_details fail: {e}")
        return

    for item in data:
        m = item.get("pair") or item.get("market") or item.get("coindcx_name")
        if m not in PAIRS:
            continue
        qty_min = _safe_float(item.get("min_quantity"), default=PAIR_RULES.get(m, {}).get("min_qty", 0.0))
        qty_step = _safe_float(item.get("step_size"), default=None)
        px_tick = _safe_float(item.get("tick_size"), default=None)
        px_min  = _safe_float(item.get("min_price"), default=None)
        px_max  = _safe_float(item.get("max_price"), default=None)

        qty_prec = _safe_int(item.get("base_currency_precision"),
                             default=PAIR_RULES.get(m, {}).get("precision", 6))
        px_prec  = _safe_int(item.get("target_currency_precision"), default=None)

        if qty_step is None: qty_step = 10 ** (-qty_prec)
        if px_prec is not None:
            px_tick = 10 ** (-px_prec)
        elif px_tick is None:
            px_prec = 6
            px_tick = 10 ** (-px_prec)

        meta = PAIR_META.setdefault(m, {})
        meta.update({
            "qty_step": qty_step,
            "qty_min": qty_min,
            "qty_prec": qty_prec,
            "px_tick": px_tick,
            "px_prec": px_prec if px_prec is not None else meta.get("px_prec", 6),
            "min_price": px_min,
            "max_price": px_max
        })

# -------------------- Precision helpers --------------------
def _q_dec(step):
    return Decimal(str(step)).normalize()

def quantize_qty(pair, qty, side=None, balance=None):
    meta = PAIR_META.get(pair, {})
    rule = PAIR_RULES.get(pair, {"precision": 6, "min_qty": 0.0001})

    step = meta.get("qty_step", 10 ** (-rule["precision"]))
    min_q = meta.get("qty_min", rule["min_qty"])
    prec  = meta.get("qty_prec", rule["precision"])

    q = Decimal(str(max(0.0, qty)))
    step_dec = _q_dec(step)
    if step_dec > 0:
        q = (q // step_dec) * step_dec  # floor to grid

    if side == "SELL" and balance is not None:
        bal = Decimal(str(max(0.0, balance)))
        if step_dec > 0:
            bal = (bal // step_dec) * step_dec
        q = min(q, bal)

    if q < Decimal(str(min_q)):
        if side == "BUY":
            q = Decimal(str(min_q))
        else:
            q = Decimal("0")

    fmt = Decimal("1e-" + str(prec)) if prec > 0 else Decimal("1")
    try:
        q = q.quantize(fmt, rounding=ROUND_DOWN)
    except:
        pass
    return float(q)

def quantize_price(pair, price, side=None):
    """
    Floor to tick derived from px_prec or px_tick; clamp to min/max if given.
    Never rounds UP.
    """
    meta = PAIR_META.get(pair, {})

    px_prec = meta.get("px_prec", None)
    if px_prec is not None:
        tick = Decimal('1').scaleb(-int(px_prec))
    else:
        px_tick = meta.get("px_tick", 10 ** (-meta.get("px_prec", 6)))
        tick = Decimal(str(px_tick))

    p = Decimal(str(max(0.0, price)))
    if tick > 0:
        p = (p // tick) * tick  # grid floor

    px_min = meta.get("min_price", None)
    px_max = meta.get("max_price", None)
    if px_min is not None: p = max(p, Decimal(str(px_min)))
    if px_max is not None: p = min(p, Decimal(str(px_max)))

    fmt = (Decimal('1').scaleb(-int(px_prec))) if px_prec is not None else tick
    try:
        p = p.quantize(fmt, rounding=ROUND_DOWN)
    except:
        pass
    return float(p)

# -------------------- Learn precision from error & retry --------------------
PRECISION_ERR_RE = re.compile(r'precision\s+should\s+be\s+(\d+)', re.IGNORECASE)

def _learn_quote_precision_from_error(pair: str, message: str) -> bool:
    """
    Parse 'USDT precision should be N' and update PAIR_META for this pair.
    Returns True if updated.
    """
    if not message:
        return False
    m = PRECISION_ERR_RE.search(str(message))
    if not m:
        return False
    n = int(m.group(1))
    meta = PAIR_META.setdefault(pair, {})
    meta['px_prec'] = n
    meta['px_tick'] = 10 ** (-n)
    scan_log.append(f"{ist_now()} | {pair} | learned quote px_prec={n} from error; will re-quantize and retry")
    return True

def _post_order_with_retry(url: str, pair: str, payload_builder, side: str, qty: float, price: float | None, is_limit: bool):
    """
    payload_builder(side, qty, price) -> dict
    Post once; if 'precision should be N' returned, learn N, re-quantize and retry once.
    """
    payload = payload_builder(side, qty, price)
    res = _signed_post(url, payload) or {}
    if isinstance(res, dict) and str(res.get('status')).lower() == 'error':
        msg = res.get('message', '')
        if _learn_quote_precision_from_error(pair, msg):
            q = quantize_qty(pair, qty, side=side)
            p = price
            if is_limit and price is not None:
                p = quantize_price(pair, price, side=side)
            payload = payload_builder(side, q, p)
            res2 = _signed_post(url, payload) or {}
            return res2
    return res

# -------------------- Market data / indicators --------------------
def fetch_all_prices():
    try:
        r = requests.get(f"{BASE_URL}/exchange/ticker", timeout=10)
        if r.ok:
            now = int(time.time())
            return {item.get("market") or item.get("pair"): {"price": float(item["last_price"]), "ts": now}
                    for item in r.json() if (item.get("market") or item.get("pair")) in PAIRS}
        else:
            _log_http_issue("ticker", r)
    except Exception as e:
        scan_log.append(f"{ist_now()} | ticker fail: {e}")
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
            candle["low"]  = min(candle["low"], price)
            candle["close"] = price
            candle["volume"] += 1
    if candle: candles.append(candle)
    candle_logs[pair] = candles[-240:]

def _ema(vals, n):
    if len(vals) < n: return None
    k = 2 / (n + 1)
    ema = sum(vals[:n]) / n
    for v in vals[n:]:
        ema = v * k + ema * (1 - k)
    return ema

def _atr_and_pct(pair, last, window=ATR_WINDOW):
    cs = candle_logs.get(pair) or []
    if len(cs) <= window: return None, None
    trs = []
    prev_close = cs[-window-1]["close"]
    for c in cs[-window:]:
        tr = max(c["high"] - c["low"], abs(c["high"] - prev_close), abs(c["low"] - prev_close))
        trs.append(tr)
        prev_close = c["close"]
    atr = sum(trs)/len(trs) if trs else None
    atr_pct = (atr/last) if (atr and last > 0) else None
    return atr, atr_pct

# -------------------- SMC gate (BOS + premium/discount) --------------------
def _swing_highs_lows(candles, lookback=10):
    if len(candles) < lookback + 2:
        return None, None
    completed = candles[:-1]
    recent = completed[-lookback:]
    high = max(c['high'] for c in recent)
    low  = min(c['low']  for c in recent)
    return high, low

def smc_gate(pair, last_price, lookback=10):
    """
    Minimal SMC:
    - BOS if last completed close > recent swing high (bull) or < swing low (bear)
    - Gate: in bull, only buy in discount (< mid); in bear, only sell in premium (> mid)
    """
    cs = candle_logs.get(pair) or []
    if len(cs) < lookback + 2:
        return True, True, 'neutral'
    completed = cs[:-1]
    last_close = completed[-1]['close']
    swing_hi, swing_lo = _swing_highs_lows(cs, lookback)
    if swing_hi is None:
        return True, True, 'neutral'
    mid = (swing_hi + swing_lo) / 2.0
    bull = last_close > swing_hi
    bear = last_close < swing_lo
    if bull:
        return (last_price <= mid), False, 'bull'
    if bear:
        return False, (last_price >= mid), 'bear'
    return True, True, 'neutral'

# -------------------- Exchange calls (names unchanged) --------------------
def place_limit_order(pair, side, qty, price):
    def _builder(s, q, p):
        q = quantize_qty(pair, q, side=s)
        p = quantize_price(pair, p, side=s)
        return {
            "market": pair, "side": s.lower(), "order_type": "limit_order",
            "price_per_unit": str(p), "total_quantity": str(q),
            "timestamp": int(time.time() * 1000)
        }
    url = f"{BASE_URL}/exchange/v1/orders/create"
    return _post_order_with_retry(url, pair, _builder, side, qty, price, is_limit=True)

def place_order(pair, side, qty):
    def _builder(s, q, _):
        q = quantize_qty(pair, q, side=s)
        return {
            "market": pair, "side": s.lower(), "order_type": "market_order",
            "total_quantity": str(q), "timestamp": int(time.time() * 1000)
        }
    url = f"{BASE_URL}/exchange/v1/orders/create"
    return _post_order_with_retry(url, pair, _builder, side, qty, None, is_limit=False)

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
            ts_ms = int(time.time() * 1000)
        apply_fill_update(market, side, avg_px, filled, ts_ms, order_id)
    return filled, avg_px

# -------------------- Maker quoting helpers --------------------
def _effective_half_spread_pct(adapt_with_atr_pct=None):
    base = (2 * FEE_PCT_PER_SIDE + TP_BUFFER_PCT) / 2.0
    if adapt_with_atr_pct: base = max(base, adapt_with_atr_pct * ATR_SPREAD_MULT)
    return max(base, (SPREAD_OFFSET_PCT or 0.0))

def _quote_prices(last, atr_pct=None):
    off = _effective_half_spread_pct(atr_pct)
    bid = round(last * (1.0 - off), 8)
    ask = round(last * (1.0 + off), 8)
    return bid, ask

def _qty_for_pair(pair, price, usdt_avail, coin_avail, side, trend_skew):
    raw_q = max(1e-12, QUOTE_USDT) / max(price, 1e-9)
    raw_q *= trend_skew
    raw_q = min(raw_q, MAX_PER_PAIR_USDT / max(price, 1e-9))
    if side == "BUY":
        raw_q = min(raw_q, usdt_avail / max(price, 1e-9))
        q = quantize_qty(pair, raw_q, side="BUY", balance=usdt_avail / max(price, 1e-9))
    else:
        raw_q = min(raw_q, coin_avail)
        q = quantize_qty(pair, raw_q, side="SELL", balance=coin_avail)
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

def _cancel_quote(pair, side_key):
    q = active_quotes.get(pair, {}).get(side_key)
    if not q: return
    oid = q.get("id")
    if oid:
        cancel_order(order_id=oid)
        scan_log.append(f"{ist_now()} | {pair} | cancel {side_key} quote {oid}")
    active_quotes[pair][side_key] = None

def cancel_all_quotes(pair):
    _cancel_quote(pair, "bid"); _cancel_quote(pair, "ask")

def _place_quote(pair, side_word, price, qty):
    price = quantize_price(pair, price, side=side_word)
    qty   = quantize_qty(pair, qty, side=side_word)
    if qty <= 0:
        scan_log.append(f"{ist_now()} | {pair} | skip quote {side_word}: qty<=0 after quantize")
        return None
    res = place_limit_order(pair, side_word, qty, price)
    oid = _extract_order_id(res)
    side_key = "bid" if side_word == "BUY" else "ask"
    active_quotes[pair][side_key] = {"id": oid, "px": price, "qty": qty, "ts": int(time.time())}
    scan_log.append(f"{ist_now()} | {pair} | quote {side_word} {qty} @ {price} | id={oid} | res={res}")
    return oid

def _check_quote_fill(pair, side_key, last_price):
    q = active_quotes.get(pair, {}).get(side_key)
    if not q or not q.get("id"): return False
    st = get_order_status(order_id=q["id"])
    rem = _fnum(st.get("remaining_quantity", st.get("remaining_qty", st.get("leaves_qty", 0))))
    status_txt = (st.get("status") or "").lower()
    filled = (rem == 0) or ("filled" in status_txt and "part" not in status_txt)
    if filled:
        order_side = "BUY" if side_key == "bid" else "SELL"
        filled_qty, avg_px = _record_fill_from_status(pair, order_side, st, q["id"])
        active_quotes[pair][side_key] = None
        scan_log.append(f"{ist_now()} | {pair} | FILL {order_side} {filled_qty} @ {avg_px} | st={st}")
        if order_side == "BUY" and filled_qty > 0 and avg_px > 0:
            last_buy_fill[pair] = {"entry": float(avg_px), "qty": float(filled_qty), "ts": int(time.time())}
        return True
    return False

def _manage_orphan_inventory(pair, now_ts, prices, balances):
    q_units = _net_inventory_units(pair)
    if q_units <= 0:
        inventory_timer.pop(pair, None); return
    if pair not in inventory_timer:
        inventory_timer[pair] = now_ts; return
    if now_ts - inventory_timer[pair] < ORPHAN_MAX_SEC:
        return
    coin = pair[:-4]
    coin_bal = balances.get(coin, 0.0)
    qty = min(0.3 * q_units, coin_bal)
    qty = quantize_qty(pair, qty, side="SELL", balance=coin_bal)
    if qty <= 0: return
    res = place_order(pair, "SELL", qty)
    oid = _extract_order_id(res)
    if oid:
        st = get_order_status(order_id=oid)
        _record_fill_from_status(pair, "SELL", st, oid)
    scan_log.append(f"{ist_now()} | {pair} | ORPHAN TIMEOUT market SELL {qty} | res={res}")
    inventory_timer[pair] = now_ts

# -------------------- Main loop (autostart-safe) --------------------
last_keepalive = 0
_autostart_lock = threading.Lock()

def scan_loop():
    global running, error_message, status_epoch, last_keepalive
    scan_log.clear()
    running = True

    while running:
        now_real = time.time()
        if now_real - last_keepalive >= KEEPALIVE_SEC:
            _keepalive_ping()
            last_keepalive = now_real

        prices = fetch_all_prices()
        now_ts = int(time.time())
        balances = get_wallet_balances()

        # build candles
        for pair in PAIRS:
            if pair in prices:
                px = prices[pair]["price"]
                tick_logs[pair].append((now_ts, px))
                if len(tick_logs[pair]) > 4000:
                    tick_logs[pair] = tick_logs[pair][-4000:]
                aggregate_candles(pair, CANDLE_INTERVAL)

        gross_usdt = _gross_inventory_usdt(prices)
        if gross_usdt > INVENTORY_USDT_CAP:
            scan_log.append(f"{ist_now()} | RISK | Gross inventory {round(gross_usdt,2)} > cap {INVENTORY_USDT_CAP} — pause new bids")

        for pair in PAIRS:
            if pair not in prices: continue
            last = prices[pair]["price"]

            # fast-move freeze
            prev = last_price_state[pair]["last"]; pts = last_price_state[pair]["ts"]
            last_price_state[pair] = {"last": last, "ts": now_ts}
            if prev and (now_ts - pts) <= FAST_MOVE_WINDOW_SEC:
                move_pct = abs(last - prev) / max(prev, 1e-9)
                if move_pct >= FAST_MOVE_PCT:
                    cancel_all_quotes(pair)
                    pair_freeze_until[pair] = now_ts + FREEZE_SEC_AFTER_FAST_MOVE
                    scan_log.append(f"{ist_now()} | {pair} | FAST MOVE {round(move_pct*100,3)}% — freeze")
                    continue
            if now_ts < pair_freeze_until[pair]:
                for side_key in ("bid", "ask"): _check_quote_fill(pair, side_key, last)
                continue

            # indicators
            closes = [c["close"] for c in (candle_logs.get(pair) or [])[:-1]]
            ema_fast = _ema(closes[-(EMA_SLOW+EMA_FAST):] + [last], EMA_FAST) if len(closes) >= EMA_SLOW else None
            ema_slow = _ema(closes[-(EMA_SLOW+EMA_FAST):] + [last], EMA_SLOW) if len(closes) >= EMA_SLOW else None
            bullish = (ema_fast is not None and ema_slow is not None and ema_fast >= ema_slow)
            slope_pct = None
            if ema_fast is not None and ema_slow is not None and last > 0:
                slope_pct = abs(ema_fast - ema_slow) / last

            atr_abs, atr_pct = _atr_and_pct(pair, last, ATR_WINDOW)

            bid_px, ask_px = _quote_prices(last, atr_pct)
            bid_px = quantize_price(pair, min(bid_px, last * (1 - 1e-6)), side="BUY")
            ask_px = quantize_price(pair, max(ask_px, last * (1 + 1e-6)), side="SELL")

            usdt = balances.get("USDT", 0.0)
            coin = pair[:-4]
            coin_bal = balances.get(coin, 0.0)
            net_units = _net_inventory_units(pair)

            reduce_bias_bid = INVENTORY_REDUCE_BIAS if net_units < 0 else 1.0
            reduce_bias_ask = INVENTORY_REDUCE_BIAS if net_units > 0 else 1.0
            trend_bias_bid = 1.2 if bullish else 0.9
            trend_bias_ask = 1.2 if not bullish else 0.9

            allow_bid, allow_ask = True, True
            if gross_usdt > INVENTORY_USDT_CAP: allow_bid = False
            if atr_pct is not None and atr_pct > MAX_ATR_PCT: allow_bid = False
            if slope_pct is not None and slope_pct > MAX_SLOPE_PCT:
                if bullish: allow_ask = False
                else:       allow_bid = False

            # --- SMC filter (final gate) ---
            smc_bid, smc_ask, smc_regime = smc_gate(pair, last, lookback=10)
            allow_bid = allow_bid and smc_bid
            allow_ask = allow_ask and smc_ask
            if smc_regime != 'neutral':
                scan_log.append(f"{ist_now()} | {pair} | SMC {smc_regime.upper()} | allow_bid={smc_bid} allow_ask={smc_ask}")

            # cancel stale quotes
            for side_key, q in list(active_quotes[pair].items()):
                if not q: continue
                age = now_ts - int(q.get("ts", now_ts))
                px  = q.get("px", last)
                drift = abs(last - px) / max(px, 1e-9)
                if age >= QUOTE_TTL_SEC or drift >= DRIFT_REQUOTE_PCT:
                    _cancel_quote(pair, side_key)

            # check fills
            for side_key in ("bid", "ask"):
                _check_quote_fill(pair, side_key, last)

            # kill-switch after BUY fill
            lb = last_buy_fill.get(pair, {})
            if lb and lb.get("entry") and now_ts >= kill_cooldown_until[pair]:
                entry = float(lb["entry"]); filled_qty = float(lb["qty"])
                if last <= entry * (1.0 - KILL_SWITCH_PCT):
                    qty_to_sell = min(filled_qty * KILL_SWITCH_SELL_FRAC, coin_bal)
                    qty_to_sell = quantize_qty(pair, qty_to_sell, side="SELL", balance=coin_bal)
                    if qty_to_sell > 0:
                        _cancel_quote(pair, "bid")
                        res = place_order(pair, "SELL", qty_to_sell)
                        oid = _extract_order_id(res)
                        if oid:
                            st = get_order_status(order_id=oid)
                            _record_fill_from_status(pair, "SELL", st, oid)
                        scan_log.append(f"{ist_now()} | {pair} | KILL-SWITCH: SOLD {qty_to_sell}")
                        kill_cooldown_until[pair] = now_ts + KILL_SWITCH_COOLDOWN_SEC
                        last_buy_fill[pair]["qty"] = max(0.0, filled_qty - qty_to_sell)

            # size and place quotes
            qty_bid = _qty_for_pair(pair, bid_px, usdt, coin_bal, "BUY",  reduce_bias_bid * trend_bias_bid) if allow_bid else 0.0
            qty_ask = _qty_for_pair(pair, ask_px, usdt, coin_bal, "SELL", reduce_bias_ask * trend_bias_ask) if allow_ask else 0.0
            if qty_bid > 0 and not active_quotes[pair]["bid"]:
                _place_quote(pair, "BUY",  bid_px, qty_bid)
            if qty_ask > 0 and coin_bal > 0 and not active_quotes[pair]["ask"]:
                _place_quote(pair, "SELL", ask_px, qty_ask)

            _manage_orphan_inventory(pair, now_ts, prices, balances)

        status["msg"], status["last"] = "Running", ist_now()
        status_epoch = int(time.time())
        print(f"[{ist_now()}] Loop active — quoting…")
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
        if v: visible_quotes[p] = v

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

# -------------------- Autostart --------------------
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
        time.sleep(1.0)
        _start_loop_once()
    except Exception as e:
        print("boot kick failed:", e)

if os.environ.get("AUTOSTART", "1") == "1":
    threading.Thread(target=_boot_kick, daemon=True).start()

if __name__ == "__main__":
    load_profit_state()
    fetch_pair_precisions()      # loads tick/step/min/precision
    _start_loop_once()
    port = int(os.environ.get("PORT", "10000"))
    app.run(host="0.0.0.0", port=port)
