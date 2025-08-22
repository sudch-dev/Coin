import os
import time
import threading
import hmac
import hashlib
import requests
import json
import traceback
import random
from flask import Flask, render_template, jsonify, request
from datetime import datetime, timedelta
from pytz import timezone
from collections import deque

# -------------------- Flask --------------------
app = Flask(__name__)

APP_BASE_URL    = os.environ.get("APP_BASE_URL", "https://coin-4k37.onrender.com")
KEEPALIVE_TOKEN = os.environ.get("KEEPALIVE_TOKEN", "")

# -------------------- Exchange creds --------------------
API_KEY = os.environ.get("API_KEY")
API_SECRET_RAW = os.environ.get("API_SECRET", "")
API_SECRET = API_SECRET_RAW.encode() if isinstance(API_SECRET_RAW, str) else API_SECRET_RAW
BASE_URL = "https://api.coindcx.com"

# -------------------- Markets --------------------
PAIRS = [
    "BTCUSDT", "ETHUSDT", "XRPUSDT", "SHIBUSDT", "SOLUSDT",
    "DOGEUSDT", "ADAUSDT", "AEROUSDT", "BNBUSDT"
]

# Local fallbacks; refreshed from live rules
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

# -------------------- Strategy knobs --------------------
CANDLE_INTERVAL    = 5     # seconds (5s candles)
POLL_SEC           = 1.0
BUY_FRACTION_USDT  = 0.30  # cap notional per trade (also used by risk-based sizing cap)
SELL_ALL_COIN      = True

# Fees and safety headroom
FEE_PCT_PER_SIDE = 0.0010
BUY_HEADROOM     = 1.0005  # 5bp cushion so rounding fits

# Keepalive
KEEPALIVE_SEC = 240

# Rules refresh cadence
RULES_REFRESH_SEC = 1800
_last_rules_refresh = 0

# ===== Fast Momentum Breakout knobs =====
HTF_EMA_LEN        = 12      # trend filter on closes (~1 min EMA over 5s bars)
ATR_N              = 36      # ATR window (~3 min)
BRK_LOOKBACK       = 36      # breakout lookback (~3 min high/low)
MIN_ATR_PCT        = 0.0008  # require ATR >= 0.08% of price (avoid chop)
RISK_PER_TRADE     = 0.01    # risk 1% of free USDT per trade
ATR_SL_MULT        = 0.8     # initial stop = entry - 0.8*ATR
ATR_TP_MULT        = 1.6     # take profit  = entry + 1.6*ATR (≈2R)
TRAIL_MULT         = 0.8     # trailing stop = highest - 0.8*ATR
COOLDOWN_SEC       = 30      # cooldown after exit/failure
MAX_CONCURRENT_POS = 1       # throttle: at most N open pairs

# -------------------- Time helpers --------------------
IST = timezone('Asia/Kolkata')
def ist_now(): return datetime.now(IST).strftime('%Y-%m-%d %H:%M:%S')
def ist_date(): return datetime.now(IST).strftime('%Y-%m-%d')
def ist_yesterday(): return (datetime.now(IST) - timedelta(days=1)).strftime('%Y-%m-%d')

# -------------------- State --------------------
tick_logs     = {p: [] for p in PAIRS}   # (ts, price) ticks
candle_logs   = {p: [] for p in PAIRS}   # {open,high,low,close,start}
scan_log      = []                       # decisions + signals
trade_log     = []                       # execution + pnl lines
running       = False
status        = {"msg": "Idle", "last": ""}
status_epoch  = 0
error_message = ""
last_keepalive = 0

# Position tracking per pair
positions = {
    # pair: {"qty": float, "entry": float, "ts": int, "stop": float, "trail": float, "highest": float, "atr": float}
}
cooldown_until = {}  # pair -> epoch until entries are blocked

# P&L persistence
PROFIT_STATE_FILE = "profit_state.json"
profit_state = {
    "cumulative_pnl": 0.0,
    "daily": {},
    "processed_orders": []
}

# Balance cache
BAL_TTL_SEC = 15
_bal_cache = {}
_bal_cache_ts = 0

# ---- Raw I/O logging config ----
IO_ENABLED   = os.environ.get("IO_ENABLED", "1") == "1"
IO_MAX_LINES = int(os.environ.get("IO_MAX_LINES", "500"))
IO_MAX_FIELD = int(os.environ.get("IO_MAX_FIELD", "2000"))
IO_LOG_FILE  = os.environ.get("IO_LOG_FILE", "io_log.txt")
io_log = deque(maxlen=IO_MAX_LINES)

# -------------------- Logging helpers --------------------
def _tlog(msg):
    line = f"{ist_now()} | {msg}"
    trade_log.append(line)
    scan_log.append(line)

def _truncate(s, n):
    try:
        s = s if isinstance(s, str) else json.dumps(s)
    except Exception:
        s = str(s)
    if s is None:
        return ""
    return s if len(s) <= n else s[:n] + f"...(truncated {len(s)-n} chars)"

def _mask_headers(h):
    if not isinstance(h, dict):
        return {}
    masked = {}
    for k, v in h.items():
        lk = k.lower()
        if lk in ("x-auth-signature", "x-auth-apikey", "authorization"):
            masked[k] = "***"
        else:
            masked[k] = v
    return masked

def log_io(direction, url, headers=None, payload=None, status=None, response=None):
    if not IO_ENABLED:
        return
    rec = {
        "ts": ist_now(),
        "dir": direction,
        "url": url,
        "status": status,
        "headers": _mask_headers(headers or {}),
        "payload": _truncate(payload, IO_MAX_FIELD) if payload is not None else None,
        "response": _truncate(response, IO_MAX_FIELD) if response is not None else None,
    }
    line = f"{rec['ts']} | {rec['dir']} | {rec['url']} | status={rec['status']} | headers={rec['headers']} | payload={rec['payload']} | response={rec['response']}"
    io_log.append(line)
    try:
        if IO_LOG_FILE:
            with open(IO_LOG_FILE, "a") as f:
                f.write(line + "\n")
    except Exception:
        pass

# -------------------- P&L helpers --------------------
def load_profit_state():
    try:
        with open(PROFIT_STATE_FILE, "r") as f:
            data = json.load(f)
        profit_state["cumulative_pnl"] = float(data.get("cumulative_pnl", 0.0))
        profit_state["daily"] = dict(data.get("daily", {}))
        profit_state["processed_orders"] = list(data.get("processed_orders", []))
    except:
        pass

def save_profit_state():
    tmp = {
        "cumulative_pnl": round(profit_state.get("cumulative_pnl", 0.0), 6),
        "daily": {k: round(v, 6) for k, v in profit_state.get("daily", {}).items()},
        "processed_orders": profit_state.get("processed_orders", [])
    }
    try:
        with open(PROFIT_STATE_FILE, "w") as f:
            json.dump(tmp, f)
    except:
        pass

def record_realized_pnl(pnl):
    pnl = float(pnl or 0.0)
    profit_state["cumulative_pnl"] = float(profit_state.get("cumulative_pnl", 0.0) + pnl)
    dkey = ist_date()
    profit_state["daily"][dkey] = float(profit_state["daily"].get(dkey, 0.0) + pnl)
    save_profit_state()

def compute_realized_pnl_today():
    return round(profit_state["daily"].get(ist_date(), 0.0), 6)

# -------------------- HTTP session + retry helpers --------------------
SESSION = requests.Session()
SESSION.headers.update({"User-Agent": "CoinDCXBot/1.0 (+health; non-browser)"})

def _http_post_with_retry(url, headers, payload, timeout=12, retries=3, backoff_base=0.5):
    for i in range(retries):
        try:
            log_io("REQ", url, headers=headers, payload=payload)
            r = SESSION.post(url, headers=headers, data=payload, timeout=timeout)
            body = r.text if hasattr(r, "text") else ""
            log_io("RES", url, status=r.status_code, response=body)

            if r.status_code in (520, 502, 503, 504):
                try:
                    cf = f" cf-ray={r.headers.get('CF-RAY','?')} cf-cache={r.headers.get('CF-Cache-Status','?')}"
                except Exception:
                    cf = ""
                _log_http_issue(f"POST {url}{cf}", r)
                if i < retries - 1:
                    time.sleep(backoff_base * (2 ** i) + random.random() * 0.25)
                    continue
            return r
        except Exception as e:
            scan_log.append(f"{ist_now()} | POST fail {url} | {e.__class__.__name__}: {e}")
            if i < retries - 1:
                time.sleep(backoff_base * (2 ** i) + random.random() * 0.25)
                continue
            return None

def _http_get_with_retry(url, timeout=12, retries=3, backoff_base=0.5):
    for i in range(retries):
        try:
            log_io("REQ", url)
            r = SESSION.get(url, timeout=timeout)
            body = r.text if hasattr(r, "text") else ""
            log_io("RES", url, status=r.status_code, response=body)

            if r.status_code in (520, 502, 503, 504):
                _log_http_issue(f"GET {url}", r)
                if i < retries - 1:
                    time.sleep(backoff_base * (2 ** i) + random.random() * 0.25)
                    continue
            return r
        except Exception as e:
            scan_log.append(f"{ist_now()} | GET fail {url} | {e.__class__.__name__}: {e}")
            if i < retries - 1:
                time.sleep(backoff_base * (2 ** i) + random.random() * 0.25)
                continue
            return None

# -------------------- Signed HTTP helpers --------------------
def hmac_signature(payload):
    return hmac.new(API_SECRET, payload.encode(), hashlib.sha256).hexdigest()

def _log_http_issue(prefix, r):
    try:
        body = r.text[:240] if hasattr(r, "text") else ""
        cf = f" cf-ray={r.headers.get('CF-RAY','?')} cf-cache={r.headers.get('CF-Cache-Status','?')}"
        _tlog(f"{prefix} HTTP {r.status_code}{cf} | {body}")
    except Exception as e:
        _tlog(f"{prefix} log-fail: {e}")

def _signed_post(url, body):
    payload = json.dumps(body, separators=(',', ':'))
    sig = hmac_signature(payload)
    headers = {"X-AUTH-APIKEY": API_KEY or "", "X-AUTH-SIGNATURE": sig, "Content-Type": "application/json"}
    try:
        r = _http_post_with_retry(url, headers, payload, timeout=12, retries=3)
        if r is None:
            _tlog(f"{url} HTTP fail (no response after retries)")
            return {}
        if not r.ok:
            _log_http_issue(f"POST {url}", r)
            return {}
        if r.headers.get("content-type", "").startswith("application/json"):
            return r.json()
        _tlog(f"POST {url} unexpected content-type: {r.headers.get('content-type','')[:60]}")
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
        headers = {}
        if KEEPALIVE_TOKEN:
            url = f"{url}?t={KEEPALIVE_TOKEN}"
            headers["X-Keepalive-Token"] = KEEPALIVE_TOKEN
        SESSION.get(url, headers=headers, timeout=5)
    except Exception:
        pass

# -------------------- Exchange helpers --------------------
def fetch_pair_precisions():
    try:
        r = _http_get_with_retry(f"{BASE_URL}/exchange/v1/markets_details", timeout=12, retries=3)
        if not r or not r.ok:
            if r:
                _log_http_issue("markets_details", r)
            return
        data = r.json()
        by_pair = {}
        for item in data:
            p = item.get("pair") or item.get("market") or item.get("coindcx_name")
            if not p:
                continue
            by_pair[p] = {
                "price_precision": int(item.get("target_currency_precision", 6)),
                "qty_precision":   int(item.get("base_currency_precision", 6)),
                "min_qty":         float(item.get("min_quantity", 0.0) or 0.0),
                "min_notional":    float(item.get("min_notional", 0.0) or 0.0)
            }
        for p in PAIRS:
            if p in by_pair:
                PAIR_RULES[p] = by_pair[p]
        scan_log.append(f"{ist_now()} | market rules refreshed")
    except Exception as e:
        scan_log.append(f"{ist_now()} | markets_details fail: {e}")

def get_wallet_balances():
    """
    Cached balances with TTL. On failure, returns last known cache.
    """
    global _bal_cache, _bal_cache_ts
    now = time.time()

    if _bal_cache and (now - _bal_cache_ts) < BAL_TTL_SEC:
        return dict(_bal_cache)

    payload = json.dumps({"timestamp": int(now * 1000)})
    sig = hmac_signature(payload)
    headers = {"X-AUTH-APIKEY": API_KEY or "", "X-AUTH-SIGNATURE": sig, "Content-Type": "application/json"}
    balances = {}
    try:
        r = _http_post_with_retry(f"{BASE_URL}/exchange/v1/users/balances", headers, payload, timeout=10, retries=3)
        if r and r.ok:
            data = r.json()
            for b in data:
                balances[b['currency']] = float(b['balance'])
            _bal_cache = balances
            _bal_cache_ts = now
            return balances
        else:
            if r is not None:
                _log_http_issue("balances", r)
    except Exception as e:
        scan_log.append(f"{ist_now()} | balances fail: {e}")

    if _bal_cache:
        age = int(now - _bal_cache_ts)
        _tlog(f"balances degraded: serving cached (age={age}s)")
        return dict(_bal_cache)

    _tlog("balances unavailable (no cache)")
    return {}

def balances_age_sec():
    return int(time.time() - _bal_cache_ts) if _bal_cache_ts else None

def fetch_all_prices():
    try:
        r = _http_get_with_retry(f"{BASE_URL}/exchange/ticker", timeout=10, retries=3)
        if r and r.ok:
            now = int(time.time())
            return {item["market"]: {"price": float(item["last_price"]), "ts": now}
                    for item in r.json() if item.get("market") in PAIRS}
        else:
            if r:
                _log_http_issue("ticker", r)
    except Exception as e:
        scan_log.append(f"{ist_now()} | ticker fail: {e}")
    return {}

# -------------------- Precision helpers --------------------
def _rules(pair): return PAIR_RULES.get(pair, {})
def _min_qty(pair): return float(_rules(pair).get("min_qty", 0.0) or 0.0)
def _qty_prec(pair): return int(_rules(pair).get("qty_precision", 6))

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

def _fee_multiplier(side):
    return 1.0 + FEE_PCT_PER_SIDE if side.upper() == "BUY" else 1.0 - FEE_PCT_PER_SIDE

def normalize_qty_for_buy(pair, price, usdt_avail):
    """
    Turn a USDT notional into a precision/min-qty safe quantity that fits fees.
    """
    if price <= 0:
        return 0.0
    denom = max(price * _fee_multiplier("BUY") * BUY_HEADROOM, 1e-12)
    q = usdt_avail / denom
    q = fmt_qty(pair, q)
    step = _qty_step(pair)
    while q >= _min_qty(pair) and (price * q * _fee_multiplier("BUY") * BUY_HEADROOM) > usdt_avail + 1e-12:
        q = fmt_qty(pair, max(0.0, q - step))
    if q < _min_qty(pair): return 0.0
    return q

def place_market(pair, side, qty):
    payload = {
        "market": pair,
        "side": side.lower(),
        "order_type": "market_order",
        "total_quantity": f"{fmt_qty(pair, qty)}",
        "timestamp": int(time.time() * 1000)
    }
    _tlog(f"{pair} | SUBMIT {side} qty={payload['total_quantity']} @ MKT")
    res = _signed_post(f"{BASE_URL}/exchange/v1/orders/create", payload) or {}
    _tlog(f"{pair} | SUBMIT RESP {side} => {res}")
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
    for k in ("id", "order_id", "orderId", "client_order_id", "clientOrderId"):
        if res.get(k):
            return str(res[k])
    try:
        if isinstance(res.get("orders"), list) and res["orders"]:
            o = res["orders"][0]
            for k in ("id", "order_id", "orderId", "client_order_id", "clientOrderId"):
                if o.get(k):
                    return str(o[k])
    except:
        pass
    d = res.get("data") or {}
    for k in ("id", "order_id", "orderId", "client_order_id", "clientOrderId"):
        if d.get(k):
            return str(d[k])
    return None

def _filled_avg_from_status(st):
    try:
        total_q  = float(st.get("total_quantity", st.get("quantity", 0)))
        remain_q = float(st.get("remaining_quantity", st.get("remaining_qty", 0)))
        exec_q   = float(st.get("executed_quantity", st.get("filled_qty", 0)))
        filled   = exec_q if exec_q > 0 else max(0.0, total_q - remain_q)
        avg_px   = float(st.get("avg_price", st.get("average_price",
                            st.get("avg_execution_price", st.get("price", 0)))))
        return filled, avg_px
    except:
        return 0.0, 0.0

# -------------------- Candle & indicators --------------------
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
    candle_logs[pair] = candles[-300:]  # ~25 minutes of 5s bars

def _ema_series(vals, n):
    if len(vals) == 0:
        return []
    k = 2 / (n + 1)
    ema_vals = []
    ema = vals[0]
    for v in vals:
        ema = v * k + ema * (1 - k)
        ema_vals.append(ema)
    return ema_vals

def _macd_series(closes):
    MACD_FAST = 12
    MACD_SLOW = 26
    MACD_SIGNAL = 9
    if len(closes) < MACD_SLOW + MACD_SIGNAL:
        return [], [], []
    ema_fast = _ema_series(closes, MACD_FAST)
    ema_slow = _ema_series(closes, MACD_SLOW)
    macd = [f - s for f, s in zip(ema_fast, ema_slow)]
    signal = _ema_series(macd, MACD_SIGNAL)
    hist = [m - s for m, s in zip(macd[-len(signal):], signal)]
    macd_al = macd[-len(signal):]
    return macd_al, signal, hist

def _swing_highs(vals, look=5):
    idx = []
    for i in range(look, len(vals)-look):
        if all(vals[i] >= vals[i-k] for k in range(1, look+1)) and all(vals[i] >= vals[i+k] for k in range(1, look+1)):
            idx.append(i)
    return idx

def _swing_lows(vals, look=5):
    idx = []
    for i in range(look, len(vals)-look):
        if all(vals[i] <= vals[i-k] for k in range(1, look+1)) and all(vals[i] <= vals[i+k] for k in range(1, look+1)):
            idx.append(i)
    return idx

def _macd_convergence_divergence(closes, macd_line):
    if len(closes) < 30 or len(macd_line) < 30:
        return "insufficient"
    N = 20
    p_seg = closes[-N:]
    m_seg = macd_line[-N:]
    p_slope = p_seg[-1] - p_seg[0]
    m_slope = m_seg[-1] - m_seg[0]
    slope_note = f"slope price={'↑' if p_slope>0 else '↓' if p_slope<0 else '→'}, macd={'↑' if m_slope>0 else '↓' if m_slope<0 else '→'}"
    look = 3
    ph = _swing_highs(closes, look); mh = _swing_highs(macd_line, look)
    pl = _swing_lows(closes, look);  ml = _swing_lows(macd_line, look)
    bear_div = bull_div = False
    if len(ph) >= 2 and len(mh) >= 2:
        p1, p2 = closes[ph[-2]], closes[ph[-1]]
        m1, m2 = macd_line[mh[-2]], macd_line[mh[-1]]
        if p2 > p1 and m2 <= m1: bear_div = True
    if len(pl) >= 2 and len(ml) >= 2:
        p1, p2 = closes[pl[-2]], closes[pl[-1]]
        m1, m2 = macd_line[ml[-2]], macd_line[ml[-1]]
        if p2 < p1 and m2 >= m1: bull_div = True
    if bear_div: return f"Bearish divergence ({slope_note})"
    if bull_div: return f"Bullish divergence ({slope_note})"
    agree = (p_slope >= 0 and m_slope >= 0) or (p_slope <= 0 and m_slope <= 0)
    return f"Convergence ({'agree' if agree else 'mixed'}; {slope_note})"

# ---- ATR, breakout, risk sizing helpers ----
def _atr_from_candles(cs, n=ATR_N):
    """Wilder-style ATR approx on 5s candles."""
    if len(cs) < n + 2:
        return 0.0
    trs = []
    prev_close = cs[-(n+1)]["close"]
    for c in cs[-n:]:
        high, low, close = c["high"], c["low"], c["close"]
        tr = max(high - low, abs(high - prev_close), abs(low - prev_close))
        trs.append(tr)
        prev_close = close
    # simple EMA of TR for speed
    k = 2 / (n + 1)
    atr = trs[0]
    for tr in trs:
        atr = tr * k + atr * (1 - k)
    return float(atr)

def _recent_hl(cs, lb=BRK_LOOKBACK):
    if len(cs) < lb:
        return None, None
    highs = [c["high"] for c in cs[-lb:]]
    lows  = [c["low"]  for c in cs[-lb:]]
    return max(highs), min(lows)

def _size_by_risk(usdt_free, price, atr):
    """
    Risk per trade = RISK_PER_TRADE * usdt_free.
    Stop distance = ATR_SL_MULT * atr  -> qty_risk = risk / stop_distance.
    Clamp by BUY_FRACTION_USDT notional cap.
    """
    if atr <= 0 or price <= 0:
        return 0.0
    stop_dist = ATR_SL_MULT * atr
    if stop_dist <= 0:
        return 0.0
    risk_amt = max(0.0, RISK_PER_TRADE * usdt_free)
    if risk_amt <= 0:
        return 0.0
    qty_risk = risk_amt / stop_dist                 # quantity in coins
    notional_cap = BUY_FRACTION_USDT * usdt_free
    qty_cap = (notional_cap / price) if notional_cap > 0 else qty_risk
    qty = min(qty_risk, qty_cap)
    return max(0.0, qty)

def _pnl_after_fees(entry, exit_px, qty):
    buy_cost  = entry * (1 + FEE_PCT_PER_SIDE) * qty
    sell_recv = exit_px * (1 - FEE_PCT_PER_SIDE) * qty
    return sell_recv - buy_cost

# -------------------- Strategy core --------------------
def strategy_scan():
    prices = fetch_all_prices()
    now_ts = int(time.time())

    # build ticks and candles
    for pair in PAIRS:
        if pair in prices:
            px = prices[pair]["price"]
            tick_logs[pair].append((now_ts, px))
            if len(tick_logs[pair]) > 6000:
                tick_logs[pair] = tick_logs[pair][-6000:]
            aggregate_candles(pair, CANDLE_INTERVAL)

    balances = get_wallet_balances()
    usdt_free = balances.get("USDT", 0.0)
    bal_age = balances_age_sec()
    buys_allowed = (bal_age is not None and bal_age <= BAL_TTL_SEC * 3)
    if not buys_allowed:
        scan_log.append(f"{ist_now()} | NOTE: buys paused (balances stale age={bal_age}s)")

    # throttle concurrent positions
    open_count = sum(1 for v in positions.values() if v.get("qty", 0.0) > 0)
    can_open_more = open_count < MAX_CONCURRENT_POS

    for pair in PAIRS:
        cs = candle_logs.get(pair) or []
        if len(cs) < max(HTF_EMA_LEN, ATR_N, BRK_LOOKBACK) + 3:
            continue

        closes = [c["close"] for c in cs]
        last   = closes[-1]
        high_b, low_b = _recent_hl(cs, BRK_LOOKBACK)
        atr    = _atr_from_candles(cs, ATR_N)
        atr_pct = (atr / last) if last > 0 else 0.0

        # trend & momentum filters
        ema_htf = _ema_series(closes, HTF_EMA_LEN)[-1] if len(closes) >= HTF_EMA_LEN else last
        macd_line, macd_signal, _ = _macd_series(closes)
        if not macd_line or not macd_signal:
            continue
        m_now, s_now = macd_line[-1], macd_signal[-1]

        # log snapshot for transparency
        scan_log.append(
            f"{ist_now()} | {pair} | last={last} ATR={round(atr,6)} ({round(100*atr_pct,3)}%) "
            f"| EMA{HTF_EMA_LEN}={round(ema_htf,6)} | MACD={round(m_now,6)} SIG={round(s_now,6)} "
            f"| HL[{BRK_LOOKBACK}] hi={high_b} lo={low_b}"
        )

        # ENTRY (LONG)
        have_pos = pair in positions and positions[pair].get("qty", 0.0) > 0.0
        cooled   = now_ts >= int(cooldown_until.get(pair, 0))
        breakout_up   = (high_b is not None) and (last > high_b)
        volatility_ok = atr_pct >= MIN_ATR_PCT
        trend_ok      = last > ema_htf
        momentum_ok   = m_now > s_now

        if (not have_pos) and cooled and buys_allowed and can_open_more and breakout_up and volatility_ok and trend_ok and momentum_ok:
            # risk-based size -> quantity; normalize via notional for fees/precision
            raw_qty = _size_by_risk(usdt_free, last, atr)
            target_notional = raw_qty * last
            qty = normalize_qty_for_buy(pair, last, target_notional)
            scan_log.append(f"{ist_now()} | {pair} | ENTRY-CALC raw_qty={raw_qty} norm_qty={qty} usdt_free={usdt_free}")
            if qty > 0:
                res = place_market(pair, "BUY", qty)
                oid = _extract_order_id(res)
                if oid:
                    st = get_order_status(order_id=oid)
                    filled, avg_px = _filled_avg_from_status(st)
                    if filled > 0 and avg_px > 0:
                        stop = max(0.0, avg_px - ATR_SL_MULT * atr)
                        positions[pair] = {
                            "qty": filled, "entry": avg_px, "ts": now_ts,
                            "stop": stop, "trail": stop, "highest": avg_px, "atr": atr
                        }
                        trade_log.append(f"{ist_now()} | {pair} | BUY {filled} @ {avg_px} | stop={round(stop,6)} | oid={oid}")
                        open_count += 1
                        can_open_more = open_count < MAX_CONCURRENT_POS
                    else:
                        scan_log.append(f"{ist_now()} | {pair} | BUY no fill | oid={oid}")
                        cooldown_until[pair] = now_ts + COOLDOWN_SEC
                else:
                    scan_log.append(f"{ist_now()} | {pair} | BUY failed (no oid) | res={res}")
                    cooldown_until[pair] = now_ts + COOLDOWN_SEC
            else:
                scan_log.append(f"{ist_now()} | {pair} | BUY qty=0 (risk/min-qty/fees)")
            continue  # after buy attempt, skip exit branch this tick

        # EXIT / MANAGEMENT
        if have_pos:
            p = positions[pair]
            qty   = p["qty"]
            entry = p["entry"]
            atr_p = p.get("atr", atr) or atr

            # trailing update
            p["highest"] = max(p.get("highest", entry), last)
            new_trail = p["highest"] - TRAIL_MULT * atr_p
            if new_trail > p.get("trail", p["stop"]):
                p["trail"] = new_trail
            active_stop = max(p.get("stop", entry - ATR_SL_MULT*atr_p), p.get("trail", 0.0))

            take_profit = last >= entry + ATR_TP_MULT * atr_p
            hard_stop   = last <= active_stop
            give_up     = (m_now < s_now) and (last < ema_htf)  # momentum loss + below trend

            scan_log.append(
                f"{ist_now()} | {pair} | MANAGE entry={entry} last={last} "
                f"stop={round(active_stop,6)} tp@{round(entry + ATR_TP_MULT*atr_p,6)} "
                f"| TP={take_profit} SL={hard_stop} giveup={give_up}"
            )

            if hard_stop or take_profit or give_up:
                res = place_market(pair, "SELL", qty)
                oid = _extract_order_id(res)
                if oid:
                    st = get_order_status(order_id=oid)
                    filled, avg_px = _filled_avg_from_status(st)
                    if filled > 0 and avg_px > 0:
                        pnl = _pnl_after_fees(entry, avg_px, min(filled, qty))
                        record_realized_pnl(pnl)
                        trade_log.append(f"{ist_now()} | {pair} | SELL {filled} @ {avg_px} | PNL={round(pnl,6)} | oid={oid}")
                        positions.pop(pair, None)
                        cooldown_until[pair] = now_ts + COOLDOWN_SEC
                        open_count -= 1
                        can_open_more = open_count < MAX_CONCURRENT_POS
                    else:
                        scan_log.append(f"{ist_now()} | {pair} | SELL no fill | oid={oid}")
                else:
                    scan_log.append(f"{ist_now()} | {pair} | SELL failed (no oid) | res={res}")

# -------------------- Main loop --------------------
_autostart_lock = threading.Lock()

def scan_loop():
    global running, status_epoch, last_keepalive, _last_rules_refresh
    scan_log.clear()
    running = True

    while running:
        now_real = time.time()

        # keepalive
        if now_real - last_keepalive >= KEEPALIVE_SEC:
            _keepalive_ping()
            last_keepalive = now_real

        # refresh pair rules periodically
        if (time.time() - _last_rules_refresh) >= RULES_REFRESH_SEC:
            fetch_pair_precisions()
            _last_rules_refresh = time.time()

        try:
            strategy_scan()
        except Exception as e:
            msg = f"{ist_now()} | scan_loop error: {e}"
            scan_log.append(msg)
            scan_log.append(traceback.format_exc().splitlines()[-1])

        status["msg"], status["last"] = "Running", ist_now()
        status_epoch = int(time.time())
        time.sleep(POLL_SEC)

    status["msg"] = "Idle"

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

    # current positions snapshot
    pos = {}
    for p, v in positions.items():
        pos[p] = {"qty": v["qty"], "entry": v["entry"], "ts": v["ts"]}

    # keepalive status
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
        "balances_age_sec": balances_age_sec(),
        "positions": pos,
        "profit_today": compute_realized_pnl_today(),
        "pnl_cumulative": round(profit_state.get("cumulative_pnl", 0.0), 6),
        "trades": trade_log[-50:][::-1],
        "scans": scan_log[-120:][::-1],
        "keepalive": keepalive_info
    })

@app.route("/io")
def get_io():
    # Return last N raw I/O lines
    return jsonify({"io": list(io_log)})

@app.route("/ping", methods=["GET", "HEAD"])
def ping():
    token = os.environ.get("KEEPALIVE_TOKEN", "")
    provided = (request.args.get("t") or
                request.headers.get("X-Keepalive-Token") or "")
    if token and provided != token:
        print(f"[{ist_now()}] /ping forbidden (bad token) method={request.method}")
        return "forbidden", 403

    print(f"[{ist_now()}] /ping ok method={request.method}")
    if request.method == "HEAD":
        return ("", 200)
    return ("pong", 200)

# -------------------- Boot helpers --------------------
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
        time.sleep(0.5)
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
