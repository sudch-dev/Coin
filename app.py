# app.py
import os
import time
import random
import threading
import hmac
import hashlib
import requests
import json
import traceback
from flask import Flask, render_template, jsonify, request
from datetime import datetime, timedelta
from pytz import timezone
from collections import defaultdict, deque

# ========================= Flask =========================
app = Flask(__name__)

APP_BASE_URL    = os.environ.get("APP_BASE_URL", "")
KEEPALIVE_TOKEN = os.environ.get("KEEPALIVE_TOKEN", "")

# ========================= Exchange creds =========================
API_KEY = os.environ.get("API_KEY", "")
API_SECRET_RAW = os.environ.get("API_SECRET", "")
API_SECRET = API_SECRET_RAW.encode() if isinstance(API_SECRET_RAW, str) else API_SECRET_RAW
BASE_URL = "https://api.coindcx.com"

# ========================= Market rules (USDT only) =========================
PAIR_RULES = {}              # pair -> {price_precision, qty_precision, min_qty, min_notional}
ALL_USDT_PAIRS = []          # discovered USDT pairs
MAX_MONITORED_PAIRS = int(os.environ.get("MAX_MONITORED_PAIRS", "200"))

# ========================= Core knobs =========================
CANDLE_INTERVAL    = 5           # seconds (5s bars)
POLL_SEC           = 1.0
FEE_PCT_PER_SIDE   = 0.0010      # taker fee per side
BUY_HEADROOM       = 1.0005
BUY_FRACTION_USDT  = 0.30        # cap notional per buy vs free USDT
KEEPALIVE_SEC      = 240
RULES_REFRESH_SEC  = 30 * 24 * 3600   # 30 days

# ---- Profiles (plus TURBO) ----
STRATEGY_MODE = os.environ.get("STRATEGY_MODE", "balanced")
STRATEGY_PROFILES = {
    "aggressive":   {"BRK_LOOKBACK":24, "MIN_ATR_PCT":0.0005, "ATR_SL_MULT":0.6, "ATR_TP_MULT":1.2, "RISK_PER_TRADE":0.015, "MAX_CONCURRENT_POS":2},
    "balanced":     {"BRK_LOOKBACK":36, "MIN_ATR_PCT":0.0008, "ATR_SL_MULT":0.7, "ATR_TP_MULT":1.6, "RISK_PER_TRADE":0.010, "MAX_CONCURRENT_POS":1},
    "conservative": {"BRK_LOOKBACK":60, "MIN_ATR_PCT":0.0012,"ATR_SL_MULT":0.8, "ATR_TP_MULT":2.0, "RISK_PER_TRADE":0.007, "MAX_CONCURRENT_POS":1},
    "turbo":        {"BRK_LOOKBACK":12, "MIN_ATR_PCT":0.0003,"ATR_SL_MULT":0.5, "ATR_TP_MULT":1.0, "RISK_PER_TRADE":0.030, "MAX_CONCURRENT_POS":3}
}
# bound at apply_profile()
BRK_LOOKBACK=36; MIN_ATR_PCT=0.0008; ATR_SL_MULT=0.7; ATR_TP_MULT=1.6; RISK_PER_TRADE=0.01; MAX_CONCURRENT_POS=1

# ---- Risk brakes / capital policy ----
DAILY_DD_STOP = float(os.environ.get("DAILY_DD_STOP", "-0.03"))  # stop opening if daily pnl < -3% of USDT
START_WITH_USDT = os.environ.get("START_WITH_USDT", "1") == "1"  # liquidate non-USDT on boot
FORCE_LIQUIDATE_ON_NEED = os.environ.get("FORCE_LIQUIDATE_ON_NEED", "1") == "1"

# ========================= Time helpers =========================
IST = timezone('Asia/Kolkata')
def ist_now(): return datetime.now(IST).strftime('%Y-%m-%d %H:%M:%S')
def ist_date(): return datetime.now(IST).strftime('%Y-%m-%d')

# ========================= State =========================
tick_logs   = defaultdict(list)     # pair -> [(ts, price)]
candle_logs = defaultdict(list)     # pair -> [{open,high,low,close,volume,start}]
scan_log    = []                    # decision + diagnostics
trade_log   = []                    # executions & important actions
io_log      = deque(maxlen=int(os.environ.get("IO_MAX_LINES", "500")))
running     = False
status      = {"msg": "Idle", "last": ""}
status_epoch= 0
last_keepalive = 0
_last_rules_refresh = 0

# positions: pair -> {"qty","entry","ts","stop","trail","highest","atr","tp"}
positions = {}
cooldown_until = {}                 # pair -> epoch when entries allowed again

# P&L persistence
PROFIT_STATE_FILE = "profit_state.json"
profit_state = {"cumulative_pnl": 0.0, "daily": {}, "processed_orders": []}

# Balance cache (rate-limited)
BAL_TTL_SEC = max(12, int(os.environ.get("BAL_TTL_SEC", "15")))   # avoid hammering balances
_bal_cache = {}
_bal_cache_ts = 0

# Runtime toggles (real trades by default)
IGNORE_BAL_AGE = False
PAPER_TRADE    = False
ADMIN_TOGGLE_KEY = os.environ.get("ADMIN_TOGGLE_KEY", "")  # set in env to allow toggles

# ========================= HTTP session: robust + retries/backoff =========================
from requests.adapters import HTTPAdapter
try:
    from urllib3.util.retry import Retry
except Exception:
    from requests.packages.urllib3.util.retry import Retry  # vendored fallback

SESSION = requests.Session()
SESSION.headers.update({
    "User-Agent": "CoinDCXBot/Allocator 3.0",
    "Connection": "keep-alive",
    "Accept": "application/json",
})
retry_cfg = Retry(
    total=5, connect=5, read=5,
    backoff_factor=0.4,                          # 0.4, 0.8, 1.6, ...
    status_forcelist=(429, 500, 502, 503, 504, 520),
    allowed_methods=False,                        # include POST
    raise_on_status=False,
)
adapter = HTTPAdapter(max_retries=retry_cfg, pool_connections=64, pool_maxsize=64)
SESSION.mount("https://", adapter)
SESSION.mount("http://", adapter)

# Circuit-breaker / pacing
_err_streak = {"get": 0, "post": 0}
_poll_backoff = 0.0  # added to POLL_SEC in the main loop after bursts of errors

# I/O logging (summarized)
IO_ENABLED   = os.environ.get("IO_ENABLED", "1") == "1"
IO_MAX_FIELD = int(os.environ.get("IO_MAX_FIELD", "2000"))
IO_LOG_FILE  = os.environ.get("IO_LOG_FILE", "io_log.txt")

def _truncate(s, n):
    try: s = s if isinstance(s, str) else json.dumps(s)
    except Exception: s = str(s)
    if s is None: return ""
    return s if len(s) <= n else s[:n] + f"...(+{len(s)-n} chars)"

def _mask_headers(h):
    if not isinstance(h, dict): return {}
    out = {}
    for k, v in h.items():
        out[k] = "***" if k.lower() in ("x-auth-signature","x-auth-apikey","authorization") else v
    return out

def log_io(direction, url, headers=None, payload=None, status=None, response=None):
    if not IO_ENABLED: return
    short_resp = None
    try:
        if response:
            rs = str(response)
            short_resp = rs if len(rs) <= 500 else f"<{len(rs)} chars>"
    except Exception:
        short_resp = "<unloggable>"
    rec = {
        "ts": ist_now(), "dir": direction, "url": url, "status": status,
        "headers": _mask_headers(headers or {}),
        "payload": _truncate(payload, IO_MAX_FIELD) if payload is not None else None,
        "response": short_resp,
    }
    line = f"{rec['ts']} | {rec['dir']} | {rec['url']} | status={rec['status']} | resp={short_resp}"
    io_log.append(line)
    try:
        if IO_LOG_FILE:
            with open(IO_LOG_FILE, "a") as f:
                f.write(line + "\n")
    except Exception:
        pass

# ========================= Signing & HTTP =========================
def hmac_signature(payload_str):
    # payload_str must be the exact string we send on the wire
    return hmac.new(API_SECRET, payload_str.encode(), hashlib.sha256).hexdigest()

def _http_post_with_retry(url, headers, payload, timeout=12, retries=2, backoff=0.5):
    """Our wrapper on top of Session retries; adds jitter + circuit-breaker."""
    global _err_streak, _poll_backoff
    for i in range(retries + 1):
        try:
            log_io("REQ", url, headers=headers, payload=payload)
            r = SESSION.post(url, headers=headers, data=payload, timeout=timeout)
            body = r.text if hasattr(r, "text") else ""
            log_io("RES", url, status=r.status_code, response=body)
            if r.status_code in (429, 500, 502, 503, 504, 520):
                time.sleep(backoff * (2 ** i) + random.random() * 0.25)
                continue
            _err_streak["post"] = 0
            return r
        except Exception as e:
            scan_log.append(f"{ist_now()} | HTTP POST err {url.split('/exchange',1)[-1]}: {e.__class__.__name__} {e}")
            time.sleep(backoff * (2 ** i) + random.random() * 0.25)
    _err_streak["post"] += 1
    _poll_backoff = min(3.0, 0.5 * _err_streak["post"])
    return None

def _http_get_with_retry(url, timeout=10, retries=2, backoff=0.5):
    global _err_streak, _poll_backoff
    for i in range(retries + 1):
        try:
            log_io("REQ", url)
            r = SESSION.get(url, timeout=timeout)
            body = r.text if hasattr(r, "text") else ""
            log_io("RES", url, status=r.status_code, response=body)
            if r.status_code in (429, 500, 502, 503, 504, 520):
                time.sleep(backoff * (2 ** i) + random.random() * 0.25)
                continue
            _err_streak["get"] = 0
            return r
        except Exception as e:
            scan_log.append(f"{ist_now()} | HTTP GET err {url.split('/exchange',1)[-1]}: {e.__class__.__name__} {e}")
            time.sleep(backoff * (2 ** i) + random.random() * 0.25)
    _err_streak["get"] += 1
    _poll_backoff = min(3.0, 0.5 * _err_streak["get"])
    return None

def _signed_post(url, body_dict, timeout=12):
    payload = json.dumps(body_dict, separators=(',', ':'))  # exact string must be signed
    sig = hmac_signature(payload)
    headers = {
        "X-AUTH-APIKEY": API_KEY,
        "X-AUTH-SIGNATURE": sig,
        "Content-Type": "application/json"
    }
    r = _http_post_with_retry(url, headers, payload, timeout=timeout, retries=2)
    if r is None:
        return {}
    if r.ok and r.headers.get("content-type","").startswith("application/json"):
        return r.json()
    _log_http_issue(f"POST {url}", r)
    return {}

def _log_http_issue(prefix, r):
    try:
        body = r.text[:240] if hasattr(r, "text") else ""
        scan_log.append(f"{ist_now()} | {prefix} HTTP {r.status_code} | {body}")
    except Exception as e:
        scan_log.append(f"{ist_now()} | {prefix} log-fail: {e}")

# ========================= Rules, balances, prices =========================
def refresh_markets_and_pairs():
    """Refresh rules and USDT market list (rarely changes)."""
    global PAIR_RULES, ALL_USDT_PAIRS
    url = f"{BASE_URL}/exchange/v1/markets_details"
    r = _http_get_with_retry(url, timeout=15, retries=2)
    if not r or not r.ok:
        if r: _log_http_issue("markets_details", r)
        return
    data = r.json()
    rules = {}
    usdt_pairs = []
    for it in data:
        p = it.get("pair") or it.get("market") or it.get("coindcx_name")
        if not p: 
            continue
        rules[p] = {
            "price_precision": int(it.get("target_currency_precision", 6)),  # PRICE
            "qty_precision":   int(it.get("base_currency_precision", 6)),    # QTY
            "min_qty":         float(it.get("min_quantity", 0.0) or 0.0),
            "min_notional":    float(it.get("min_notional", 0.0) or 0.0),
        }
        if p.endswith("USDT"):
            usdt_pairs.append(p)
    PAIR_RULES = rules
    ALL_USDT_PAIRS = sorted(usdt_pairs)
    scan_log.append(f"{ist_now()} | market rules refreshed | USDT pairs={len(ALL_USDT_PAIRS)}")

def get_wallet_balances():
    """Respects BAL_TTL_SEC to avoid hammering the API."""
    global _bal_cache, _bal_cache_ts
    now = time.time()
    if _bal_cache and (now - _bal_cache_ts) < BAL_TTL_SEC:
        return dict(_bal_cache)
    payload = json.dumps({"timestamp": int(now * 1000)}, separators=(',', ':'))
    sig = hmac_signature(payload)
    headers = {"X-AUTH-APIKEY": API_KEY, "X-AUTH-SIGNATURE": sig, "Content-Type": "application/json"}
    r = _http_post_with_retry(f"{BASE_URL}/exchange/v1/users/balances", headers, payload, timeout=12, retries=2)
    if r and r.ok:
        balances = {}
        try:
            for b in r.json():
                balances[b["currency"]] = float(b["balance"])
            _bal_cache, _bal_cache_ts = balances, now
            return balances
        except Exception as e:
            scan_log.append(f"{ist_now()} | balances parse fail: {e}")
    else:
        if r: _log_http_issue("balances", r)
    return dict(_bal_cache) if _bal_cache else {}

def balances_age_sec():
    return int(time.time() - _bal_cache_ts) if _bal_cache_ts else None

def fetch_all_prices():
    """
    Pull tickers and keep USDT pairs only. Timeout short; cap the pairs per tick.
    """
    r = _http_get_with_retry(f"{BASE_URL}/exchange/ticker", timeout=7, retries=2)
    if r and r.ok:
        now = int(time.time())
        ret = {}
        count = 0
        for item in r.json():
            m = item.get("market")
            if not m or not m.endswith("USDT"): continue
            if m not in PAIR_RULES: continue
            ret[m] = {"price": float(item["last_price"]), "ts": now}
            count += 1
            if count >= MAX_MONITORED_PAIRS: break
        return ret
    else:
        if r: _log_http_issue("ticker", r)
    return {}

# ========================= Precision & formatting =========================
def _qty_prec(pair):  return int(PAIR_RULES.get(pair, {}).get("qty_precision", 6))
def _price_prec(pair):return int(PAIR_RULES.get(pair, {}).get("price_precision", 6))
def _min_qty(pair):   return float(PAIR_RULES.get(pair, {}).get("min_qty", 0.0))
def _min_notional(pair): return float(PAIR_RULES.get(pair, {}).get("min_notional", 0.0))

def fmt_price(pair, price):
    pp = _price_prec(pair)
    return float(f"{float(price):.{pp}f}")

def fmt_qty_floor(pair, qty):
    """Floor to allowed step, then enforce min_qty. Never rounds up."""
    qp   = _qty_prec(pair)
    step = 10 ** (-qp)
    if qty <= 0:
        return 0.0
    q = float(qty)
    q = (int(q / step)) * step  # floor
    q = float(f"{q:.{qp}f}")
    return q if q >= _min_qty(pair) else 0.0

def _qty_step(pair): return 10 ** (-_qty_prec(pair))
def _fee_multiplier(side): return 1.0 + FEE_PCT_PER_SIDE if side.upper()=="BUY" else 1.0 - FEE_PCT_PER_SIDE

def normalize_qty_for_buy(pair, price, usdt_avail):
    if price <= 0: return 0.0
    denom = max(price * _fee_multiplier("BUY") * BUY_HEADROOM, 1e-12)
    q = usdt_avail / denom
    q = fmt_qty_floor(pair, q)
    step = _qty_step(pair)
    while q >= _min_qty(pair) and (price * q * _fee_multiplier("BUY") * BUY_HEADROOM) > usdt_avail + 1e-12:
        q = max(0.0, q - step)
        q = fmt_qty_floor(pair, q)
    if q < _min_qty(pair): return 0.0
    return q

# ========================= Orders (real / paper) =========================
def place_market(pair, side, qty):
    """
    Create a MARKET order. Floors qty to step; enforces min_qty.
    """
    # paper disabled by default; keep path for testing
    if PAPER_TRADE:
        px = candle_logs.get(pair, [])[-1]["close"] if candle_logs.get(pair) else 0.0
        oid = f"paper-{int(time.time()*1000)}"
        trade_log.append(f"{ist_now()} | {pair} | PAPER SUBMIT {side} qty={qty} @ MKT (px≈{px}) | oid={oid}")
        return {"id": oid, "status": "accepted", "side": side.lower(), "qty": qty}

    qty = fmt_qty_floor(pair, qty)
    if qty <= 0:
        scan_log.append(f"{ist_now()} | {pair} | ABORT {side}: qty too small after floor")
        return {}

    payload = {
        "market": pair,
        "side": side.lower(),                  # "buy" / "sell"
        "order_type": "market_order",
        "total_quantity": f"{qty}",
        "timestamp": int(time.time() * 1000),
    }

    trade_log.append(f"{ist_now()} | {pair} | SUBMIT {side} qty={payload['total_quantity']} @ MKT")
    res = _signed_post(f"{BASE_URL}/exchange/v1/orders/create", payload, timeout=10)
    trade_log.append(f"{ist_now()} | {pair} | SUBMIT RESP => {res}")
    return res

def get_order_status(order_id=None, client_order_id=None):
    body = {"timestamp": int(time.time() * 1000)}
    if order_id: body["id"] = order_id
    if client_order_id: body["client_order_id"] = client_order_id
    return _signed_post(f"{BASE_URL}/exchange/v1/orders/status", body, timeout=12) or {}

def _extract_order_id(res: dict):
    if not isinstance(res, dict): return None
    try:
        if isinstance(res.get("orders"), list) and res["orders"]:
            o = res["orders"][0]
            for k in ("id","order_id","orderId","client_order_id","clientOrderId"):
                if o.get(k): return str(o[k])
    except Exception:
        pass
    for k in ("id","order_id","orderId","client_order_id","clientOrderId"):
        if res.get(k): return str(res[k])
    d = res.get("data") or {}
    for k in ("id","order_id","orderId","client_order_id","clientOrderId"):
        if d.get(k): return str(d[k])
    return None

def _filled_avg_from_status(st):
    try:
        total_q  = float(st.get("total_quantity", st.get("quantity", 0)))
        remain_q = float(st.get("remaining_quantity", st.get("remaining_qty", 0)))
        exec_q   = float(st.get("executed_quantity", st.get("filled_qty", 0)))
        filled   = exec_q if exec_q > 0 else max(0.0, total_q - remain_q)
        avg_px   = float(st.get("avg_price", st.get("average_price", st.get("avg_execution_price", st.get("price", 0)))))
        return filled, avg_px
    except Exception:
        return 0.0, 0.0

# ========================= P&L helpers =========================
def load_profit_state():
    try:
        with open(PROFIT_STATE_FILE, "r") as f:
            data = json.load(f)
        profit_state["cumulative_pnl"] = float(data.get("cumulative_pnl", 0.0))
        profit_state["daily"] = dict(data.get("daily", {}))
        profit_state["processed_orders"] = list(data.get("processed_orders", []))
    except Exception:
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
    except Exception:
        pass

def record_realized_pnl(pnl):
    pnl = float(pnl or 0.0)
    profit_state["cumulative_pnl"] = float(profit_state.get("cumulative_pnl", 0.0) + pnl)
    d = ist_date()
    profit_state["daily"][d] = float(profit_state["daily"].get(d, 0.0) + pnl)
    save_profit_state()

def compute_realized_pnl_today():
    return round(profit_state["daily"].get(ist_date(), 0.0), 6)

# ========================= Liquidation helpers =========================
def safe_sell_qty(pair, wallet_qty, last_px=None):
    """Clamp DOWN to wallet amount & step; enforce min_qty; return 0 if too small/notional too small."""
    if wallet_qty <= 0: 
        return 0.0
    qp = _qty_prec(pair)
    step = 10 ** (-qp)
    mq = _min_qty(pair)

    q = float(wallet_qty)
    q = (int(q / step)) * step
    q = float(f"{q:.{qp}f}")
    if q < mq:
        return 0.0
    if last_px is not None:
        min_notional = float(_min_notional(pair))
        if last_px * q < min_notional:
            return 0.0
    return q

def liquidate_all_non_usdt(reason="boot"):
    """
    Sell any non-USDT balances that have a valid USDT market and meet min-qty/notional.
    Logs an explicit reason if a coin is skipped.
    """
    bal = get_wallet_balances()
    prices = fetch_all_prices()  # we only keep *USDT* pairs here

    for cur, amt in bal.items():
        if cur == "USDT" or amt <= 0:
            continue

        usdt_pair = f"{cur}USDT"
        if usdt_pair not in PAIR_RULES:
            scan_log.append(f"{ist_now()} | {cur} | SKIP-LIQ: no USDT market (rules missing)")
            continue

        last_px = float(prices.get(usdt_pair, {}).get("price", 0.0))
        q = safe_sell_qty(usdt_pair, amt, last_px)
        if q <= 0:
            rq = _min_qty(usdt_pair)
            mn = float(_min_notional(usdt_pair))
            scan_log.append(
                f"{ist_now()} | {cur} | SKIP-LIQ: qty/notional too small (bal={amt}, min_qty={rq}, min_notional={mn}, px={last_px})"
            )
            continue

        trade_log.append(f"{ist_now()} | {usdt_pair} | FORCE-LIQ {reason} | SELL {q} @ MKT")
        res = place_market(usdt_pair, "SELL", q)
        oid = _extract_order_id(res)
        if oid:
            st = get_order_status(order_id=oid)
            filled, px = _filled_avg_from_status(st)
            trade_log.append(f"{ist_now()} | {usdt_pair} | FORCE-LIQ status oid={oid} filled={filled} px={px}")
        else:
            trade_log.append(f"{ist_now()} | {usdt_pair} | FORCE-LIQ failed (no oid) | res={res}")

def ensure_usdt_liquidity(usdt_needed):
    bal = get_wallet_balances()
    have = float(bal.get("USDT", 0.0))
    if have >= usdt_needed: return True
    short = usdt_needed - have
    if not FORCE_LIQUIDATE_ON_NEED:
        scan_log.append(f"{ist_now()} | LIQUIDITY short={short} (skipped: FORCE_LIQUIDATE_ON_NEED=False)")
        return False

    # Close open positions first
    for pair, pos in list(positions.items()):
        q = safe_sell_qty(pair, pos["qty"], last_px=None)
        if q <= 0:
            scan_log.append(f"{ist_now()} | {pair} | LIQUIDITY-EXIT skipped: qty below min step/min_qty")
            continue
        trade_log.append(f"{ist_now()} | {pair} | LIQUIDITY-EXIT | SELL {q} @ MKT")
        res = place_market(pair, "SELL", q)
        oid = _extract_order_id(res)
        if oid:
            st = get_order_status(order_id=oid)
            filled, px = _filled_avg_from_status(st)
            pnl = (px - pos["entry"]) * min(filled, q) * (1 - 2*FEE_PCT_PER_SIDE)  # fee-aware approx
            record_realized_pnl(pnl)
            trade_log.append(f"{ist_now()} | {pair} | LIQUIDITY-EXIT filled={filled} px={px} pnl={round(pnl,6)}")
            positions.pop(pair, None)
            bal = get_wallet_balances(); have = float(bal.get("USDT",0.0))
            if have >= usdt_needed: return True

    # Sell any stray balances (non-USDT)
    bal = get_wallet_balances()
    prices = fetch_all_prices()
    for cur, amt in bal.items():
        if cur == "USDT" or amt <= 0:
            continue
        pair = f"{cur}USDT"
        if pair not in PAIR_RULES:
            scan_log.append(f"{ist_now()} | {cur} | LIQUIDITY-SELL skipped: no USDT market")
            continue
        last_px = float(prices.get(pair, {}).get("price", 0.0))
        q = safe_sell_qty(pair, amt, last_px)
        if q <= 0:
            scan_log.append(f"{ist_now()} | {cur} | LIQUIDITY-SELL skipped: qty/notional too small (bal={amt}, px={last_px})")
            continue
        trade_log.append(f"{ist_now()} | {pair} | LIQUIDITY-SELL wallet | SELL {q} @ MKT")
        res = place_market(pair, "SELL", q)
        oid = _extract_order_id(res)
        if oid:
            get_order_status(order_id=oid)  # best-effort
            bal = get_wallet_balances(); have = float(bal.get("USDT",0.0))
            if have >= usdt_needed: return True

    bal = get_wallet_balances(); have = float(bal.get("USDT",0.0))
    scan_log.append(f"{ist_now()} | LIQUIDITY still short={max(0.0, usdt_needed-have)} after liquidation")
    return have >= usdt_needed

# ========================= Indicators =========================
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
            candle["low"]  = min(candle["low"],  price)
            candle["close"]= price
            candle["volume"] += 1
    if candle: candles.append(candle)
    candle_logs[pair] = candles[-300:]  # ~25 min

def _ema_series(vals, n):
    if not vals: return []
    k = 2/(n+1)
    ema_vals=[]; ema = vals[0]
    for v in vals:
        ema = v*k + ema*(1-k)
        ema_vals.append(ema)
    return ema_vals

def _macd_series(closes):
    MACD_FAST=12; MACD_SLOW=26; MACD_SIGNAL=9
    if len(closes) < MACD_SLOW + MACD_SIGNAL: return [],[],[]
    ema_fast = _ema_series(closes, MACD_FAST)
    ema_slow = _ema_series(closes, MACD_SLOW)
    macd = [f-s for f,s in zip(ema_fast, ema_slow)]
    signal = _ema_series(macd, MACD_SIGNAL)
    hist = [m - s for m,s in zip(macd[-len(signal):], signal)]
    return macd[-len(signal):], signal, hist

def _atr_from_candles(cs, n):
    if len(cs) < n + 2: return 0.0
    trs=[]; prev_close = cs[-(n+1)]["close"]
    for c in cs[-n:]:
        high,low,close = c["high"], c["low"], c["close"]
        tr = max(high-low, abs(high-prev_close), abs(low-prev_close))
        trs.append(tr); prev_close = close
    k = 2/(n+1); atr = trs[0]
    for tr in trs:
        atr = tr*k + atr*(1-k)
    return float(atr)

def _recent_hl(cs, lb):
    if len(cs) < lb: return None, None
    highs=[c["high"] for c in cs[-lb:]]; lows=[c["low"] for c in cs[-lb:]]
    return max(highs), min(lows)

# ========================= Scoring / selection =========================
def fee_aware_tp(entry, atr):
    gross_target = entry + ATR_TP_MULT * atr
    fee_buffer   = entry * (2 * FEE_PCT_PER_SIDE + 0.0001)
    return gross_target + fee_buffer

def _size_by_risk(usdt_free, price, atr, risk_per_trade):
    if atr<=0 or price<=0: return 0.0
    stop_dist = ATR_SL_MULT * atr
    if stop_dist <= 0: return 0.0
    risk_amt = max(0.0, risk_per_trade * usdt_free)
    if risk_amt <= 0: return 0.0
    qty_risk = risk_amt / stop_dist
    cap_notional = BUY_FRACTION_USDT * usdt_free
    qty_cap = (cap_notional / price) if cap_notional>0 else qty_risk
    return max(0.0, min(qty_risk, qty_cap))

def score_pair(pair, cs):
    """Return (score, details) — higher is better; demote if any gate fails."""
    if len(cs) < max(36, BRK_LOOKBACK) + 3:
        return -1e9, {"reason": "insufficient_candles"}
    closes = [c["close"] for c in cs]
    last = closes[-1]
    ema_htf = _ema_series(closes, 12)[-1] if len(closes)>=12 else last
    macd_line, macd_signal, _ = _macd_series(closes)
    if not macd_line or not macd_signal:
        return -1e9, {"reason":"no_macd"}
    m_now, s_now = macd_line[-1], macd_signal[-1]
    atr = _atr_from_candles(cs, 36)
    atr_pct = (atr/last) if last>0 else 0.0
    hi, lo = _recent_hl(cs, BRK_LOOKBACK)

    volatility_ok = atr_pct >= MIN_ATR_PCT
    trend_ok      = last > ema_htf
    momentum_ok   = m_now > s_now
    breakout_up   = (hi is not None) and (last > hi)

    breakout_mag = ((last - (hi or last)) / (atr if atr>0 else 1e-9))
    ema_premium  = (last - ema_htf) / (atr if atr>0 else 1e-9)
    mom_delta    = (m_now - s_now) / (abs(s_now)+1e-9)
    score = 100*atr_pct + 50*breakout_mag + 20*mom_delta + 10*ema_premium

    det = {
        "last": last, "ema": ema_htf, "macd": m_now, "sig": s_now,
        "atr": atr, "atr_pct": atr_pct, "hi": hi, "lo": lo,
        "gates": {"volatility_ok": volatility_ok, "trend_ok": trend_ok,
                  "momentum_ok": momentum_ok, "breakout_up": breakout_up},
        "components": {"atr_pct": atr_pct, "breakout_mag": breakout_mag,
                       "mom_delta": mom_delta, "ema_premium": ema_premium}
    }
    if not (volatility_ok and trend_ok and momentum_ok and breakout_up):
        score -= 1e6
    return score, det

# ========================= Strategy core =========================
def strategy_scan():
    prices = fetch_all_prices()
    now_ts = int(time.time())

    # Build 5s candles
    for pair, obj in prices.items():
        px = obj["price"]
        tick_logs[pair].append((now_ts, px))
        if len(tick_logs[pair]) > 6000:
            tick_logs[pair] = tick_logs[pair][-6000:]
        aggregate_candles(pair, CANDLE_INTERVAL)

    # Wallet / buy gating (respect balances TTL unless override)
    balances = get_wallet_balances()
    usdt_free = float(balances.get("USDT", 0.0))
    bal_age = balances_age_sec()
    buys_allowed = (bal_age is not None and bal_age <= BAL_TTL_SEC * 3)
    if IGNORE_BAL_AGE:
        buys_allowed = True
        scan_log.append(f"{ist_now()} | OVERRIDE: ignore_balance_age=True (age={bal_age}s)")
    elif not buys_allowed:
        scan_log.append(f"{ist_now()} | NOTE: buys paused (balances stale age={bal_age}s)")

    # Risk brake
    usdt_basis = balances.get("USDT", 0.0) or 1.0
    if compute_realized_pnl_today() <= DAILY_DD_STOP * usdt_basis:
        buys_allowed = False
        scan_log.append(f"{ist_now()} | DAILY DD STOP | buys blocked")

    # ---- SCORE all available pairs, pick the best ----
    best_pair = None; best_score = -1e12; best_det = None
    for pair, cs in candle_logs.items():
        if not cs: continue
        score, det = score_pair(pair, cs)
        scan_log.append(
            f"{ist_now()} | {pair} | score={round(score,3)} | Px={det.get('last')} ATR%={round(100*det.get('atr_pct',0),3)} "
            f"Gates v={det['gates']['volatility_ok']} t={det['gates']['trend_ok']} m={det['gates']['momentum_ok']} brk={det['gates']['breakout_up']}"
        )
        if score > best_score:
            best_score, best_pair, best_det = score, pair, det

    # Capacity
    open_count = sum(1 for v in positions.values() if v.get("qty", 0.0) > 0)
    can_open_more = open_count < MAX_CONCURRENT_POS

    # ---- ENTRY (allocator) ----
    if best_pair and can_open_more and buys_allowed:
        last = best_det["last"]; atr = best_det["atr"]; atr_pct = best_det["atr_pct"]
        gates = best_det["gates"]
        have_pos = best_pair in positions and positions[best_pair].get("qty", 0.0) > 0.0
        cooled = now_ts >= int(cooldown_until.get(best_pair, 0))

        scan_log.append(
            f"{ist_now()} | SELECT {best_pair} | best_score={round(best_score,3)} | Px={last} ATR%={round(100*atr_pct,3)} "
            f"| Gates={gates} | have_pos={have_pos} cooled={cooled}"
        )

        if (not have_pos) and cooled and all(gates.values()):
            # Calculate size then secure liquidity
            raw_qty = _size_by_risk(usdt_free, last, atr, RISK_PER_TRADE)
            target_notional = raw_qty * last
            min_needed = max(target_notional, _min_notional(best_pair))
            scan_log.append(f"{ist_now()} | {best_pair} | ENTRY-CALC usdt_free={usdt_free} raw_qty={raw_qty} notional={target_notional} min_need={min_needed}")

            if ensure_usdt_liquidity(min_needed):
                # refresh balances, normalize & place
                balances = get_wallet_balances()
                usdt_free = float(balances.get("USDT", 0.0))
                raw_qty = _size_by_risk(usdt_free, last, atr, RISK_PER_TRADE)
                target_notional = raw_qty * last
                qty = normalize_qty_for_buy(best_pair, last, target_notional)
                scan_log.append(f"{ist_now()} | {best_pair} | ENTRY-NORM qty={qty} usdt_free={usdt_free}")

                if qty > 0:
                    res = place_market(best_pair, "BUY", qty)
                    oid = _extract_order_id(res)
                    if oid:
                        st = get_order_status(order_id=oid)
                        filled, avg_px = _filled_avg_from_status(st)
                        if filled > 0 and avg_px > 0:
                            stop = max(0.0, avg_px - ATR_SL_MULT * atr)
                            tprice = fee_aware_tp(avg_px, atr)
                            positions[best_pair] = {
                                "qty": filled, "entry": avg_px, "ts": now_ts,
                                "stop": stop, "trail": stop, "highest": avg_px, "atr": atr, "tp": tprice
                            }
                            trade_log.append(f"{ist_now()} | {best_pair} | BUY {filled} @ {avg_px} | stop={round(stop,6)} tp={round(tprice,6)} | oid={oid}")
                            open_count += 1; can_open_more = open_count < MAX_CONCURRENT_POS
                        else:
                            scan_log.append(f"{ist_now()} | {best_pair} | BUY no fill | oid={oid}")
                            cooldown_until[best_pair] = now_ts + 20
                    else:
                        scan_log.append(f"{ist_now()} | {best_pair} | BUY failed (no oid) | res={res}")
                        cooldown_until[best_pair] = now_ts + 20
                else:
                    scan_log.append(f"{ist_now()} | {best_pair} | BUY qty=0 (risk/min-qty/fees)")
            else:
                scan_log.append(f"{ist_now()} | {best_pair} | ENTRY aborted: insufficient liquidity")

    # ---- EXIT / MANAGEMENT ----
    for pair, p in list(positions.items()):
        cs = candle_logs.get(pair) or []
        if not cs: continue
        closes = [c["close"] for c in cs]
        last = closes[-1]
        atr_p = p.get("atr", _atr_from_candles(cs, 36))
        p["highest"] = max(p.get("highest", p["entry"]), last)
        new_trail = p["highest"] - ATR_SL_MULT * atr_p
        if new_trail > p.get("trail", p["stop"]): p["trail"] = new_trail
        active_stop = max(p.get("stop", p["entry"] - ATR_SL_MULT*atr_p), p.get("trail", 0.0))
        tp_price = p.get("tp", fee_aware_tp(p["entry"], atr_p))
        take_profit = last >= tp_price
        hard_stop   = last <= active_stop
        ema_htf = _ema_series(closes, 12)[-1] if len(closes)>=12 else last
        macd_line, macd_signal, _ = _macd_series(closes)
        give_up = (macd_line and macd_signal) and (macd_line[-1] < macd_signal[-1]) and (last < ema_htf)

        scan_log.append(
            f"{ist_now()} | {pair} | MANAGE entry={p['entry']} last={last} stop={round(active_stop,6)} "
            f"tp={round(tp_price,6)} | TP={take_profit} SL={hard_stop} giveup={give_up}"
        )

        if hard_stop or take_profit or give_up:
            q = p["qty"]
            res = place_market(pair, "SELL", q)
            oid = _extract_order_id(res)
            if oid:
                st = get_order_status(order_id=oid)
                filled, avg_px = _filled_avg_from_status(st)
                if filled > 0 and avg_px > 0:
                    pnl = (avg_px - p["entry"]) * min(filled, q) * (1 - 2*FEE_PCT_PER_SIDE)
                    record_realized_pnl(pnl)
                    trade_log.append(f"{ist_now()} | {pair} | SELL {filled} @ {avg_px} | PNL={round(pnl,6)} | oid={oid}")
                    positions.pop(pair, None)
                    cooldown_until[pair] = int(time.time()) + 30
                else:
                    scan_log.append(f"{ist_now()} | {pair} | SELL no fill | oid={oid}")
            else:
                scan_log.append(f"{ist_now()} | {pair} | SELL failed (no oid) | res={res}")

# ========================= Keepalive & loop =========================
def _keepalive_ping():
    try:
        if APP_BASE_URL:
            url = f"{APP_BASE_URL}/ping"
            headers = {}
            if KEEPALIVE_TOKEN:
                url = f"{url}?t={KEEPALIVE_TOKEN}"
                headers["X-Keepalive-Token"] = KEEPALIVE_TOKEN
            SESSION.get(url, headers=headers, timeout=5)
    except Exception:
        pass

_autostart_lock = threading.Lock()

def scan_loop():
    global running, status_epoch, last_keepalive, _last_rules_refresh, _poll_backoff
    scan_log.clear()
    running = True
    while running:
        now_real = time.time()
        try:
            if now_real - last_keepalive >= KEEPALIVE_SEC:
                _keepalive_ping()
                last_keepalive = now_real
            if (time.time() - _last_rules_refresh) >= RULES_REFRESH_SEC:
                refresh_markets_and_pairs()
                _last_rules_refresh = time.time()
            strategy_scan()
        except Exception as e:
            scan_log.append(f"{ist_now()} | scan_loop error: {e}")
            scan_log.append(traceback.format_exc().splitlines()[-1])
        status["msg"], status["last"] = "Running", ist_now()
        status_epoch = int(time.time())
        # sleep with adaptive backoff
        sleep_for = POLL_SEC + float(_poll_backoff or 0.0)
        time.sleep(sleep_for)
        _poll_backoff = max(0.0, _poll_backoff - 0.2)  # decay towards normal
    status["msg"] = "Idle"

# ========================= Routes =========================
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
    pos = {p: {"qty": v["qty"], "entry": v["entry"], "ts": v["ts"]} for p, v in positions.items()}
    now_real = time.time(); last_age = now_real - (last_keepalive or 0)
    keepalive_info = {
        "enabled": bool(KEEPALIVE_TOKEN), "interval_sec": KEEPALIVE_SEC,
        "last_ping_epoch": int(last_keepalive or 0),
        "last_ping_age_sec": (max(0, int(last_age)) if last_keepalive else None),
        "next_due_sec": (max(0, int(KEEPALIVE_SEC - last_age)) if last_keepalive else None),
        "app_base_url": APP_BASE_URL,
    }
    return jsonify({
        "status": status["msg"], "last": status["last"], "status_epoch": status_epoch,
        "usdt": usdt_total, "balances_age_sec": balances_age_sec(),
        "positions": pos, "profit_today": compute_realized_pnl_today(),
        "pnl_cumulative": round(profit_state.get("cumulative_pnl", 0.0), 6),
        "trades": trade_log[-70:][::-1], "scans": scan_log[-180:][::-1],
        "keepalive": keepalive_info,
        "flags": {"ignore_balance_age": IGNORE_BAL_AGE, "paper_trade": PAPER_TRADE, "strategy_mode": STRATEGY_MODE}
    })

@app.route("/io")
def get_io():
    return jsonify({"io": list(io_log)})

@app.route("/toggle", methods=["POST"])
def toggle():
    try:
        data = request.get_json(force=True, silent=True) or {}
        key   = str(data.get("key",""))
        name  = str(data.get("name","")).strip().lower()
        state = data.get("state", None)
        if not ADMIN_TOGGLE_KEY or key != ADMIN_TOGGLE_KEY:
            return jsonify({"ok": False, "error": "unauthorized"}), 403

        def parse_state(x):
            if isinstance(x, bool): return x
            if isinstance(x, str):
                x = x.strip().lower()
                if x in ("1","true","on","yes","y"): return True
                if x in ("0","false","off","no","n"): return False
            return x

        global IGNORE_BAL_AGE, PAPER_TRADE, STRATEGY_MODE
        if name == "ignore_balance_age":
            val = bool(parse_state(state)); IGNORE_BAL_AGE = val
        elif name == "paper_trade":
            val = bool(parse_state(state)); PAPER_TRADE = val
        elif name == "strategy_mode":
            if state not in ("aggressive","balanced","conservative","turbo"):
                return jsonify({"ok": False, "error": "invalid mode"}), 400
            STRATEGY_MODE = state; apply_profile(); val = STRATEGY_MODE
        elif name == "force_rules_refresh":
            refresh_markets_and_pairs(); val = "done"
        else:
            return jsonify({"ok": False, "error": "unknown toggle"}), 400

        scan_log.append(f"{ist_now()} | TOGGLE {name} -> {val}")
        return jsonify({"ok": True, "name": name, "state": val})
    except Exception as e:
        return jsonify({"ok": False, "error": str(e)}), 500

@app.route("/ping", methods=["GET","HEAD"])
def ping():
    token = os.environ.get("KEEPALIVE_TOKEN", "")
    provided = (request.args.get("t") or request.headers.get("X-Keepalive-Token") or "")
    if token and provided != token:
        print(f"[{ist_now()}] /ping forbidden (bad token) method={request.method}")
        return "forbidden", 403
    print(f"[{ist_now()}] /ping ok method={request.method}")
    return ("", 200) if request.method == "HEAD" else ("pong", 200)

# ========================= Boot helpers =========================
def apply_profile():
    global BRK_LOOKBACK, MIN_ATR_PCT, ATR_SL_MULT, ATR_TP_MULT, RISK_PER_TRADE, MAX_CONCURRENT_POS
    p = STRATEGY_PROFILES.get(STRATEGY_MODE, STRATEGY_PROFILES["balanced"])
    BRK_LOOKBACK   = p["BRK_LOOKBACK"]
    MIN_ATR_PCT    = p["MIN_ATR_PCT"]
    ATR_SL_MULT    = p["ATR_SL_MULT"]
    ATR_TP_MULT    = p["ATR_TP_MULT"]
    RISK_PER_TRADE = p["RISK_PER_TRADE"]
    MAX_CONCURRENT_POS = p.get("MAX_CONCURRENT_POS", 1)
    scan_log.append(f"{ist_now()} | STRATEGY mode applied: {STRATEGY_MODE} => {p}")

def _start_loop_once():
    global running
    with _autostart_lock:
        if not running:
            running = True
            t = threading.Thread(target=scan_loop, daemon=True)
            t.start()

def _boot_kick():
    try:
        refresh_markets_and_pairs()
        time.sleep(0.5)
        if START_WITH_USDT:
            liquidate_all_non_usdt(reason="boot")
        _start_loop_once()
    except Exception as e:
        print("boot kick failed:", e)

if os.environ.get("AUTOSTART", "1") == "1":
    apply_profile()
    threading.Thread(target=_boot_kick, daemon=True).start()

if __name__ == "__main__":
    apply_profile()
    refresh_markets_and_pairs()
    if START_WITH_USDT:
        liquidate_all_non_usdt(reason="boot")
    _start_loop_once()
    port = int(os.environ.get("PORT", "10000"))
    app.run(host="0.0.0.0", port=port)
