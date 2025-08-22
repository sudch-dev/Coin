import os, time, threading, hmac, hashlib, requests, json, traceback
from flask import Flask, render_template, jsonify, request
from datetime import datetime
from pytz import timezone
from collections import deque

# -------------------- Flask & Env --------------------
app = Flask(__name__)

APP_BASE_URL   = os.environ.get("APP_BASE_URL", "https://coin-4k37.onrender.com")
KEEPALIVE_TOKEN= os.environ.get("KEEPALIVE_TOKEN", "")
API_KEY        = os.environ.get("API_KEY", "")
API_SECRET_RAW = os.environ.get("API_SECRET", "")
API_SECRET     = API_SECRET_RAW.encode() if isinstance(API_SECRET_RAW, str) else API_SECRET_RAW
BASE_URL       = "https://api.coindcx.com"

# Pairs (LTC removed)
PAIRS = ["BTCUSDT","ETHUSDT","XRPUSDT","SHIBUSDT","SOLUSDT","DOGEUSDT","ADAUSDT","AEROUSDT","BNBUSDT"]

# Local defaults; refreshed live
PAIR_RULES = {
    "BTCUSDT":  {"price_precision": 1, "qty_precision": 4, "min_qty": 0.001},
    "ETHUSDT":  {"price_precision": 2, "qty_precision": 6, "min_qty": 0.0001},
    "XRPUSDT":  {"price_precision": 4, "qty_precision": 4, "min_qty": 0.1},
    "SHIBUSDT": {"price_precision": 8, "qty_precision": 4, "min_qty": 10000},
    "DOGEUSDT": {"price_precision": 5, "qty_precision": 4, "min_qty": 0.001},
    "SOLUSDT":  {"price_precision": 2, "qty_precision": 4, "min_qty": 0.01},
    "AEROUSDT": {"price_precision": 3, "qty_precision": 2, "min_qty": 0.01},
    "ADAUSDT":  {"price_precision": 4, "qty_precision": 2, "min_qty": 0.1},
    "BNBUSDT":  {"price_precision": 3, "qty_precision": 4, "min_qty": 0.001},
}

# -------------------- Strategy Settings --------------------
CANDLE_INTERVAL = 5      # seconds per candle
POLL_SEC        = 1.0    # tick every 1 sec
EMA_FAST, EMA_SLOW = 5, 20
MACD_FAST, MACD_SLOW, MACD_SIGNAL = 12, 26, 9

INVEST_FRAC_USDT = 0.30  # 30% of free USDT on buy
TP_TARGET        = 0.01  # 1% profit target (exit on sell signal OR 1% profit)
KEEPALIVE_SEC    = 240

# -------------------- State --------------------
IST = timezone("Asia/Kolkata")
def ist_now(): return datetime.now(IST).strftime("%Y-%m-%d %H:%M:%S")

tick_logs   = {p: [] for p in PAIRS}
candle_logs = {p: [] for p in PAIRS}
scan_log    = deque(maxlen=1000)   # rolling buffer
trade_log   = deque(maxlen=300)
order_book  = deque(maxlen=200)    # open/filled/cancelled snapshots we add
profit_state = {"daily": {}, "cumulative": 0.0}

running = False
status  = {"msg":"Idle","last":"—"}
status_epoch = 0
last_keepalive = 0
error_message = ""

# active positions (simple: one entry per pair)
entries = {}  # pair -> {"price": float, "qty": float, "ts": int}

# -------------------- HTTP / Exchange helpers --------------------
def hmac_signature(payload):
    return hmac.new(API_SECRET, payload.encode(), hashlib.sha256).hexdigest()

def _signed_post(url, body):
    payload = json.dumps(body, separators=(",",":"))
    headers = {"X-AUTH-APIKEY": API_KEY, "X-AUTH-SIGNATURE": hmac_signature(payload), "Content-Type": "application/json"}
    try:
        r = requests.post(url, headers=headers, data=payload, timeout=10)
        if not r.ok:
            try:
                scan_log.append(f"{ist_now()} | POST {url} HTTP {r.status_code} | {r.text[:240]}")
            except:
                scan_log.append(f"{ist_now()} | POST {url} HTTP {r.status_code}")
            return {}
        return r.json()
    except Exception as e:
        scan_log.append(f"{ist_now()} | POST fail {url} | {e}")
        return {}

def fetch_pair_precisions():
    try:
        r = requests.get(f"{BASE_URL}/exchange/v1/markets_details", timeout=12)
        if not r.ok:
            scan_log.append(f"{ist_now()} | markets_details HTTP {r.status_code}")
            return
        n = 0
        for item in r.json():
            p = item.get("pair") or item.get("market") or item.get("coindcx_name")
            if p in PAIRS:
                PAIR_RULES[p] = {
                    "price_precision": int(item.get("target_currency_precision", 6)),
                    "qty_precision":   int(item.get("base_currency_precision", 6)),
                    "min_qty":         float(item.get("min_quantity", 0.0) or 0.0),
                }
                n += 1
        scan_log.append(f"{ist_now()} | market rules refreshed for {n} pairs")
    except Exception as e:
        scan_log.append(f"{ist_now()} | markets_details fail: {e}")

def fmt_price(pair, price):
    pp = PAIR_RULES.get(pair, {}).get("price_precision", 6)
    try: return float(f"{float(price):.{pp}f}")
    except: return float(price)

def fmt_qty(pair, qty):
    qp = PAIR_RULES.get(pair, {}).get("qty_precision", 6)
    mq = PAIR_RULES.get(pair, {}).get("min_qty", 0.0)
    try:
        q = max(float(qty), mq)
        return float(f"{q:.{qp}f}")
    except:
        return 0.0

def get_balances():
    payload = json.dumps({"timestamp": int(time.time()*1000)})
    headers = {"X-AUTH-APIKEY": API_KEY, "X-AUTH-SIGNATURE": hmac_signature(payload), "Content-Type":"application/json"}
    try:
        r = requests.post(f"{BASE_URL}/exchange/v1/users/balances", headers=headers, data=payload, timeout=10)
        if not r.ok:
            scan_log.append(f"{ist_now()} | balances HTTP {r.status_code}")
            return {}
        return {b["currency"]: float(b["balance"]) for b in r.json()}
    except Exception as e:
        scan_log.append(f"{ist_now()} | balances fail: {e}")
        return {}

def fetch_prices():
    try:
        r = requests.get(f"{BASE_URL}/exchange/ticker", timeout=10)
        if not r.ok:
            scan_log.append(f"{ist_now()} | ticker HTTP {r.status_code}")
            return {}
        d = {}
        for i in r.json():
            m = i.get("market")
            if m in PAIRS:
                try: d[m] = float(i["last_price"])
                except: pass
        return d
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
    res = _signed_post(f"{BASE_URL}/exchange/v1/orders/create", body)
    oid = str(res.get("id") or res.get("order_id") or res.get("client_order_id") or "")
    order_book.append({"time": ist_now(), "pair": pair, "side": side, "qty": qty, "id": oid or "?", "state": ("ok" if oid else "error"), "raw": res})
    trade_log.append(f"{ist_now()} | {pair} {side} {qty} | id={oid} | res={res}")
    return oid, res

# -------------------- Candles & Indicators --------------------
def update_candles(pair, ts, price):
    ticks = tick_logs[pair]
    ticks.append((ts, price))
    if len(ticks) > 2000: del ticks[:-1000]

    wstart = ts - (ts % CANDLE_INTERVAL)
    cl = candle_logs[pair]
    if not cl or cl[-1]["start"] != wstart:
        cl.append({"open": price, "high": price, "low": price, "close": price, "start": wstart})
    else:
        c = cl[-1]
        c["high"] = max(c["high"], price)
        c["low"]  = min(c["low"], price)
        c["close"]= price
    candle_logs[pair] = cl[-1000:]

def ema(values, n):
    if len(values) < n: return None
    k = 2/(n+1)
    e = sum(values[:n])/n
    for v in values[n:]:
        e = v*k + e*(1-k)
    return e

def macd(values):
    # Return full MACD with histogram based on last values
    if len(values) < max(MACD_SLOW, MACD_SIGNAL)+5:
        return None, None, None
    def _ema(seq, n):
        k=2/(n+1); e=sum(seq[:n])/n
        for v in seq[n:]: e = v*k + e*(1-k)
        return e
    efast = _ema(values, MACD_FAST)
    eslow = _ema(values, MACD_SLOW)
    line  = efast - eslow
    # quick n-step signal approx using last MACD_SIGNAL values as constant line (good enough here)
    sig = line
    k=2/(MACD_SIGNAL+1)
    for _ in range(MACD_SIGNAL):
        sig = line*k + sig*(1-k)
    hist = line - sig
    return line, sig, hist

def log_scan(pair, msg):
    scan_log.append(f"{ist_now()} | {pair} | {msg}")

# -------------------- Strategy Loop --------------------
def scan_loop():
    global running, status, status_epoch, last_keepalive
    running = True
    fetch_pair_precisions()
    log_scan("SYS", "loop started")

    while running:
        now = int(time.time())

        # self keepalive ping
        if now - last_keepalive >= KEEPALIVE_SEC:
            try:
                url = f"{APP_BASE_URL}/ping"
                hdr = {}
                if KEEPALIVE_TOKEN:
                    url += f"?t={KEEPALIVE_TOKEN}"
                    hdr = {"X-Keepalive-Token": KEEPALIVE_TOKEN}
                requests.get(url, headers=hdr, timeout=5)
                last_keepalive = now
                log_scan("SYS", "keepalive ping sent")
            except Exception as e:
                log_scan("SYS", f"keepalive ping failed: {e}")

        prices   = fetch_prices()
        balances = get_balances()
        usdt     = float(balances.get("USDT", 0.0))

        for pair in PAIRS:
            px = prices.get(pair)
            if not px:
                log_scan(pair, "no price; skip")
                continue

            update_candles(pair, now, px)
            closes = [c["close"] for c in candle_logs[pair]]

            # Always log indicator readiness
            if len(closes) < max(EMA_SLOW, MACD_SLOW, MACD_SIGNAL)+2:
                log_scan(pair, f"waiting data: closes={len(closes)} need>={max(EMA_SLOW, MACD_SLOW, MACD_SIGNAL)+2}")
                continue

            e5  = ema(closes, EMA_FAST)
            e20 = ema(closes, EMA_SLOW)
            m_line, m_sig, m_hist = macd(closes)

            ema_state = "bull" if e5 and e20 and e5 > e20 else ("bear" if e5 and e20 and e5 < e20 else "flat")
            macd_state = "bull" if (m_line is not None and m_sig is not None and m_line > m_sig) else ("bear" if (m_line is not None and m_sig is not None and m_line < m_sig) else "flat")

            # Divergence (very light): compare last 3 swing closes vs MACD line
            div = "none"
            if len(closes) >= 10 and m_line is not None:
                win = closes[-6:]
                p_up   = win[-1] > win[0]
                mac_up = m_line > 0
                if p_up and not mac_up: div = "bearish_div"
                if (not p_up) and mac_up: div = "bullish_div"

            log_scan(pair, f"px={px:.8f} ema5={e5:.8f} ema20={e20:.8f} ema_state={ema_state} "
                           f"macd={m_line:.6f} sig={m_sig:.6f} hist={m_hist:.6f} macd_state={macd_state} div={div}")

            have_pos = (pair in entries)
            gates_buy  = (ema_state=="bull" and macd_state=="bull")
            gates_sell = (ema_state=="bear" and macd_state=="bear")

            # BUY gate
            if not have_pos and gates_buy and usdt > 5:
                invest_usdt = usdt * INVEST_FRAC_USDT
                q = fmt_qty(pair, invest_usdt / px)
                if q > 0:
                    oid, res = place_market(pair, "BUY", q)
                    if oid:
                        entries[pair] = {"price": px, "qty": q, "ts": now}
                        log_scan(pair, f"BUY filled qty={q} entry={px:.8f} oid={oid}")
                    else:
                        log_scan(pair, f"BUY rejected qty={q} res={res}")
                else:
                    log_scan(pair, f"BUY qty too small; invest={invest_usdt}")

            # SELL gate – only if we have a position; exit on sell-signal OR 1% TP
            if have_pos:
                entry_px = entries[pair]["price"]
                gain = (px - entry_px) / max(1e-9, entry_px)
                tp_hit = (gain >= TP_TARGET)
                if gates_sell or tp_hit:
                    qty = fmt_qty(pair, float(balances.get(pair[:-4], 0.0)))
                    if qty > 0:
                        oid, res = place_market(pair, "SELL", qty)
                        if oid:
                            log_scan(pair, f"SELL filled qty={qty} px={px:.8f} entry={entry_px:.8f} gain={gain*100:.3f}% reason={'TP' if tp_hit else 'SELL-signal'} oid={oid}")
                            entries.pop(pair, None)
                        else:
                            log_scan(pair, f"SELL rejected qty={qty} res={res}")
                    else:
                        log_scan(pair, f"SELL skip — wallet {pair[:-4]} qty=0")

            # Decision summary each tick
            log_scan(pair, f"gates: BUY={'open' if gates_buy and not have_pos else 'closed'} "
                           f"SELL={'open' if (have_pos and (gates_sell or (pair in entries and (px-entries[pair]['price'])/entries[pair]['price']>=TP_TARGET))) else 'closed'} "
                           f"pos={'YES' if have_pos else 'NO'}")

        status["msg"], status["last"] = "Running", ist_now()
        global status_epoch
        status_epoch = int(time.time())
        time.sleep(POLL_SEC)

    status["msg"] = "Idle"
    log_scan("SYS", "loop stopped")

# -------------------- Routes --------------------
@app.route("/")
def index():
    return render_template("index.html")

@app.route("/start", methods=["POST"])
def start():
    global running
    if not running:
        t = threading.Thread(target=scan_loop, daemon=True)
        t.start()
    return jsonify({"ok": True, "status": "started"})

@app.route("/stop", methods=["POST"])
def stop():
    global running
    running = False
    return jsonify({"ok": True, "status": "stopped"})

@app.route("/status")
def get_status():
    # balances quick read (non-blocking if exchange slow)
    try:
        balances = get_balances()
    except:
        balances = {}

    now_real = time.time()
    last_age = now_real - (last_keepalive or 0)
    keepalive_info = {
        "enabled": bool(KEEPALIVE_TOKEN),
        "interval_sec": KEEPALIVE_SEC,
        "last_ping_age_sec": int(last_age) if last_keepalive else None,
        "next_due_sec": (max(0, int(KEEPALIVE_SEC - last_age)) if last_keepalive else None),
        "app_base_url": APP_BASE_URL,
    }

    # pretty orderbook summary (last 30)
    ob = list(order_book)[-30:]
    tr = list(trade_log)[-30:]
    logs = list(scan_log)[-120:]

    # positions view
    pos = {p: {"entry": v["price"], "qty": v["qty"], "ts": v["ts"]} for p, v in entries.items()}

    return jsonify({
        "status": status["msg"],
        "last": status["last"],
        "status_epoch": status_epoch,
        "balances": balances,
        "positions": pos,
        "trades": tr,
        "order_book": ob,
        "scans": logs,
        "keepalive": keepalive_info,
        "error": error_message
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

# -------------------- Boot --------------------
if __name__ == "__main__":
    fetch_pair_precisions()
    port = int(os.environ.get("PORT", "10000"))
    app.run(host="0.0.0.0", port=port)
