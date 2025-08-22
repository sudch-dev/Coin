# app.py — compact EMA crossover (TP 0.5%, no SL), USDT-only, real trades, keepalive
import os, time, json, hmac, hashlib, threading, requests
from flask import Flask, render_template, jsonify
from datetime import datetime
from pytz import timezone
from collections import defaultdict, deque
from requests.adapters import HTTPAdapter
try:
    from urllib3.util.retry import Retry
except Exception:
    from requests.packages.urllib3.util.retry import Retry

# ---------- Flask ----------
app = Flask(__name__)

# ---------- Creds / API ----------
API_KEY = os.environ.get("API_KEY", "")
API_SECRET = (os.environ.get("API_SECRET", "") or "").encode()
BASE_URL = "https://api.coindcx.com"

# ---------- Keepalive ----------
APP_BASE_URL  = os.environ.get("APP_BASE_URL", "")  # e.g. https://your-app.onrender.com
KEEPALIVE_SEC = 240
_last_keepalive = 0
def _keepalive_ping():
    if not APP_BASE_URL: return
    try: requests.get(f"{APP_BASE_URL}/ping", timeout=5)
    except Exception: pass

# ---------- Timing / knobs ----------
IST = timezone("Asia/Kolkata")
def ist_now(): return datetime.now(IST).strftime("%Y-%m-%d %H:%M:%S")

CANDLE_INTERVAL = 5
LOOP_SEC        = 1.0
RULES_REFRESH_SEC = 30*24*3600   # 30 days
BAL_TTL_SEC     = 15

# Sizing / policy (kept same)
RISK_PER_TRADE   = 0.01   # 1% risk proxy
BUY_NOTIONAL_CAP = 0.30   # cap vs free USDT
TP_PCT           = 0.005  # +0.5%/-0.5% TP
SHORT_SELL_FRACTION = 0.30  # if shorting from inventory, sell up to 30% of coin balance

START_WITH_USDT = True                 # if True, sell non-USDT coins at boot (limits shorting)
FORCE_LIQUIDATE_ON_NEED = True         # free USDT before a long entry if required

# ---------- State ----------
PAIR_RULES = {}              # pair -> {price_precision, qty_precision, min_qty, min_notional}
ALL_USDT_PAIRS = []          # discovered USDT pairs
tick_logs   = defaultdict(list)
candle_logs = defaultdict(list)
# positions[pair] = {"side":"LONG"/"SHORT","qty":float,"entry":float,"tp":float}
positions   = {}
scan_log, trade_log = [], []
running = False
status = {"msg":"Idle", "last":""}
status_epoch = 0
_last_rules_refresh = 0
_bal_cache, _bal_ts = {}, 0

# ---------- Optional raw I/O for /io ----------
io_log = deque(maxlen=300)
def io(line: str):
    try: io_log.append(f"{ist_now()} | {line}")
    except Exception: pass

# ---------- HTTP session (retries) ----------
SESSION = requests.Session()
SESSION.headers.update({"User-Agent":"CoinDCXBot/3.0","Accept":"application/json","Connection":"keep-alive"})
adapter = HTTPAdapter(max_retries=Retry(total=5, connect=5, read=5, backoff_factor=0.4,
                                        status_forcelist=(429,500,502,503,504,520),
                                        allowed_methods=False, raise_on_status=False))
SESSION.mount("https://", adapter); SESSION.mount("http://", adapter)

# ---------- Utils ----------
def hmac_signature(payload_str: str) -> str:
    return hmac.new(API_SECRET, payload_str.encode(), hashlib.sha256).hexdigest()

def _signed_post(url, body, timeout=12):
    payload = json.dumps(body, separators=(',',':'))
    headers = {"X-AUTH-APIKEY": API_KEY, "X-AUTH-SIGNATURE": hmac_signature(payload), "Content-Type":"application/json"}
    try:
        io(f"REQ {url} | {payload[:220]}")
        r = SESSION.post(url, headers=headers, data=payload, timeout=timeout)
        io(f"RES {url} | {r.status_code} | {r.text[:220] if hasattr(r,'text') else ''}")
        if r.ok and r.headers.get("content-type","").startswith("application/json"):
            return r.json()
    except Exception as e:
        scan_log.append(f"{ist_now()} | POST fail {url.split('/exchange',1)[-1]}: {e.__class__.__name__} {e}")
        io(f"ERR {url} | {e}")
    return {}

def _http_get(url, timeout=10):
    try:
        io(f"GET {url}")
        r = SESSION.get(url, timeout=timeout)
        io(f"GOT {url} | {r.status_code}")
        if r.ok: return r
        scan_log.append(f"{ist_now()} | GET {url.split('/exchange',1)[-1]} HTTP {r.status_code}")
    except Exception as e:
        scan_log.append(f"{ist_now()} | GET fail {url.split('/exchange',1)[-1]}: {e.__class__.__name__} {e}")
        io(f"ERR {url} | {e}")
    return None

# ---------- Market rules / balances / prices ----------
def refresh_markets_and_pairs():
    global PAIR_RULES, ALL_USDT_PAIRS
    r = _http_get(f"{BASE_URL}/exchange/v1/markets_details", timeout=15)
    if not r: return
    rules, usdt_pairs = {}, []
    for it in r.json():
        p = it.get("pair") or it.get("market") or it.get("coindcx_name")
        if not p: continue
        rules[p] = {
            "price_precision": int(it.get("target_currency_precision", 6)),
            "qty_precision":   int(it.get("base_currency_precision", 6)),
            "min_qty":         float(it.get("min_quantity", 0) or 0.0),
            "min_notional":    float(it.get("min_notional", 0) or 0.0)
        }
        if p.endswith("USDT"): usdt_pairs.append(p)
    PAIR_RULES = rules
    ALL_USDT_PAIRS = sorted(usdt_pairs)
    scan_log.append(f"{ist_now()} | market rules refreshed | USDT pairs={len(ALL_USDT_PAIRS)}")

def get_wallet_balances():
    global _bal_cache, _bal_ts
    now = time.time()
    if _bal_cache and (now - _bal_ts) < BAL_TTL_SEC:
        return dict(_bal_cache)
    body = {"timestamp": int(now*1000)}
    res = _signed_post(f"{BASE_URL}/exchange/v1/users/balances", body, timeout=12)
    balances = {}
    try:
        for b in res or []:
            balances[b["currency"]] = float(b["balance"])
        _bal_cache, _bal_ts = balances, now
    except Exception:
        pass
    return balances

def fetch_all_prices():
    r = _http_get(f"{BASE_URL}/exchange/ticker", timeout=7)
    if not r: return {}
    now = int(time.time())
    ret = {}
    for it in r.json():
        m = it.get("market")
        if m in PAIR_RULES and m.endswith("USDT"):
            ret[m] = {"price": float(it["last_price"]), "ts": now}
    return ret

# ---------- Precision ----------
def _rules(pair): return PAIR_RULES.get(pair, {})
def _pp(pair): return int(_rules(pair).get("price_precision", 6))
def _qp(pair): return int(_rules(pair).get("qty_precision", 6))
def _min_qty(pair): return float(_rules(pair).get("min_qty", 0.0))
def _min_notional(pair): return float(_rules(pair).get("min_notional", 0.0))
def fmt_price(pair, px): return float(f"{float(px):.{_pp(pair)}f}")
def fmt_qty_floor(pair, qty):
    qp = _qp(pair); step = 10**(-qp)
    q = max(0.0, float(qty))
    q = (int(q/step))*step
    q = float(f"{q:.{qp}f}")
    return q if q >= _min_qty(pair) else 0.0

# ---------- Orders ----------
def place_market(pair, side, qty):
    qty = fmt_qty_floor(pair, qty)
    if qty <= 0:
        scan_log.append(f"{ist_now()} | {pair} | ABORT {side}: qty too small")
        return {}
    payload = {"market": pair, "side": side.lower(), "order_type": "market_order",
               "total_quantity": f"{qty}", "timestamp": int(time.time()*1000)}
    trade_log.append(f"{ist_now()} | {pair} | SUBMIT {side} qty={qty} @ MKT")
    res = _signed_post(f"{BASE_URL}/exchange/v1/orders/create", payload, timeout=10) or {}
    trade_log.append(f"{ist_now()} | {pair} | SUBMIT RESP => {res}")
    return res

def get_order_status(order_id=None, client_order_id=None):
    body = {"timestamp": int(time.time()*1000)}
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
    except Exception: pass
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

# ---------- Candles / EMA ----------
def aggregate_candles(pair):
    t = tick_logs[pair]
    if not t: return
    candles, candle, lastw = [], None, None
    for ts, px in sorted(t, key=lambda x: x[0]):
        w = ts - (ts % CANDLE_INTERVAL)
        if w != lastw:
            if candle: candles.append(candle)
            candle = {"open": px, "high": px, "low": px, "close": px, "volume": 1, "start": w}
            lastw = w
        else:
            candle["high"] = max(candle["high"], px); candle["low"] = min(candle["low"], px)
            candle["close"] = px; candle["volume"] += 1
    if candle: candles.append(candle)
    candle_logs[pair] = candles[-300:]

def _ema_series(vals, n):
    if len(vals)==0: return []
    k=2/(n+1); out=[]; e=vals[0]
    for v in vals: e=v*k+e*(1-k); out.append(e)
    return out

# ---------- Sizing helpers ----------
def size_for_long(px, usdt_free):
    # proxy risk = TP distance; position size bounded by RISK_PER_TRADE and BUY_NOTIONAL_CAP
    if px<=0: return 0.0
    risk_amt = max(0.0, RISK_PER_TRADE*usdt_free)
    stop_dist = px*TP_PCT if TP_PCT>0 else px*0.005
    if stop_dist<=0: return 0.0
    qty_risk = risk_amt/stop_dist
    qty_cap  = (BUY_NOTIONAL_CAP*usdt_free)/px if usdt_free>0 else 0.0
    return min(qty_risk, qty_cap)

def size_for_short_from_inventory(coin_bal):
    return max(0.0, SHORT_SELL_FRACTION * float(coin_bal or 0.0))

# ---------- Capital policy (strict USDT) ----------
def liquidate_all_non_usdt(reason="boot"):
    bal=get_wallet_balances()
    if not bal: return
    prices=fetch_all_prices()
    sold=skipped=0
    for cur,amt in bal.items():
        if cur=="USDT" or amt<=0: continue
        pair=f"{cur}USDT"
        if pair not in PAIR_RULES: 
            skipped+=1; continue
        last_px=float(prices.get(pair,{}).get("price",0.0))
        q=fmt_qty_floor(pair, amt)
        if q<=0: 
            skipped+=1; continue
        if last_px and _min_notional(pair)>0 and last_px*q<_min_notional(pair):
            skipped+=1; continue
        trade_log.append(f"{ist_now()} | {pair} | FORCE-LIQ {reason} | SELL {q} @ MKT")
        res=place_market(pair,"SELL",q); oid=_extract_order_id(res)
        if oid:
            st=get_order_status(order_id=oid); filled,px=_filled_avg_from_status(st)
            trade_log.append(f"{ist_now()} | {pair} | LIQ status oid={oid} filled={filled} px={px}")
        sold+=1
    scan_log.append(f"{ist_now()} | LIQ summary: sold={sold}, skipped(no USDT market/too small)={skipped}")

def ensure_usdt_liquidity(usdt_needed):
    if not FORCE_LIQUIDATE_ON_NEED: return True
    bal=get_wallet_balances(); have=float(bal.get("USDT",0.0))
    if have>=usdt_needed: return True
    # close open LONGs first
    for pair,p in list(positions.items()):
        if p.get("side")!="LONG": continue
        q=fmt_qty_floor(pair, p["qty"])
        if q<=0: continue
        res=place_market(pair,"SELL",q); oid=_extract_order_id(res)
        if oid:
            st=get_order_status(order_id=oid); filled,px=_filled_avg_from_status(st)
            trade_log.append(f"{ist_now()} | {pair} | LIQ-EXIT LONG filled={filled} px={px}")
            positions.pop(pair,None)
        bal=get_wallet_balances(); have=float(bal.get("USDT",0.0))
        if have>=usdt_needed: return True
    # sell stray coins (strict)
    bal=get_wallet_balances(); prices=fetch_all_prices()
    for cur,amt in bal.items():
        if cur=="USDT" or amt<=0: continue
        pair=f"{cur}USDT"
        if pair not in PAIR_RULES: continue
        last_px=float(prices.get(pair,{}).get("price",0.0))
        q=fmt_qty_floor(pair, amt)
        if q<=0: continue
        if last_px and _min_notional(pair)>0 and last_px*q<_min_notional(pair): continue
        place_market(pair,"SELL",q)
        bal=get_wallet_balances(); have=float(bal.get("USDT",0.0))
        if have>=usdt_needed: return True
    return False

# ---------- Strategy tick (EMA crossover only) ----------
def strategy_tick():
    prices=fetch_all_prices(); now=int(time.time())
    # ticks & candles
    for pair,obj in prices.items():
        px=obj["price"]; tick_logs[pair].append((now,px))
        if len(tick_logs[pair])>6000: tick_logs[pair]=tick_logs[pair][-6000:]
        aggregate_candles(pair)

    balances=get_wallet_balances(); usdt_free=float(balances.get("USDT",0.0))
    any_signal=False

    for pair, cs in list(candle_logs.items()):
        if len(cs)<20: continue
        closes=[c["close"] for c in cs]
        ema5=_ema_series(closes,5)
        ema13=_ema_series(closes,13)
        if len(ema5)<2 or len(ema13)<2: continue
        e5_prev,e5_now = ema5[-2], ema5[-1]
        e13_prev,e13_now = ema13[-2], ema13[-1]
        last = closes[-1]

        pos = positions.get(pair)

        # --- LONG entry: EMA5 crosses above EMA13 ---
        if (pos is None) and (e5_prev <= e13_prev and e5_now > e13_now):
            any_signal=True
            # ensure capital
            # risk proxy = TP distance, target notional = min(cap, risk based)
            need_notional = BUY_NOTIONAL_CAP*usdt_free
            if ensure_usdt_liquidity(need_notional):
                balances=get_wallet_balances(); usdt_free=float(balances.get("USDT",0.0))
                qty=size_for_long(last, usdt_free); qty=fmt_qty_floor(pair, qty)
                if qty>0:
                    tp=fmt_price(pair, last*(1.0+TP_PCT))
                    res=place_market(pair,"BUY",qty); oid=_extract_order_id(res)
                    if oid:
                        st=get_order_status(order_id=oid); filled,avg=_filled_avg_from_status(st)
                        if filled>0 and avg>0:
                            positions[pair]={"side":"LONG","qty":filled,"entry":avg,"tp":tp}
                            trade_log.append(f"{ist_now()} | {pair} | BUY {filled} @ {avg} | TP={tp} | oid={oid}")
                else:
                    scan_log.append(f"{ist_now()} | {pair} | LONG qty too small")

        # --- SHORT entry: EMA5 crosses below EMA13 (needs coin inventory) ---
        elif (pos is None) and (e5_prev >= e13_prev and e5_now < e13_now):
            any_signal=True
            coin = pair[:-4]
            coin_bal = float(balances.get(coin, 0.0))
            qty = fmt_qty_floor(pair, size_for_short_from_inventory(coin_bal))
            if qty>0:
                tp=fmt_price(pair, last*(1.0-TP_PCT))
                res=place_market(pair,"SELL",qty); oid=_extract_order_id(res)
                if oid:
                    st=get_order_status(order_id=oid); filled,avg=_filled_avg_from_status(st)
                    if filled>0 and avg>0:
                        positions[pair]={"side":"SHORT","qty":filled,"entry":avg,"tp":tp}
                        trade_log.append(f"{ist_now()} | {pair} | SHORT SELL {filled} @ {avg} | TP={tp} | oid={oid}")
            else:
                scan_log.append(f"{ist_now()} | {pair} | SHORT skipped (no coin inventory)")

        # --- Exit LONG at TP ---
        elif pos and pos.get("side")=="LONG":
            if last >= pos["tp"]:
                q=fmt_qty_floor(pair, pos["qty"])
                if q>0:
                    res=place_market(pair,"SELL",q); oid=_extract_order_id(res)
                    if oid:
                        st=get_order_status(order_id=oid); filled,avg=_filled_avg_from_status(st)
                        trade_log.append(f"{ist_now()} | {pair} | SELL {filled} @ {avg} | oid={oid}")
                positions.pop(pair, None)

        # --- Exit SHORT at TP (buy back lower) ---
        elif pos and pos.get("side")=="SHORT":
            if last <= pos["tp"]:
                q=fmt_qty_floor(pair, pos["qty"])
                if q>0:
                    res=place_market(pair,"BUY",q); oid=_extract_order_id(res)
                    if oid:
                        st=get_order_status(order_id=oid); filled,avg=_filled_avg_from_status(st)
                        trade_log.append(f"{ist_now()} | {pair} | BUYBACK {filled} @ {avg} | oid={oid}")
                positions.pop(pair, None)

    if not any_signal:
        scan_log.append(f"{ist_now()} | no entry")

# ---------- Main loop ----------
def scan_loop():
    global running, status_epoch, _last_rules_refresh, _last_keepalive
    scan_log.clear(); running=True
    while running:
        try:
            # keepalive
            if APP_BASE_URL and (time.time() - _last_keepalive) >= KEEPALIVE_SEC:
                _keepalive_ping(); _last_keepalive = time.time()
            # rules
            if time.time()-_last_rules_refresh>=RULES_REFRESH_SEC or not PAIR_RULES:
                refresh_markets_and_pairs(); _last_rules_refresh=time.time()
            strategy_tick()
            status["msg"]="Running"; status["last"]=ist_now(); status_epoch=int(time.time())
            time.sleep(LOOP_SEC)
        except Exception as e:
            scan_log.append(f"{ist_now()} | loop err: {e.__class__.__name__} {e}")
            time.sleep(2)

# ---------- Routes ----------
@app.route("/")
def index(): return render_template("index.html")

@app.route("/start", methods=["POST"])
def start():
    global running
    if not running:
        threading.Thread(target=scan_loop, daemon=True).start()
    running=True
    return jsonify({"status":"started"})

@app.route("/stop", methods=["POST"])
def stop():
    global running; running=False
    return jsonify({"status":"stopped"})

@app.route("/status")
def get_status():
    bal = get_wallet_balances() or {}
    coins = {}
    if ALL_USDT_PAIRS:
        for p in ALL_USDT_PAIRS[:50]:
            c = p[:-4]
            coins[c] = float(bal.get(c, 0.0))
    return jsonify({
        "status": status.get("msg","Idle"),
        "last": status.get("last",""),
        "status_epoch": status_epoch,
        "usdt": float(bal.get("USDT", 0.0)),
        "positions": positions or {},
        "coins": coins or {},
        "trades": trade_log[-20:][::-1] if trade_log else [],
        "scans":  scan_log[-100:][::-1] if scan_log else [],
    })

@app.route("/io")
def get_io():
    return jsonify(list(io_log))   # optional Raw I/O feed

@app.route("/ping")
def ping(): return "pong"

# ---------- Boot ----------
def boot():
    refresh_markets_and_pairs()
    if START_WITH_USDT:
        liquidate_all_non_usdt("boot")

if __name__ == "__main__":
    boot()
    app.run(host="0.0.0.0", port=int(os.environ.get("PORT","10000")))
