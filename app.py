import os
import json
import time
import hmac
import hashlib
from typing import Dict, List, Any

import requests
from flask import Flask, redirect, request, render_template, jsonify
from dotenv import load_dotenv

load_dotenv()

# ──────────────────────────────────────────────────────────────────────────────
# Config
# ──────────────────────────────────────────────────────────────────────────────
API_KEY = os.getenv("API_KEY", "").strip()
API_SECRET = os.getenv("API_SECRET", "").strip().encode()
BASE_URL = "https://api.coindcx.com"

# Universe: 5 major USDT pairs
PAIRS = ["BTCUSDT", "ETHUSDT", "BNBUSDT", "SOLUSDT", "DOGEUSDT"]

app = Flask(__name__, template_folder="templates")
smc_status_cache = {}

# ──────────────────────────────────────────────────────────────────────────────
# Minimal CoinDCX client
# ──────────────────────────────────────────────────────────────────────────────
class CoinDCXClient:
    def __init__(self, api_key: str, api_secret: bytes, base_url: str):
        if not api_key or not api_secret:
            raise RuntimeError("API_KEY and API_SECRET must be set in environment/.env")
        self.api_key = api_key
        self.api_secret = api_secret
        self.base_url = base_url

    def _sign(self, payload: dict) -> tuple[dict, str]:
        body = json.dumps(payload, separators=(",", ":"))
        signature = hmac.new(self.api_secret, body.encode(), hashlib.sha256).hexdigest()
        headers = {
            "X-AUTH-APIKEY": self.api_key,
            "X-AUTH-SIGNATURE": signature,
            "Content-Type": "application/json",
        }
        return headers, body

    def balances(self):
        url = f"{self.base_url}/exchange/v1/users/balances"
        payload = {"timestamp": int(time.time() * 1000)}
        headers, body = self._sign(payload)
        r = requests.post(url, headers=headers, data=body, timeout=10)
        r.raise_for_status()
        return r.json()

    def ticker(self):
        r = requests.get(f"{self.base_url}/exchange/ticker", timeout=10)
        r.raise_for_status()
        return r.json()

    def markets_details(self):
        r = requests.get(f"{self.base_url}/exchange/v1/markets_details", timeout=12)
        r.raise_for_status()
        return r.json()

client = CoinDCXClient(API_KEY, API_SECRET, BASE_URL)

# ──────────────────────────────────────────────────────────────────────────────
# Precision helpers
# ──────────────────────────────────────────────────────────────────────────────
def _pair_precisions(client: CoinDCXClient) -> Dict[str, int]:
    precisions: Dict[str, int] = {}
    try:
        md = client.markets_details()
        for row in md:
            p = row.get("pair")
            if p:
                try:
                    precisions[p] = int(row.get("target_currency_precision", 6))
                except Exception:
                    pass
    except Exception:
        pass
    return precisions

def _round_price(p, precision):
    try:
        return float(f"{float(p):.{precision}f}")
    except Exception:
        return float(p)

# ──────────────────────────────────────────────────────────────────────────────
# Lightweight SMC logic (candles & live)
# ──────────────────────────────────────────────────────────────────────────────
def ema(values: List[float], period: int) -> List[float]:
    """Simple EMA."""
    if period <= 1 or len(values) == 0:
        return values[:]
    k = 2 / (period + 1)
    out, ema_val = [], None
    for v in values:
        ema_val = v if ema_val is None else (v * k + ema_val * (1 - k))
        out.append(ema_val)
    return out

def psar_series(candles: List[Dict[str, float]], step=0.02, max_step=0.2) -> List[Dict[str, Any]]:
    """Classic Wilder PSAR computed on completed candles."""
    n = len(candles)
    if n < 2:
        return []
    psar = [None] * n
    bull = True
    af = step
    ep = candles[0]["high"]
    sar = candles[0]["low"]
    if candles[1]["close"] < candles[0]["close"]:
        bull = False
        ep = candles[0]["low"]
        sar = candles[0]["high"]
    psar[0] = sar
    for i in range(1, n):
        c_prev = candles[i - 1]
        c = candles[i]
        sar = sar + af * (ep - sar)
        if bull:
            sar = min(sar, c_prev["low"], c["low"])
        else:
            sar = max(sar, c_prev["high"], c["high"])
        reverse = False
        if bull and c["low"] < sar:
            reverse = True
        elif (not bull) and c["high"] > sar:
            reverse = True
        if reverse:
            bull = not bull
            sar = ep
            af = step
            if bull:
                ep = c["high"]
                sar = min(sar, c_prev["low"], c["low"])
            else:
                ep = c["low"]
                sar = max(sar, c_prev["high"], c["high"])
        else:
            if bull and c["high"] > ep:
                ep = c["high"]; af = min(af + step, max_step)
            elif (not bull) and c["low"] < ep:
                ep = c["low"];  af = min(af + step, max_step)
        psar[i] = sar
    return [{"sar": psar[i], "bull": candles[i]["close"] > psar[i]} for i in range(n)]

def swing_highs_lows(candles: List[Dict[str, float]], w: int = 2):
    sh, sl = [], []
    for i in range(w, len(candles) - w):
        hi, lo = candles[i]["high"], candles[i]["low"]
        if all(hi >= candles[j]["high"] for j in range(i - w, i + w + 1) if j != i):
            sh.append(i)
        if all(lo <= candles[j]["low"]  for j in range(i - w, i + w + 1) if j != i):
            sl.append(i)
    return sh, sl

def structure_bos_choch(candles: List[Dict[str, float]], sh: List[int], sl: List[int]):
    """Compact BOS/CHOCH based on last two swing highs/lows."""
    last_hi = sh[-2:] if len(sh) >= 2 else []
    last_lo = sl[-2:] if len(sl) >= 2 else []
    bos = None
    choch = None
    note = []

    if last_hi and last_lo:
        hh = candles[last_hi[-1]]["high"] > candles[last_hi[-2]]["high"] if len(last_hi) >= 2 else False
        ll = candles[last_lo[-1]]["low"]  < candles[last_lo[-2]]["low"]  if len(last_lo) >= 2 else False

        if hh and not ll:
            bos = "bullish"; note.append("BOS: HH")
        elif ll and not hh:
            bos = "bearish"; note.append("BOS: LL")

        if len(last_hi) >= 2 and len(last_lo) >= 2:
            choch = "bullish" if (last_hi[-1] > last_lo[-1]) else "bearish"
            note.append(f"CHOCH:{choch}")

    return bos, choch, ", ".join(note) if note else ""

def decide_signal(candles: List[Dict[str, float]], opts: Dict[str, Any]):
    """Combine EMA/PSAR/Structure for a simple BUY/SELL/HOLD decision."""
    closes = [c["close"] for c in candles]
    ema_fast = ema(closes, int(opts.get("ema_fast", 5)))
    ema_slow = ema(closes, int(opts.get("ema_slow", 10)))
    ps = psar_series(
        candles,
        step=float(opts.get("psar_step", 0.02)),
        max_step=float(opts.get("psar_max", 0.2)),
    )
    sh, sl = swing_highs_lows(candles, w=2)
    bos, choch, s_note = structure_bos_choch(candles, sh, sl)

    note = []
    bullish_ma = None
    if ema_fast and ema_slow and len(ema_fast) == len(closes) and len(ema_slow) == len(closes):
        bullish_ma = ema_fast[-1] > ema_slow[-1]
        note.append(f"EMA: {'bull' if bullish_ma else 'bear'}")

    ps_bull = ps[-1]["bull"] if ps else None
    if ps_bull is not None:
        note.append(f"PSAR: {'bull' if ps_bull else 'bear'}")

    if s_note:
        note.append(s_note)

    signal = "HOLD"
    if bullish_ma is True and ps_bull is True and bos == "bullish":
        signal = "BUY"
    elif bullish_ma is False and ps_bull is False and bos == "bearish":
        signal = "SELL"
    elif choch == "bullish" and ps_bull is True:
        signal = "BUY"
    elif choch == "bearish" and ps_bull is False:
        signal = "SELL"

    return signal, "; ".join(note)

def normalize_candle(c) -> Dict[str, float]:
    t = c.get("time") or c.get("ts") or c.get("date")
    return {
        "time": int(t) if t is not None else None,
        "open": float(c["open"]),
        "high": float(c["high"]),
        "low": float(c["low"]),
        "close": float(c["close"]),
        "volume": float(c.get("volume", 0.0)),
    }

def run_smc_scan_on_candles(
    candles_by_pair: Dict[str, List[Dict[str, float]]],
    pairs: List[str],
    options: Dict[str, Any] = None,
):
    """Full SMC on uploaded candles."""
    options = options or {}
    out = {
        "status": "ok",
        "timestamp": int(time.time()),
        "universe": list(pairs),
        "results": []
    }

    for p in pairs:
        arr = candles_by_pair.get(p, [])
        if not isinstance(arr, list) or len(arr) < 10:
            out["results"].append({
                "pair": p,
                "signal": "NA",
                "notes": "Insufficient candles (min 10)"
            })
            continue

        try:
            candles = [normalize_candle(x) for x in arr]
            candles.sort(key=lambda z: (z["time"] if z["time"] is not None else 0))
            sig, notes = decide_signal(candles, options)
            out["results"].append({
                "pair": p,
                "signal": sig,
                "notes": notes,
                "last_close": candles[-1]["close"],
                "last_time": candles[-1]["time"],
            })
        except Exception as e:
            out["results"].append({
                "pair": p,
                "signal": "ERR",
                "notes": f"error: {e}"
            })

    return out

def run_smc_scan_coindcx(client: CoinDCXClient, pairs: List[str]):
    """Live scan using /exchange/ticker (no candles)."""
    out = {"timestamp": int(time.time()), "status": "ok", "universe": list(pairs), "results": []}
    try:
        tick = client.ticker()
    except Exception as e:
        return {"status": "error", "error": f"ticker_error: {e}", "universe": list(pairs), "results": []}

    if not isinstance(tick, list) or not tick:
        out["status"] = "error"
        out["error"] = "ticker_unavailable"
        return out

    last_by_pair: Dict[str, float] = {}
    for row in tick:
        mkt = row.get("market")
        if mkt in pairs:
            try:
                price = float(row.get("last_price"))
                last_by_pair[mkt] = price
            except Exception:
                continue

    precisions = _pair_precisions(client)

    for p in pairs:
        px = last_by_pair.get(p)
        if px is None:
            out["results"].append({"pair": p, "price": None, "signal": "NA", "notes": "No price"})
            continue
        prec = precisions.get(p, 4)
        out["results"].append({
            "pair": p,
            "price": _round_price(px, prec),
            "signal": "HOLD",
            "notes": "SMC stub — send candles via /api/smc-scan for full logic."
        })
    return out

# ──────────────────────────────────────────────────────────────────────────────
# Routes
# ──────────────────────────────────────────────────────────────────────────────
@app.route("/")
def home():
    # Similar UX flow as the Kite app
    return redirect("/login")

@app.route("/login")
def login():
    # No OAuth needed; try balances to validate (non-fatal)
    try:
        client.balances()
    except Exception:
        pass
    return redirect("/dashboard")

@app.route("/dashboard")
def dashboard():
    # If template exists use it; else simple inline page
    try:
        return render_template("index.html", pairs=PAIRS)
    except Exception:
        html = f"""
        <html><head><title>CoinDCX SMC</title></head>
        <body style="font-family: system-ui; margin:20px">
          <h3>CoinDCX SMC Scanner</h3>
          <p>Endpoints:</p>
          <ul>
            <li><code>GET /api/smc-status</code> — live prices for {', '.join(PAIRS)}</li>
            <li><code>POST /api/smc-scan</code> — run SMC on uploaded candles</li>
            <li><code>GET /api/precisions</code> — target precisions for pairs</li>
            <li><code>GET /ping</code></li>
          </ul>
        </body></html>
        """
        return html

@app.route("/api/smc-status")
def api_smc_status():
    global smc_status_cache
    try:
        smc_status_cache = run_smc_scan_coindcx(client, PAIRS)
        return jsonify(smc_status_cache)
    except Exception as e:
        return jsonify({"status": "error", "error": str(e)}), 500

@app.route("/api/smc-scan", methods=["POST"])
def api_smc_scan():
    """
    JSON:
    {
      "pairs": ["BTCUSDT","ETHUSDT"],
      "candles": {
        "BTCUSDT": [{"time":..., "open":..., "high":..., "low":..., "close":..., "volume":...}, ...],
        "ETHUSDT": [ ... ]
      },
      "options": { "ema_fast":5, "ema_slow":10, "psar_step":0.02, "psar_max":0.2 }
    }
    """
    try:
        payload = request.get_json(force=True, silent=False) or {}
    except Exception:
        return jsonify({"status": "error", "error": "invalid_json"}), 400

    pairs = payload.get("pairs")
    candles = payload.get("candles")
    options = payload.get("options", {})

    if not isinstance(pairs, list) or not pairs:
        return jsonify({"status": "error", "error": "missing_pairs"}), 400
    if not isinstance(candles, dict) or not all(isinstance(candles.get(p), list) for p in pairs):
        return jsonify({"status": "error", "error": "missing_or_invalid_candles"}), 400

    try:
        res = run_smc_scan_on_candles(candles_by_pair=candles, pairs=pairs, options=options)
        return jsonify(res)
    except Exception as e:
        return jsonify({"status": "error", "error": str(e)}), 500

@app.route("/api/precisions")
def api_precisions():
    """Return {pair: target_currency_precision} for precision-aware UI formatting."""
    try:
        mp = _pair_precisions(client)
        filtered = {p: mp.get(p, 4) for p in PAIRS}
        return jsonify({"status": "ok", "precisions": filtered})
    except Exception as e:
        return jsonify({"status": "error", "error": str(e)}), 500

@app.route("/ping")
def ping():
    return "pong"

# ──────────────────────────────────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    port = int(os.environ.get("PORT", "10000"))
    app.run(host="0.0.0.0", port=port)
