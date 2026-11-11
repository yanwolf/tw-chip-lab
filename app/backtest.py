"""全市場回測：「低位單峰密集 → 帶量突破」。

訊號（第 t 日收盤後判斷，全部只用 t 日以前的資料）：
  1. 單峰密集：t−1 日 90% 籌碼集中度 ≤ conc_th
  2. 低位：t−1 日籌碼主峰 ≤ 近 250 日最高價 × low_pos
  3. 突破：t 日收盤站上 t−1 日籌碼上緣（p95 或 p85），且 t−1 日收盤還在上緣以下
  4. 帶量：t 日量 ≥ 20 日均量 × vol_mult
  5. 流動性：20 日均成交金額 ≥ min_value，收盤 ≥ min_price
進場：t+1 日開盤（開盤已近漲停則跳過，買不到）。
出場（收盤判斷，次日開盤執行）：
  - 停損：收盤 < max(訊號日籌碼下緣 p5, 進場價 × (1 − max_stop))
  - 移動停利：收盤 < 持有期最高收盤 × (1 − trail)
  - 時間：持有滿 max_hold 日
成本：手續費 0.1425% × 折扣（買賣各一次）＋證交稅 0.3%（賣）。

另做「事件研究」：訊號後 5/10/20/60 日報酬，對比同日全市場合格股票的平均，
超額報酬才是判斷這個訊號有沒有用的關鍵。
"""
import threading
import time
import traceback
from collections import defaultdict
from datetime import date

import numpy as np

from . import chips, db

DEFAULTS = {
    "start": "", "end": "",
    "markets": ["TWSE", "TPEX"],
    "warmup": 250,
    "decay": 1.0,
    "conc_th": 0.15,
    "low_pos": 0.80,
    "edge": "p95",
    "vol_mult": 1.5,
    "min_value": 20_000_000,
    "min_price": 10,
    "max_stop": 0.10,
    "trail": 0.15,
    "max_hold": 60,
    "fee_discount": 0.6,
    "skip_limit_up": True,
}
HORIZONS = (5, 10, 20, 60)
FEE = 0.001425
TAX = 0.003


def norm_params(p):
    out = dict(DEFAULTS)
    for k, v in (p or {}).items():
        if k not in DEFAULTS:
            continue
        d = DEFAULTS[k]
        try:
            if isinstance(d, bool):
                out[k] = bool(v)
            elif isinstance(d, int) and not isinstance(d, bool):
                out[k] = int(float(v))
            elif isinstance(d, float):
                out[k] = float(v)
            else:
                out[k] = v
        except (TypeError, ValueError):
            pass
    if out["edge"] not in ("p95", "p85"):
        out["edge"] = "p95"
    return out


def rolling_mean(x, w):
    c = np.cumsum(np.insert(x, 0, 0.0))
    out = np.full(len(x), np.nan)
    if len(x) >= w:
        out[w - 1:] = (c[w:] - c[:-w]) / w
    return out


def rolling_max(x, w):
    out = np.full(len(x), np.nan)
    from collections import deque
    dq = deque()
    for i, v in enumerate(x):
        while dq and x[dq[-1]] <= v:
            dq.pop()
        dq.append(i)
        if dq[0] <= i - w:
            dq.popleft()
        out[i] = x[dq[0]]
    return out


def run_stock(s, p, date_index):
    """單一股票：回傳 (trades, signals, fwd_signal, baseline_acc)。"""
    m = chips.compute(s, decay=p["decay"])
    n = len(s.c)
    c, o = s.c, s.o
    val20 = rolling_mean(s.val, 20)          # 原始成交金額

    vol20 = rolling_mean(s.v, 20)
    hi250 = rolling_max(s.h, 250)
    edge = m[p["edge"]]
    fee = FEE * p["fee_discount"]
    dates = s.dates
    start, end = p["start"] or "0000", p["end"] or "9999"

    eligible = np.zeros(n, dtype=bool)
    for i in range(max(p["warmup"], 20), n):
        eligible[i] = (val20[i] >= p["min_value"]) and (c[i] >= p["min_price"]) \
            and start <= dates[i] <= end

    # 事件研究基準：每個合格日的前瞻報酬（t+1 開盤買、t+k 收盤）
    base = []
    for k in HORIZONS:
        r = np.full(n, np.nan)
        if n > k + 1:
            r[:n - k] = c[k:] / np.r_[o[1:], np.nan][:n - k] - 1
        r[~eligible] = np.nan
        base.append(r)

    signals, trades, fwd = [], [], []
    hold_until = -1
    for t in range(max(p["warmup"], 60), n):
        if not eligible[t]:
            continue
        j = t - 1
        if np.isnan(m["conc90"][j]) or np.isnan(hi250[j]) or np.isnan(vol20[j]):
            continue
        if m["conc90"][j] > p["conc_th"]:
            continue
        if m["poc"][j] > hi250[j] * p["low_pos"]:
            continue
        if not (c[t] > edge[j] and c[j] <= edge[j]):
            continue
        if s.v[t] < vol20[j] * p["vol_mult"]:
            continue
        sig = {"code": s.code, "name": s.name, "market": s.market, "date": dates[t],
               "close": round(float(c[t]), 2), "conc90": round(float(m["conc90"][j]), 4),
               "poc": round(float(m["poc"][j]), 2), "edge": round(float(edge[j]), 2),
               "low": round(float(m["p5"][j]), 2),
               "vol_x": round(float(s.v[t] / vol20[j]), 2),
               "profit": round(float(m["profit"][t]), 3)}
        signals.append(sig)
        fwd.append((date_index.get(dates[t]), [float(b[t]) for b in base]))
        if t <= hold_until:
            sig["in_pos"] = True        # 已持有同一檔，不重複進場（仍列入事件研究）
            continue
        if t + 1 >= n:
            sig["pending"] = True       # 最新一天的訊號，明天開盤才會進場
            continue
        entry_i = t + 1
        entry = float(o[entry_i])
        if p["skip_limit_up"] and entry >= c[t] * 1.095:
            sig["skipped"] = "開盤近漲停"
            continue
        stop = max(float(m["p5"][j]), entry * (1 - p["max_stop"]))
        peak = entry
        exit_i, reason = None, None
        for i in range(entry_i, n):
            peak = max(peak, float(c[i]))
            why = None
            if c[i] < stop:
                why = "停損"
            elif c[i] < peak * (1 - p["trail"]):
                why = "移動停利"
            elif i - entry_i + 1 >= p["max_hold"]:
                why = "時間"
            if why:
                if i + 1 < n:
                    exit_i, reason = i + 1, why
                break
        if exit_i is None:
            px, reason, open_pos, exit_i = float(c[-1]), "持有中", True, n - 1
        else:
            px, open_pos = float(o[exit_i]), False
        net = px * (1 - fee - TAX) / (entry * (1 + fee)) - 1
        trades.append({**sig, "entry_date": dates[entry_i], "entry": round(entry, 2),
                       "stop": round(stop, 2), "exit_date": dates[exit_i],
                       "exit": round(px, 2), "reason": reason, "open": open_pos,
                       "days": int(exit_i - entry_i), "ret": round(net, 4)})
        hold_until = exit_i
    return trades, signals, fwd, base, eligible


def summarize(trades):
    closed = [t for t in trades if not t["open"]]
    r = np.array([t["ret"] for t in closed]) if closed else np.array([])
    if not len(r):
        return {"n": 0, "open": len(trades)}
    wins, losses = r[r > 0], r[r <= 0]
    pf = wins.sum() / -losses.sum() if losses.sum() < 0 else None
    return {"n": int(len(r)), "open": len(trades) - len(closed),
            "win": round(float((r > 0).mean()), 4),
            "avg": round(float(r.mean()), 4), "median": round(float(np.median(r)), 4),
            "avg_win": round(float(wins.mean()), 4) if len(wins) else None,
            "avg_loss": round(float(losses.mean()), 4) if len(losses) else None,
            "pf": round(float(pf), 3) if pf else None,
            "days": round(float(np.mean([t["days"] for t in closed])), 1),
            "best": round(float(r.max()), 4), "worst": round(float(r.min()), 4)}


class BacktestJob:
    def __init__(self):
        self.thread = None
        self.state = {"running": False, "msg": "閒置", "done": 0, "total": 0,
                      "error": None, "run_id": None}

    def running(self):
        return bool(self.thread and self.thread.is_alive())

    def start(self, params):
        if self.running():
            return False
        p = norm_params(params)
        self.thread = threading.Thread(target=self._run, args=(p,), daemon=True)
        self.thread.start()
        return True

    def _run(self, p):
        st = self.state
        st.update(running=True, error=None, done=0, run_id=None, msg="讀取資料")
        t0 = time.time()
        try:
            st["run_id"] = run(p, st)
            st["msg"] = f"完成（{time.time() - t0:.0f} 秒）"
        except Exception as e:
            traceback.print_exc()
            st["error"] = str(e)
            st["msg"] = f"失敗：{e}"
        finally:
            st["running"] = False


def run(p, st=None):
    st = st if st is not None else {}
    shares = db.load_shares()
    allc = [(c, n, m) for c, n, m in db.codes() if m in p["markets"]]
    st["total"] = len(allc)
    con = db.connect()
    all_dates = sorted({d for (d,) in con.execute("SELECT DISTINCT date FROM daily")})
    date_index = {d: i for i, d in enumerate(all_dates)}
    nd = len(all_dates)
    bsum = np.zeros((len(HORIZONS), nd))
    bcnt = np.zeros((len(HORIZONS), nd))
    trades, signals, fwd = [], [], []
    est = 0
    for k, (code, _, _) in enumerate(allc):
        st["done"] = k
        st["msg"] = f"計算 {code}（{k}/{len(allc)}）"
        s = chips.load(code, shares.get(code), con)
        if s is None:
            continue
        est += s.turn_est
        tr, sg, fw, base, elig = run_stock(s, p, date_index)
        trades += tr
        signals += sg
        fwd += fw
        idx = np.array([date_index[d] for d in s.dates])
        for h, b in enumerate(base):
            ok = ~np.isnan(b)
            np.add.at(bsum[h], idx[ok], b[ok])
            np.add.at(bcnt[h], idx[ok], 1)
    con.close()

    # 事件研究
    with np.errstate(invalid="ignore", divide="ignore"):
        bmean = bsum / bcnt
    event = []
    for h, k in enumerate(HORIZONS):
        sv, bv = [], []
        for di, rets in fwd:
            r = rets[h]
            if di is None or np.isnan(r) or np.isnan(bmean[h, di]):
                continue
            sv.append(r)
            bv.append(bmean[h, di])
        sv, bv = np.array(sv), np.array(bv)
        if len(sv):
            ex = sv - bv
            event.append({"h": k, "n": int(len(sv)),
                          "sig": round(float(sv.mean()), 4),
                          "base": round(float(bv.mean()), 4),
                          "excess": round(float(ex.mean()), 4),
                          "hit": round(float((ex > 0).mean()), 4),
                          "t": round(float(ex.mean() / (ex.std(ddof=1) / np.sqrt(len(ex)))), 2)
                          if len(ex) > 2 and ex.std() > 0 else None})
        else:
            event.append({"h": k, "n": 0})

    by_year = defaultdict(list)
    for t in trades:
        by_year[t["entry_date"][:4]].append(t)
    years = [{"year": y, **summarize(v)} for y, v in sorted(by_year.items())]
    reasons = defaultdict(list)
    for t in trades:
        if not t["open"]:
            reasons[t["reason"]].append(t["ret"])
    reason_stats = [{"reason": k, "n": len(v), "avg": round(float(np.mean(v)), 4)}
                    for k, v in reasons.items()]

    last_date = all_dates[-1] if all_dates else None
    latest = [s for s in signals if s["date"] == last_date]
    trades.sort(key=lambda t: t["entry_date"])
    summary = {**summarize(trades), "signals": len(signals),
               "skipped": sum(1 for s in signals if s.get("skipped")),
               "stocks": len(allc), "turn_est": est,
               "range": [p["start"] or (all_dates[0] if all_dates else ""),
                         p["end"] or (last_date or "")],
               "excess20": next((e.get("excess") for e in event if e["h"] == 20), None)}
    result = {"event": event, "years": years, "reasons": reason_stats,
              "trades": trades, "latest": latest, "last_date": last_date}
    st["msg"] = "儲存結果"
    return db.save_run(p, summary, result)


def chip_view(code, day=None, decay=1.0, bars=120):
    """單股籌碼圖資料：到指定日為止的 K 線＋該日籌碼分布。"""
    s = chips.load(code, db.load_shares().get(code))
    if s is None:
        return None
    i = len(s.dates) - 1
    if day:
        while i > 0 and s.dates[i] > day:
            i -= 1
    m = chips.compute(s, decay=decay, keep_dist_at=i)
    a = max(0, i - bars + 1)
    dist, grid = m["dist"], m["grid"]
    nz = np.nonzero(dist > dist.max() * 0.002)[0]
    lo, hi = (int(nz[0]), int(nz[-1]) + 1) if len(nz) else (0, len(grid))
    r2 = lambda x: [round(float(v), 2) for v in x]
    return {"code": s.code, "name": s.name, "market": s.market, "date": s.dates[i],
            "turn_est": s.turn_est, "events": s.events[-10:],
            "bars": {"d": s.dates[a:i + 1], "o": r2(s.o[a:i + 1]), "h": r2(s.h[a:i + 1]),
                     "l": r2(s.l[a:i + 1]), "c": r2(s.c[a:i + 1])},
            "grid": r2(grid[lo:hi]), "dist": [round(float(v), 5) for v in dist[lo:hi]],
            "m": {k: round(float(m[k][i]), 2 if k[0] == "p" and k != "profit" else 4) for k in
                  ("p5", "p15", "p50", "p85", "p95", "poc", "profit", "conc90", "conc70")}}


JOB = BacktestJob()
