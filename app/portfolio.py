"""大盤狀態（寬度、等權指數）與投組模擬。

大盤寬度：每天「收盤站上 N 日均線」的股票比例（全部普通股，還原價）。
等權指數：每天所有有成交股票的平均日報酬累乘，當作投組的比較基準。
兩者都只用當天收盤以前的資料，訊號在 t 日收盤判斷、t+1 開盤進場，不會偷看未來。

投組模擬：同時最多持有 max_pos 檔，每檔投入「當下總資產 ÷ max_pos」，
同一天訊號超過空位時依 rank 排序取前面的；每天用收盤價計算市值。
"""
import numpy as np

from . import chips

_CACHE = {}


def market_context(all_dates, codes, shares, con, ma=60, st=None):
    key = (len(all_dates), all_dates[-1] if all_dates else "", ma, tuple(sorted(codes)))
    if key in _CACHE:
        return _CACHE[key]
    n = len(all_dates)
    di = {d: i for i, d in enumerate(all_dates)}
    above, cnt = np.zeros(n), np.zeros(n)
    rsum, rcnt = np.zeros(n), np.zeros(n)
    for k, code in enumerate(codes):
        if st is not None:
            st["msg"] = f"{st.get('prefix') or ''} 計算大盤狀態（{k}/{len(codes)}）".strip()
        s = chips.load(code, shares.get(code), con)
        if s is None:
            continue
        idx = np.array([di[d] for d in s.dates])
        c = s.c
        if len(c) >= ma:
            cs = np.cumsum(np.insert(c, 0, 0.0))
            m = np.full(len(c), np.nan)
            m[ma - 1:] = (cs[ma:] - cs[:-ma]) / ma
            ok = ~np.isnan(m)
            np.add.at(above, idx[ok], (c[ok] > m[ok]).astype(float))
            np.add.at(cnt, idx[ok], 1)
        r = np.clip(c[1:] / c[:-1] - 1, -0.5, 0.5)
        np.add.at(rsum, idx[1:], r)
        np.add.at(rcnt, idx[1:], 1)
    with np.errstate(invalid="ignore", divide="ignore"):
        breadth = np.where(cnt >= min(50, max(1, len(codes) // 2)), above / cnt, np.nan)
        daily = np.where(rcnt > 0, rsum / rcnt, 0.0)
    ew = np.cumprod(1 + daily)
    out = {"breadth": breadth, "ew": ew}
    _CACHE.clear()
    _CACHE[key] = out
    return out


def simulate(trades, paths, all_dates, ew, max_pos=10, rank="conc"):
    """trades 與 paths 一一對應；paths[i] = (進場日索引, 每日收盤 ÷ 實際進場價 的陣列)。"""
    if not trades:
        return None
    di = {d: i for i, d in enumerate(all_dates)}
    n = len(all_dates)
    by_entry = {}
    for t, pth in zip(trades, paths):
        by_entry.setdefault(di[t["entry_date"]], []).append((t, pth))
    keyf = (lambda x: x[0]["conc90"]) if rank == "conc" else (lambda x: -x[0]["vol_x"])
    start = min(by_entry)
    cash, pos = 1.0, []
    eq = np.full(n, np.nan)
    taken = skipped = 0
    expo = []
    for d in range(start, n):
        # 1) 今天開盤出場
        keep = []
        for p in pos:
            if p["exit"] == d:
                cash += p["alloc"] * (1 + p["ret"])
            else:
                keep.append(p)
        pos = keep
        # 2) 今天開盤進場（用昨收市值估總資產）
        cands = sorted(by_entry.get(d, []), key=keyf)
        if cands:
            mv = sum(p["alloc"] * p["last"] for p in pos)
            total = cash + mv
            for t, (s0, path) in cands:
                if len(pos) >= max_pos:
                    skipped += 1
                    continue
                alloc = min(total / max_pos, cash)
                if alloc <= total / max_pos * 0.5:
                    skipped += 1
                    continue
                cash -= alloc
                ex = di[t["exit_date"]] if not t["open"] else n + 1
                pos.append({"alloc": alloc, "start": s0, "path": path, "exit": ex,
                            "ret": t["ret"], "last": 1.0})
                taken += 1
        # 3) 收盤計值
        mv = 0.0
        for p in pos:
            k = d - p["start"]
            if 0 <= k < len(p["path"]):
                p["last"] = float(p["path"][k])
            mv += p["alloc"] * p["last"]
        eq[d] = cash + mv
        expo.append(mv / eq[d] if eq[d] > 0 else 0)

    e = eq[start:]
    dates = all_dates[start:]
    b = ew[start:] / ew[start]
    peak = np.maximum.accumulate(e)
    dd = e / peak - 1
    bpeak = np.maximum.accumulate(b)
    from datetime import date
    yrs = max((date.fromisoformat(dates[-1]) - date.fromisoformat(dates[0])).days / 365.25, 1e-9)
    years = []
    for y in sorted({d[:4] for d in dates}):
        ix = [i for i, d in enumerate(dates) if d[:4] == y]
        i0 = ix[0] - 1 if ix[0] > 0 else 0
        i1 = ix[-1]
        years.append({"year": y,
                      "ret": round(float(e[i1] / (e[i0] if ix[0] > 0 else 1.0) - 1), 4),
                      "bench": round(float(b[i1] / b[i0] - 1), 4),
                      "mdd": round(float((e[i0:i1 + 1] / np.maximum.accumulate(e[i0:i1 + 1]) - 1).min()), 4)})
    step = max(1, len(e) // 400)
    sel = list(range(0, len(e), step))
    if sel[-1] != len(e) - 1:
        sel.append(len(e) - 1)
    return {"start": dates[0], "end": dates[-1],
            "total": round(float(e[-1] - 1), 4),
            "cagr": round(float(e[-1] ** (1 / yrs) - 1), 4),
            "mdd": round(float(dd.min()), 4),
            "bench_total": round(float(b[-1] - 1), 4),
            "bench_cagr": round(float(b[-1] ** (1 / yrs) - 1), 4),
            "bench_mdd": round(float((b / bpeak - 1).min()), 4),
            "taken": taken, "skipped": skipped,
            "exposure": round(float(np.mean(expo)), 3) if expo else 0,
            "max_pos": max_pos, "years": years,
            "curve": {"d": [dates[i] for i in sel],
                      "v": [round(float(e[i]), 4) for i in sel],
                      "b": [round(float(b[i]), 4) for i in sel]}}
