"""資料健康檢查：找出還原價或原始資料的異常。

全市場等權指數若出現每年數百 % 的漲幅，代表某些股票的日報酬被系統性高估，
最常見的原因是還原因子判斷錯誤（漲跌方向解析錯、除權息日誤判）。
這裡分上市／上櫃、成交額高低統計：每年等權報酬、上漲比例、還原事件數量，並列出最極端的日報酬。
"""
import heapq

import numpy as np

from . import chips, db


def run(min_value=20_000_000):
    shares = db.load_shares()
    con = db.connect()
    all_dates = [d for (d,) in con.execute("SELECT DISTINCT date FROM daily ORDER BY date")]
    di = {d: i for i, d in enumerate(all_dates)}
    nd = len(all_dates)
    grp = {}                                   # (mkt, liq) -> dict of arrays
    ev = {}
    top = []
    for code, name, mkt in db.codes():
        s = chips.load(code, shares.get(code), con)
        if s is None:
            continue
        e = ev.setdefault(mkt, {"stocks": 0, "years": 0.0, "events": 0, "down": 0})
        e["stocks"] += 1
        e["years"] += len(s.dates) / 245
        e["events"] += len(s.events)
        e["down"] += sum(1 for _, f in s.events if f < 1)
        if len(s.c) < 2:
            continue
        r = s.c[1:] / s.c[:-1] - 1
        rr = s.craw[1:] / s.craw[:-1] - 1
        cs = np.cumsum(np.insert(s.val, 0, 0.0))
        k = np.arange(len(s.val))
        val20 = (cs[k + 1] - cs[np.maximum(k - 19, 0)]) / np.minimum(k + 1, 20)
        idx = np.array([di[d] for d in s.dates[1:]])
        liq = val20[:-1] >= min_value
        x = np.clip(r, -0.5, 0.5)
        for name_, m in (("liquid", liq), ("thin", ~liq)):
            g = grp.setdefault((mkt, name_), {a: np.zeros(nd) for a in ("sum", "n", "pos", "big")})
            np.add.at(g["sum"], idx[m], x[m])
            np.add.at(g["n"], idx[m], 1)
            np.add.at(g["pos"], idx[m], (r[m] > 0).astype(float))
            np.add.at(g["big"], idx[m], (np.abs(r[m]) > 0.11).astype(float))
        for i in np.argsort(-np.abs(r))[:3]:
            item = (abs(float(r[i])), code, s.name, mkt, s.dates[i + 1], round(float(s.craw[i]), 2),
                    round(float(s.craw[i + 1]), 2), round(float(r[i]), 4), round(float(rr[i]), 4))
            if len(top) < 20:
                heapq.heappush(top, item)
            elif item[0] > top[0][0]:
                heapq.heapreplace(top, item)
    con.close()

    yrs_of = np.array([d[:4] for d in all_dates])
    years = []
    for (mkt, liq), g in sorted(grp.items()):
        for y in sorted(set(yrs_of)):
            m = (yrs_of == y) & (g["n"] > 0)
            n = g["n"][m].sum()
            if not n:
                continue
            dm = g["sum"][m] / g["n"][m]
            years.append({"market": mkt, "liq": liq, "year": y, "n": int(n),
                          "mean": round(float(g["sum"][m].sum() / n), 5),
                          "up": round(float(g["pos"][m].sum() / n), 3),
                          "big": int(g["big"][m].sum()),
                          "ew": round(float(np.prod(1 + dm) - 1), 4)})
    events = [{"market": m, "stocks": v["stocks"],
               "per_year": round(v["events"] / v["years"], 2) if v["years"] else None,
               "events": v["events"],
               "down_share": round(v["down"] / v["events"], 3) if v["events"] else None}
              for m, v in sorted(ev.items())]
    extremes = [{"code": c, "name": n, "market": m, "date": d, "prev_raw": p0, "raw": p1,
                 "adj_ret": a, "raw_ret": b}
                for _, c, n, m, d, p0, p1, a, b in sorted(top, reverse=True)]
    return {"years": years, "events": events, "extremes": extremes, "min_value": min_value}
