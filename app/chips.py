"""籌碼分布（成本分布 / CYQ）模型。

每日更新：
    舊籌碼 × (1 − 換手率×衰減係數) ＋ 當日新籌碼 × (換手率×衰減係數)
當日新籌碼以三角形分布攤在 [最低, 最高]，峰值在當日均價（成交金額 ÷ 成交股數）。
這就是通達信 CYQ 的做法；它是「推估」，不是真實持股資料。

價格先做「還原」（向前調整除權息、減資）：
證交所與櫃買的漲跌是相對「參考價」計算，所以 參考價 = 收盤 − 漲跌；
參考價 ≠ 前一交易日收盤 的那天就是除權息／減資日，調整因子 = 參考價 ÷ 前收。
"""
import numpy as np

from . import db

NBINS = 300


class Series:
    __slots__ = ("code", "name", "market", "dates", "o", "h", "l", "c", "v", "amt",
                 "val", "craw", "shares", "turn", "turn_est", "events")


def load(code, shares=None, con=None):
    """讀取並整理單一股票：去掉無成交日、還原價、換手率。資料不足回傳 None。"""
    rows = db.series(code, con)
    rows = [r for r in rows if r[4] and r[5] and r[1] and r[2] and r[3] and r[5] > 0]
    if len(rows) < 30:
        return None
    n = len(rows)
    raw = np.array([[r[1], r[2], r[3], r[4]] for r in rows], dtype=float)
    vol = np.array([r[5] for r in rows], dtype=float)
    amt = np.array([r[6] or 0 for r in rows], dtype=float)
    chg = [r[7] for r in rows]

    factor = np.ones(n)          # factor[i]：第 i 天（含）之前價格要乘的因子
    events = []
    for i in range(1, n):
        if chg[i] is None:
            continue
        ref = raw[i, 3] - chg[i]
        prev = raw[i - 1, 3]
        if ref > 0 and prev > 0:
            f = ref / prev
            if abs(f - 1) > 0.002 and 0.05 < f < 20:
                factor[:i] *= f
                events.append((rows[i][0], round(f, 4)))

    s = Series()
    s.code = code
    s.name = rows[-1][8]
    s.market = rows[-1][9]
    s.dates = [r[0] for r in rows]
    adj = raw * factor[:, None]
    s.o, s.h, s.l, s.c = adj[:, 0], adj[:, 1], adj[:, 2], adj[:, 3]
    # 均價：成交金額 ÷ 成交股數（原始價），再乘還原因子
    avg_raw = np.where(amt > 0, amt / np.maximum(vol, 1), raw[:, 3])
    s.amt = np.clip(avg_raw * factor, s.l, s.h)          # 這裡存「還原均價」
    s.v = vol
    s.val = np.where(amt > 0, amt, raw[:, 3] * vol)       # 原始成交金額
    s.craw = raw[:, 3]                                   # 原始收盤（算市值用）
    s.shares = shares if shares and shares > 0 else None
    if shares and shares > 0:
        s.turn = np.clip(vol / shares, 0, 1)
        s.turn_est = False
    else:                         # 沒有股本：假設中位數換手 1%
        med = np.median(vol[vol > 0]) if np.any(vol > 0) else 1
        s.turn = np.clip(0.01 * vol / med, 0, 1)
        s.turn_est = True
    s.events = events
    return s


def grid_for(s):
    lo = float(np.min(s.l)) * 0.9
    hi = float(np.max(s.h)) * 1.1
    return np.geomspace(max(lo, 0.01), hi, NBINS)


def _new_chips(grid, lo, hi, avg):
    """當日新籌碼：[lo, hi] 三角形分布，峰在 avg。"""
    if hi - lo < 1e-9:
        w = np.zeros(len(grid))
        w[int(np.argmin(np.abs(grid - avg)))] = 1.0
        return w
    left = np.where(avg > lo, (grid - lo) / max(avg - lo, 1e-12), 1.0)
    right = np.where(hi > avg, (hi - grid) / max(hi - avg, 1e-12), 1.0)
    w = np.where(grid <= avg, left, right)
    w = np.where((grid >= lo) & (grid <= hi), np.clip(w, 0, None), 0.0)
    # 讓區間內至少有小權重（避免三角形端點落在 bin 之間全為 0）
    w = np.where((grid >= lo) & (grid <= hi), w + 0.05, 0.0)
    t = w.sum()
    if t <= 0:
        w = np.zeros(len(grid))
        w[int(np.argmin(np.abs(grid - avg)))] = 1.0
        return w
    return w / t


QS = np.array([0.05, 0.15, 0.50, 0.85, 0.95])


def compute(s, decay=1.0, keep_dist_at=None):
    """逐日算籌碼指標。回傳 dict of arrays（長度 = 交易日數）。

    p5/p15/p50/p85/p95：籌碼分位價；poc：最大峰價位；profit：獲利比例（現價以下籌碼占比）；
    conc90 / conc70：集中度 =（上緣−下緣）÷（上緣＋下緣），越小越集中。
    keep_dist_at：指定索引時，另外回傳該日的分布（畫圖用）。
    """
    grid = grid_for(s)
    n = len(s.c)
    out = {k: np.full(n, np.nan) for k in
           ("p5", "p15", "p50", "p85", "p95", "poc", "profit", "conc90", "conc70")}
    dist = None
    kept = None
    for i in range(n):
        w = _new_chips(grid, s.l[i], s.h[i], s.amt[i])
        if dist is None:
            dist = w
        else:
            t = min(float(s.turn[i]) * decay, 1.0)
            dist = dist * (1 - t) + w * t
        cs = np.cumsum(dist)
        tot = cs[-1]
        idx = np.searchsorted(cs, QS * tot)
        idx = np.clip(idx, 0, NBINS - 1)
        p5, p15, p50, p85, p95 = grid[idx]
        out["p5"][i], out["p15"][i], out["p50"][i] = p5, p15, p50
        out["p85"][i], out["p95"][i] = p85, p95
        out["poc"][i] = grid[int(np.argmax(dist))]
        k = int(np.searchsorted(grid, s.c[i], side="right")) - 1
        out["profit"][i] = cs[k] / tot if k >= 0 else 0.0
        out["conc90"][i] = (p95 - p5) / (p95 + p5)
        out["conc70"][i] = (p85 - p15) / (p85 + p15)
        if keep_dist_at is not None and i == keep_dist_at:
            kept = dist / tot
    if keep_dist_at is not None:
        out["dist"] = kept
        out["grid"] = grid
    return out
