"""實盤追蹤：每月入金 → 依訊號用零股操作，對照同樣金額定期定額買 ETF。

- 參數預設＝回測 #42（集中度 0.15、成交額 2,000 萬～1 億、市值 ≤ 500 億、停利 0.4、持有 60、停損 0.1、最多 10 檔）。
- 每天收盤資料更新後自動掃描最新訊號（只算符合流動性、市值條件的股票，約 1～2 分鐘）。
- 持股的停損價、移動停利線、持有天數，用跟回測完全相同的規則計算；觸發就列入「明日賣出」。
- 對照組：每筆入金照設定比例買 ETF（還原價＝含息），用同樣的現金流比較。
"""
import json
import math
import os
import threading
import time
import traceback
from datetime import date, timedelta

import numpy as np

from . import backtest, chips, db, fetch

SCAN_PATH = os.path.join(db.DATA_DIR, "live_scan.json")

DEFAULT_CFG = {
    "params": {"conc_th": 0.15, "low_pos": 1.0, "vol_mult": 1.5, "max_mcap": 500, "min_mcap": 0,
               "min_value": 20_000_000, "max_value": 100_000_000, "trail": 0.4, "max_hold": 60,
               "max_stop": 0.1, "mode": "chip", "edge": "p95", "decay": 1.0, "min_price": 10,
               "warmup": 250},
    "from_run": 42,
    "max_pos": 10,
    "min_order": 5000,       # 每檔最低投入金額
    "limit_up": 1.03,        # 買進限價＝前收 × 1.03
    "sell_limit": 0.97,      # 賣出限價＝前收 × 0.97（出場以成交為優先）
    "fee_rate": 0.001425,
    "fee_discount": 0.6,
    "min_fee": 1,            # 零股每筆最低手續費（依券商調整）
    "tax": 0.003,
}
DEFAULT_BENCH = {"00878": 0.5, "00981A": 0.5}


def cfg():
    c = dict(DEFAULT_CFG)
    c.update(db.get_setting("live_cfg") or {})
    c["params"] = {**DEFAULT_CFG["params"], **(c.get("params") or {})}
    return c


def bench():
    b = db.get_setting("live_bench")
    return b if b else dict(DEFAULT_BENCH)


def tick(p):
    for lim, t in ((10, 0.01), (50, 0.05), (100, 0.1), (500, 0.5), (1000, 1)):
        if p < lim:
            return t
    return 5


def floor_tick(p):
    t = tick(p)
    return round(math.floor(p / t + 1e-9) * t, 2)


def ceil_tick(p):
    t = tick(p)
    return round(math.ceil(p / t - 1e-9) * t, 2)


def ark_lines(sells, buys, c):
    """轉成方舟下單網站「批次貼上」格式：代號,buy/sell,股數,價格,備註（零股一筆最多 999 股）。"""
    out = []
    for x in sells:
        price = ceil_tick(x["last"] * c.get("sell_limit", 0.97))
        q = int(x["shares"])
        while q > 0:
            n = min(q, 999)
            out.append(f"{x['code']},sell,{n},{price},籌碼峰 {x['reason']}")
            q -= n
    for x in buys:
        out.append(f"{x['code']},buy,{min(int(x['shares']), 999)},{x['limit']},籌碼峰 集中度{x['conc90'] * 100:.1f}%")
    return "\n".join(out)


def fee_of(c, amount):
    return max(c["min_fee"], round(amount * c["fee_rate"] * c["fee_discount"]))


# ---------- 每日掃描 ----------
def scan(st=None):
    c = cfg()
    p = backtest.norm_params(c["params"])
    shares = db.load_shares()
    con = db.connect()
    last = con.execute("SELECT MAX(date) FROM daily").fetchone()[0]
    out = []
    allc = db.codes()
    for k, (code, _, _) in enumerate(allc):
        if st is not None:
            st["msg"] = f"掃描 {code}（{k}/{len(allc)}）"
            st["done"], st["total"] = k, len(allc)
        s = chips.load(code, shares.get(code), con)
        if s is None or s.dates[-1] != last or len(s.c) < max(p["warmup"], 60) + 2:
            continue
        # 先用最後一天的流動性、股價、市值篩掉，省下籌碼計算
        v20 = float(np.mean(s.val[-20:]))
        if v20 < p["min_value"] or (p["max_value"] > 0 and v20 > p["max_value"]) or s.craw[-1] < p["min_price"]:
            continue
        if p["max_mcap"] > 0 and (not s.shares or s.craw[-1] * s.shares / 1e8 > p["max_mcap"]):
            continue
        di = {d: i for i, d in enumerate(s.dates)}
        _, sigs, _, _, _, _ = backtest.run_stock(s, p, di)
        for g in sigs:
            if g["date"] == last:
                g["raw_close"] = round(float(s.craw[-1]), 2)
                out.append(g)
    con.close()
    out.sort(key=lambda g: g["conc90"])
    res = {"date": last, "created": db.now(), "signals": out, "params": p}
    with open(SCAN_PATH, "w", encoding="utf-8") as f:
        json.dump(res, f, ensure_ascii=False)
    return res


def last_scan():
    try:
        with open(SCAN_PATH, encoding="utf-8") as f:
            return json.load(f)
    except (OSError, ValueError):
        return None


class ScanJob:
    def __init__(self):
        self.thread = None
        self.state = {"running": False, "msg": "閒置", "done": 0, "total": 0, "error": None}

    def running(self):
        return bool(self.thread and self.thread.is_alive())

    def start(self):
        if self.running():
            return False
        self.thread = threading.Thread(target=self._run, daemon=True)
        self.thread.start()
        return True

    def _run(self):
        st = self.state
        st.update(running=True, error=None, msg="準備掃描")
        t0 = time.time()
        try:
            # 先補對照 ETF 的歷史（只補缺的月份）
            start = first_date() or fetch.today_tpe()
            for code in bench():
                if code[0] == "0":
                    st["msg"] = f"補 ETF {code} 資料"
                    try:
                        fetch.backfill_code(code, start - timedelta(days=90))
                    except Exception as e:
                        st["error"] = f"{code} 補資料失敗：{e}"
            scan(st)
            st["msg"] = f"完成（{time.time() - t0:.0f} 秒）"
        except Exception as e:
            traceback.print_exc()
            st["error"] = str(e)
            st["msg"] = f"失敗：{e}"
        finally:
            st["running"] = False


JOB = ScanJob()


def auto_loop():
    """資料更新到新的一天、而且抓取沒在跑時，自動掃描一次。"""
    while True:
        try:
            s = db.stats()
            ls = last_scan()
            if s["last"] and (not ls or ls["date"] < s["last"]) and not fetch.JOB.running() \
                    and not JOB.running() and not backtest.JOB.running():
                JOB.start()
        except Exception:
            traceback.print_exc()
        time.sleep(300)


# ---------- 帳戶狀態 ----------
def first_date():
    rows = db.live_rows("live_cash")
    return date.fromisoformat(rows[0]["date"]) if rows else None


def _series(code, cache, con):
    if code not in cache:
        cache[code] = chips.load(code, db.load_shares().get(code), con)
    return cache[code]


def state():
    c = cfg()
    p = backtest.norm_params(c["params"])
    cash_rows = db.live_rows("live_cash")
    fills = db.live_rows("live_fills")
    con = db.connect()
    cache = {}
    names = {code: n for code, n, _ in db.codes(all_codes=True)}

    # 現金
    deposited = sum(r["amount"] for r in cash_rows)
    cash = deposited
    lots = {}                      # code -> [ {date, price, shares, cost_per_share} ]
    realized = 0.0
    for f in fills:
        amt = f["price"] * f["shares"]
        if f["side"] == "buy":
            cash -= amt + f["fee"]
            lots.setdefault(f["code"], []).append(
                {"date": f["date"], "price": f["price"], "shares": f["shares"],
                 "cps": (amt + f["fee"]) / f["shares"]})
        else:
            cash += amt - f["fee"] - f["tax"]
            q = f["shares"]
            for lot in lots.get(f["code"], []):
                take = min(q, lot["shares"])
                realized += take * (f["price"] - lot["cps"]) - (f["fee"] + f["tax"]) * take / f["shares"]
                lot["shares"] -= take
                q -= take
                if q <= 0:
                    break
            lots[f["code"]] = [l for l in lots.get(f["code"], []) if l["shares"] > 0]

    # 持股與出場檢查（跟回測相同規則，用還原價計算）
    holdings, sells = [], []
    mv = 0.0
    for code, ls in lots.items():
        if not ls:
            continue
        sh = sum(l["shares"] for l in ls)
        cost = sum(l["shares"] * l["cps"] for l in ls)
        entry_d = min(l["date"] for l in ls)
        s = _series(code, cache, con)
        h = {"code": code, "name": names.get(code, code), "shares": sh, "entry_date": entry_d,
             "cost": round(cost / sh, 2)}
        if s is None:
            h["note"] = "沒有行情資料"
            holdings.append(h)
            continue
        last_raw = float(s.craw[-1])
        h["last"] = round(last_raw, 2)
        h["value"] = round(sh * last_raw)
        h["pnl"] = round(sh * last_raw * (1 - c["fee_rate"] * c["fee_discount"] - c["tax"]) / cost - 1, 4)
        mv += sh * last_raw
        ie = next((i for i, d in enumerate(s.dates) if d >= entry_d), None)
        if ie is None:
            holdings.append(h)
            continue
        adj_k = s.c[ie] / s.craw[ie]                       # 進場日原始價 → 還原價
        entry_adj = float(min(l["price"] for l in ls if l["date"] == entry_d) * adj_k)
        m = chips.compute(s, decay=p["decay"]) if ie >= 2 else None
        p5 = float(m["p5"][ie - 2]) if m is not None and not np.isnan(m["p5"][ie - 2]) else 0.0
        stop = max(p5, entry_adj * (1 - p["max_stop"]))
        peak = max(entry_adj, float(np.max(s.c[ie:])))
        days = len(s.dates) - ie
        lastc = float(s.c[-1])
        to_raw = s.craw[-1] / s.c[-1]
        h.update({"stop": round(stop * to_raw, 2), "trail_line": round(peak * (1 - p["trail"]) * to_raw, 2),
                  "peak": round(peak * to_raw, 2), "days": days, "max_hold": p["max_hold"]})
        why = None
        if lastc < stop:
            why = "停損"
        elif lastc < peak * (1 - p["trail"]):
            why = "移動停利"
        elif days >= p["max_hold"]:
            why = "持有到期"
        if why:
            h["exit"] = why
            sells.append({"code": code, "name": h["name"], "shares": sh, "reason": why,
                          "est": round(sh * last_raw), "last": last_raw})
        holdings.append(h)

    equity = cash + mv
    # 明日買進建議
    sc = last_scan()
    held = {h["code"] for h in holdings}
    staying = len(holdings) - len(sells)
    slots_cap = min(c["max_pos"], int(equity // c["min_order"])) if c["min_order"] > 0 else c["max_pos"]
    free = max(0, slots_cap - staying)
    budget_each = equity / c["max_pos"] if c["max_pos"] else 0
    budget_each = max(budget_each, c["min_order"])
    cash_after = cash + sum(x["est"] * (1 - c["fee_rate"] * c["fee_discount"] - c["tax"]) for x in sells)
    buys = []
    for g in (sc or {}).get("signals", []):
        if len(buys) >= free:
            break
        if g["code"] in held:
            continue
        price = floor_tick(g["raw_close"] * c["limit_up"])
        amt = min(budget_each, cash_after)
        n = min(999, int(amt // (price * (1 + c["fee_rate"] * c["fee_discount"]))))
        if n <= 0 or n * price < c["min_order"] * 0.8:
            break
        buys.append({"code": g["code"], "name": g["name"], "shares": n, "limit": price,
                     "amount": round(n * price), "conc90": g["conc90"], "close": g["raw_close"],
                     "stop_hint": g["low"]})
        cash_after -= n * price + fee_of(c, n * price)

    # ETF 對照：每筆入金照比例買進（還原價＝含息）
    b = bench()
    bval, bmissing = 0.0, []
    for code, w in b.items():
        s = _series(code, cache, con)
        if s is None:
            bmissing.append(code)
            continue
        units = 0.0
        for r in cash_rows:
            i = next((k for k, d in enumerate(s.dates) if d >= r["date"]), None)
            if i is None:
                i = len(s.dates) - 1
            units += r["amount"] * w / float(s.c[i])
        bval += units * float(s.c[-1])
    con.close()
    return {
        "cfg": c, "bench": b, "deposited": round(deposited), "cash": round(cash),
        "market_value": round(mv), "equity": round(equity), "realized": round(realized),
        "ret": round(equity / deposited - 1, 4) if deposited else None,
        "bench_value": round(bval) if not bmissing else None, "bench_missing": bmissing,
        "bench_ret": round(bval / deposited - 1, 4) if deposited and not bmissing else None,
        "holdings": sorted(holdings, key=lambda h: h["entry_date"]), "sells": sells, "buys": buys,
        "free_slots": free, "ark_text": ark_lines(sells, buys, c), "scan": {k: sc[k] for k in ("date", "created")} if sc else None,
        "scan_count": len(sc["signals"]) if sc else 0,
        "cash_rows": cash_rows, "fills": fills,
    }


def add_fill(d, code, side, price, shares, fee=None, tax=None, note=""):
    c = cfg()
    amt = price * shares
    if fee is None or fee == "":
        fee = fee_of(c, amt)
    if tax is None or tax == "":
        tax = round(amt * c["tax"]) if side == "sell" else 0
    return db.live_insert("live_fills", {"date": d, "code": code.strip().upper(), "side": side,
                                         "price": float(price), "shares": int(shares),
                                         "fee": float(fee), "tax": float(tax), "note": note})


def apply_run(rid):
    r = db.get_run(int(rid))
    if not r:
        raise ValueError(f"找不到回測 #{rid}")
    keys = set(DEFAULT_CFG["params"])
    c = cfg()
    c["params"] = {k: v for k, v in r["params"].items() if k in keys}
    c["max_pos"] = int(r["params"].get("max_pos", c["max_pos"]))
    c["from_run"] = int(rid)
    db.set_setting("live_cfg", c)
    return c


def today():
    return fetch.today_tpe().isoformat()
