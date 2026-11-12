"""離線測試：解析格式、還原價、籌碼模型、整體回測流程（合成資料）。"""
import os, sys, tempfile
from datetime import date, timedelta

TMP = tempfile.mkdtemp()
os.environ["DATA_DIR"] = TMP
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
from app import db, fetch, chips, backtest

ok = 0
def check(cond, msg):
    global ok
    assert cond, msg
    ok += 1
    print("  ✓", msg)

print("解析")
twse = {"stat": "OK", "tables": [
    {"title": "價格指數", "fields": ["指數", "收盤指數"], "data": [["發行量加權", "22000"]]},
    {"title": "每日收盤行情", "fields": ["證券代號", "證券名稱", "成交股數", "成交筆數", "成交金額",
     "開盤價", "最高價", "最低價", "收盤價", "漲跌(+/-)", "漲跌價差", "最後揭示買價"],
     "data": [["2330", "台積電", "30,000,000", "50000", "30,000,000,000", "1,000.00", "1,010.00",
               "995.00", "1,005.00", "<p style= color:red>+</p>", "5.00", "1,005"],
              ["0050", "元大台灣50", "1", "1", "1", "1", "1", "1", "1", "+", "0", "1"],
              ["1101", "台泥", "1,000", "10", "30,000", "30.00", "30.50", "29.80", "29.90",
               "<p style= color:green>-</p>", "0.10", "29.9"],
              ["1102", "亞泥", "0", "0", "0", "--", "--", "--", "--", "X", "0.00", ""]]}]}
rows = fetch.parse_twse(twse)
check(len(rows) == 3, "證交所：排除 ETF，保留 3 檔")
check(rows[0]["close"] == 1005 and rows[0]["change"] == 5, "證交所：千分位與漲跌號")
check(rows[1]["change"] == -0.1, "證交所：跌")
check(rows[2]["close"] is None and rows[2]["change"] is None, "證交所：無成交")
check(fetch.parse_twse({"stat": "很抱歉，沒有符合條件的資料!"}) is None, "證交所：休市")
tpex_new = {"stat": "ok", "tables": [{"fields": ["代號", "名稱", "收盤 ", "漲跌", "開盤 ", "最高 ",
            "最低", "均價 ", "成交股數  ", "成交金額(元)", "成交筆數 "],
            "data": [["6488", "環球晶", "500.00", "-5.00 ", "505.00", "508.00", "498.00", "502",
                      "1,234,000", "619,468,000", "3000"]]}]}
r = fetch.parse_tpex(tpex_new)
check(r[0]["close"] == 500 and r[0]["change"] == -5 and r[0]["volume"] == 1234000, "櫃買新版格式")
tpex_old = {"aaData": [["6488", "環球晶", "500.00", "+5.00", "495.00", "501.00", "494.00", "498",
                        "1,000", "498,000", "10"]]}
r = fetch.parse_tpex(tpex_old)
check(r[0]["open"] == 495 and r[0]["change"] == 5, "櫃買舊版格式")
check(fetch.parse_tpex({"tables": [{"fields": ["代號", "收盤"], "data": []}]}) is None, "櫃買：空表＝休市")

tw_d = dict(twse, date="20240102")
check(fetch.check_date(tw_d, date(2024, 1, 2), "x") is None, "日期驗證：相符")
try:
    fetch.check_date(tw_d, date(2024, 1, 3), "x"); bad = False
except ValueError:
    bad = True
check(bad, "日期驗證：回傳日期不符 → 報錯")
try:
    fetch.check_date({"tables": [{"date": "113/11/20", "fields": [], "data": []}]}, date(2024, 11, 21), "x"); bad = False
except ValueError:
    bad = True
check(bad, "日期驗證：民國日期也能比對")
ex_new = {"stat": "ok", "tables": [{"fields": ["除權息日期", "代號", "名稱", "除權息前收盤價", "除權息參考價", "權值"],
          "data": [["113/07/15", "6488", "環球晶", "500.00", "490.00", "0"], ["113/07/15", "00679B", "ETF", "30", "29", "0"]]}]}
check(fetch.parse_adj(ex_new, "TPEX_EX") == [("6488", "2024-07-15", 0.98)], "除權息表：新版格式、排除 ETF")
ex_tw = {"stat": "OK", "fields": ["資料日期", "股票代號", "股票名稱", "除權息前收盤價", "除權息參考價"],
         "data": [["113年09月12日", "2330", "台積電", "950.00", "946.00"]]}
check(fetch.parse_adj(ex_tw, "TWSE_EX") == [("2330", "2024-09-12", round(946 / 950, 6))], "除權息表：證交所格式")
ex_old = {"aaData": [["113/07/15", "6488", "環球晶", "500.00", "490.00"]]}
check(fetch.parse_adj(ex_old, "TPEX_EX") == [("6488", "2024-07-15", 0.98)], "除權息表：櫃買舊版 aaData")
check(fetch.parse_adj({"foo": 1}, "TWSE_EX") is None, "除權息表：認不出格式 → None（改試下一個網址）")

print("合成資料")
db.init()
rng = np.random.default_rng(7)
d0 = date(2023, 1, 2)
days = []
d = d0
while len(days) < 520:
    if d.weekday() < 5:
        days.append(d.isoformat())
    d += timedelta(days=1)

def make(code, kind):
    px = 50.0
    out = []
    for i, ds in enumerate(days):
        if kind == "base":
            if i < 200: px = 60 * 0.996 ** i * (1 + rng.normal(0, 0.01))  # 下跌
            elif i < 330: px = 27 * (1 + rng.normal(0, 0.008))       # 低檔盤整（單峰密集）
            elif i < 345: px *= 1.025                                # 突破
            else: px *= 1 + rng.normal(0.001, 0.015)
            vol = 3e6 if 330 <= i < 345 else 8e5
        else:
            px *= 1 + rng.normal(0.0003, 0.02)
            vol = 1e6 * (1 + abs(rng.normal()))
        px = max(px, 1)
        hi, lo = px * (1 + abs(rng.normal(0, 0.008))), px * (1 - abs(rng.normal(0, 0.008)))
        op = lo + (hi - lo) * rng.random()
        out.append((ds, round(op, 2), round(hi, 2), round(lo, 2), round(px, 2), vol))
    return out

series = {"1111": make("1111", "base"), "2222": make("2222", "rw"), "3333": make("3333", "rw")}
# 3333 在第 300 天除息 2 元
for i, ds in enumerate(days):
    rowsd = []
    for code, ser in series.items():
        _, o, h, l, c, v = ser[i]
        prev = ser[i - 1][4] if i else c
        if code == "3333" and i >= 300:
            o, h, l, c = o - 2, h - 2, l - 2, c - 2
        ref = prev - 2 if (code == "3333" and i == 300) else (prev if i else c)
        if code == "3333" and 0 < i < 300:
            ref = prev
        if code == "3333" and i > 300:
            ref = series[code][i - 1][4] - 2
        rowsd.append({"code": code, "name": "測" + code, "open": o, "high": h, "low": l, "close": c,
                      "volume": v, "value": v * c, "change": round(c - ref, 2)})
    db.save_day(rowsd, ds, "TWSE", "ok")
db.save_shares([("1111", 2e7), ("2222", 1e8)], "TWSE")

prev299 = series["3333"][299][4]
db.save_adj([("3333", days[300], round((prev299 - 2) / prev299, 6))], "TWSE_EX")
s = chips.load("3333", None)
check(len(s.events) == 1 and s.events[0][0] == days[300], "還原：抓到除息日")
check(abs(s.c[299] - (series["3333"][299][4] - 2)) < 0.05, "還原：除息前價格已向下調整")
check(s.turn_est, "無股本 → 換手率估計模式")

s = chips.load("1111", 2e7)
m = chips.compute(s)
check(m["conc90"][325] < m["conc90"][150], "盤整後籌碼比下跌段集中")
check(abs(m["poc"][325] - 27) / 27 < 0.05, "主峰落在盤整價 27 附近")
check(0 <= np.nanmin(m["profit"]) and np.nanmax(m["profit"]) <= 1, "獲利比例在 0~1")

print("回測")
p = backtest.norm_params({"conc_th": 0.15, "min_value": 1e6, "low_pos": 0.85})
rid = backtest.run(p)
r = db.get_run(rid)
check(r is not None and "event" in r, "回測完成並存檔")
codes_sig = {t["code"] for t in r["trades"]}
check("1111" in codes_sig, f"合成的低位突破股有觸發（{len(r['trades'])} 筆交易）")
t = next(t for t in r["trades"] if t["code"] == "1111")
check(t["entry_date"] > t["date"], "進場在訊號日之後（無前視）")
check(len(r["event"]) == 4, "事件研究 4 個週期")
check(set(r["tiers"]) == {"val", "cap"} and len(r["tiers"]["val"]["rows"]) == 3, "分層統計：成交額、市值")
check(sum(x["trades"].get("n", 0) + x["trades"].get("open", 0) for x in r["tiers"]["val"]["rows"]) == len(r["trades"]),
      "分層交易數加總 = 全部交易")
check(t["mcap"] is not None and t["ct"] >= 0, "有股本 → 有市值分層")
v = backtest.chip_view("1111", days[340])
check(v and abs(sum(v["dist"]) - 1) < 0.02 and v["date"] == days[340], "籌碼圖資料")
print("穩健性功能")
e20 = next(e for e in r["event"] if e["h"] == 20)
check(all(k in e20 for k in ("median", "trim", "t_day", "days")), "事件研究有中位數、去極值、按日 t")
check("years" in r["tiers"]["cap"]["rows"][0], "分層有逐年")
rb = db.get_run(backtest.run(backtest.norm_params({"mode": "box", "conc_th": 0.2, "min_value": 1e6})))
check(rb["summary"]["signals"] >= 0 and rb["params"]["mode"] == "box", "對照組（價格箱型）可執行")
rm = db.get_run(backtest.run(backtest.norm_params({"conc_th": 0.15, "min_value": 1e6, "low_pos": 0.85, "max_mcap": 1})))
check(rm["summary"]["signals"] == 0, "市值上限 1 億 → 全部排除")
before = len(db.list_runs())
job = backtest.BacktestJob()
job._run(backtest.norm_params({"min_value": 1e6, "sweep_key": "conc_th", "sweep": "0.10,0.15,0.20"}))
runs = db.list_runs()
check(len(runs) == before + 3 and {x["params"]["conc_th"] for x in runs[:3]} == {0.1, 0.15, 0.2}, "敏感度掃描存成 3 次回測")
check(all(x["params"]["sweep"] == "" for x in runs[:3]), "掃描產生的回測不帶掃描設定")
check(backtest.sweep_values(backtest.norm_params({"sweep_key": "conc_th", "sweep": "0.10、0.15 0.20"})) == [0.1, 0.15, 0.2],
      "掃描值接受頓號、空白分隔")
try:
    backtest.BacktestJob().start({"sweep": "0.1,0.2"}); bad = False
except ValueError:
    bad = True
check(bad, "填了掃描值但沒選參數 → 直接報錯，不會默默跑一般回測")
job2 = backtest.BacktestJob()
n0 = len(db.list_runs())
job2._run(backtest.norm_params({"min_value": 1e6, "sweep_key": "max_hold", "sweep": "40,90"}))
rs = db.list_runs()[:2]
check(len(db.list_runs()) == n0 + 2 and {x["params"]["max_hold"] for x in rs} == {40, 90}
      and all(isinstance(x["params"]["max_hold"], int) for x in rs), "掃描最長持有（整數參數）")
print("投組與新參數")
r0 = db.get_run(backtest.run(backtest.norm_params({"min_value": 1e6})))
pt = r0["portfolio"]
check(pt and pt["curve"]["v"][0] > 0 and -1 < pt["mdd"] <= 0 and pt["taken"] >= 1, "投組模擬有資金曲線、回撤")
check(r0["summary"]["cagr"] == pt["cagr"], "摘要帶年化報酬")
r1 = db.get_run(backtest.run(backtest.norm_params({"min_value": 1e6, "slippage": 0.01})))
t0 = {(t["code"], t["entry_date"]): t["ret"] for t in r0["trades"]}
t1 = {(t["code"], t["entry_date"]): t["ret"] for t in r1["trades"]}
k = next(iter(t0.keys() & t1.keys()))
check(t1[k] < t0[k] - 0.015, "滑價 1% 每邊 → 每筆報酬少約 2%")
r2 = db.get_run(backtest.run(backtest.norm_params({"min_value": 1e6, "breadth_min": 1.01})))
check(r2["summary"]["signals"] == 0, "大盤濾網門檻 > 100% → 沒有訊號")
r3 = db.get_run(backtest.run(backtest.norm_params({"min_value": 1e6, "max_value": 1})))
check(r3["summary"]["signals"] == 0, "成交額上限 1 元 → 沒有訊號")
from app import portfolio
con_ = db.connect()
mc = portfolio.market_context(sorted({d for (d,) in con_.execute("SELECT DISTINCT date FROM daily")}),
                              ["1111", "2222", "3333"], db.load_shares(), con_, 60)
con_.close()
bb = mc["breadth"][~np.isnan(mc["breadth"])]
check(len(bb) > 100 and bb.min() >= 0 and bb.max() <= 1, "大盤寬度介於 0~1")
fake = [{"entry_date": days[10], "exit_date": days[12], "open": False, "ret": 0.10, "conc90": 0.1, "vol_x": 2},
        {"entry_date": days[10], "exit_date": days[12], "open": False, "ret": 0.10, "conc90": 0.2, "vol_x": 2}]
paths_ = [(10, np.array([1.0, 1.05])), (10, np.array([1.0, 1.05]))]
sim = portfolio.simulate(fake, paths_, days, np.ones(len(days)), max_pos=1)
check(sim["taken"] == 1 and sim["skipped"] == 1 and abs(sim["total"] - 0.10) < 1e-9,
      "投組：空位 1 檔只買 1 檔，100% 投入賺 10%")
sim2 = portfolio.simulate(fake, paths_, days, np.ones(len(days)), max_pos=4)
check(sim2["taken"] == 2 and abs(sim2["total"] - 0.05) < 1e-9, "投組：4 個空位各 25%，兩檔各賺 10% → 總 5%")
# 漲跌少負號的資料不應被當成除權息
db.save_day([{"code": "4444", "name": "少負號", "open": 10, "high": 10, "low": 10, "close": 10, "volume": 1e6, "value": 1e7, "change": 0}], days[0], "TPEX", "ok")
pxs = [10.0]
for i in range(1, 60):
    pxs.append(round(pxs[-1] * (0.98 if i % 2 else 1.03), 2))
    chg = round(pxs[-1] - pxs[-2], 2)
    db.save_day([{"code": "4444", "name": "少負號", "open": pxs[-1], "high": pxs[-1], "low": pxs[-1], "close": pxs[-1],
                  "volume": 1e6, "value": 1e7, "change": abs(chg)}], days[i], "TPEX", "ok")
s4 = chips.load("4444", None)
check(len(s4.events) == 0 and abs(s4.c[-1] - pxs[-1]) < 1e-6, "漲跌少負號 → 不誤判成除權息、價格不被扭曲")
# 減資：不比價且開盤跳空超過漲跌停 → 自動還原
rows5 = []
for i in range(80):
    px = 10.0 if i < 50 else 40.0
    rows5.append((days[i], px, i == 50))
for d5, px, xday in rows5:
    db.save_day([{"code": "5555", "name": "減資", "open": px, "high": px, "low": px, "close": px,
                  "volume": 1e6, "value": 1e7, "change": None if xday else 0.0}], d5, "TWSE", "ok")
s5 = chips.load("5555", None)
check(len(s5.events) == 1 and abs(s5.c[49] - 40) < 1e-6, "減資跳空（不比價日）自動還原，價格連續")
# 櫃買快照偵測
snap = [{"code": f"{6000 + i}", "name": "x", "open": 10, "high": 10, "low": 10, "close": 10 + i * 0.1,
         "volume": 1000 + i, "value": 1, "change": 0} for i in range(60)]
db.save_day(snap, "2030-01-02", "TPEX", "ok")
check(fetch._same_as_last(snap, "TPEX", date(2030, 1, 3)), "櫃買：跟前一天一模一樣 → 判定為快照")
snap2 = [dict(r, close=r["close"] + 0.05) for r in snap]
check(not fetch._same_as_last(snap2, "TPEX", date(2030, 1, 3)), "櫃買：正常變動 → 不誤判")
# 面額變更（10 元→1 元）：櫃買漲跌相對參考價 → 用收盤−漲跌當參考價
for i in range(80):
    px = 720.0 if i < 50 else 79.2
    chg6 = 7.2 if i == 50 else 0.0
    db.save_day([{"code": "6666", "name": "面額*", "open": px, "high": px, "low": px, "close": px,
                  "volume": 1e6, "value": 1e7, "change": chg6}], days[i], "TPEX", "ok")
s6 = chips.load("6666", None)
check(len(s6.events) == 1 and abs(s6.events[0][1] - 0.1) < 1e-6 and abs(s6.c[49] - 72) < 1e-6,
      "面額變更：用收盤−漲跌推回參考價，價格連續")
# 官方事件跟價格對不上（沒跳空）→ 不採用
for i in range(80):
    px = 40.0 + i * 0.1
    db.save_day([{"code": "7777", "name": "錯位", "open": px, "high": px, "low": px, "close": px,
                  "volume": 1e6, "value": 1e7, "change": 0.1}], days[i], "TWSE", "ok")
db.save_adj([("7777", days[60], 0.676)], "TWSE_CR")
s7 = chips.load("7777", None)
check(len(s7.events) == 0, "官方事件但當天沒跳空 → 判定錯位、不採用")
# 大額除息（5%）當天開盤漲停：仍要採用
for i in range(80):
    px = 100.0 if i < 60 else (95 * 1.1 if i == 60 else 104.5)
    db.save_day([{"code": "8888", "name": "大息", "open": px, "high": px, "low": px, "close": px,
                  "volume": 1e6, "value": 1e7, "change": None if i == 60 else 0.0}], days[i], "TWSE", "ok")
db.save_adj([("8888", days[60], 0.95)], "TWSE_EX")
s8 = chips.load("8888", None)
check(len(s8.events) == 1 and abs(s8.events[0][1] - 0.95) < 1e-9, "大額除息當天開盤漲停 → 仍採用官方因子")
from app import diag
dg = diag.run(min_value=1e6)
check(dg["events"] and dg["years"] and len(dg["extremes"]) > 0, "資料檢查有輸出")
ev3 = {e["market"]: e for e in dg["events"]}["TWSE"]
check(ev3["events"] == 3, "資料檢查：上市 3 個還原事件，錯位事件不計")
diag.DiagJob()._run()
lr = diag.last_result()
check(lr and lr["created"] and lr["events"], "資料檢查在背景跑完並存檔")
print("實盤追蹤")
from app import live
db.set_setting("live_bench", {"00878": 1.0})
check(fetch.is_stock("00878", fetch.bench_codes()) and not fetch.is_stock("00878"), "對照 ETF 抓資料時保留，一般情況排除")
for i, ds in enumerate(days):
    px = round(20 * (1.0005 ** i), 2)
    fetch._save_one({"code": "00878", "name": "國泰永續高股息", "open": px, "high": px, "low": px, "close": px,
                     "volume": 1e7, "value": 2e8, "change": 0.0}, ds, "TWSE")
check(all(c != "00878" for c, _, _ in db.codes()), "ETF 不會進回測母體")
db.live_insert("live_cash", {"date": days[400], "amount": 10000, "note": ""})
db.live_insert("live_cash", {"date": days[420], "amount": 10000, "note": ""})
p1111 = series["1111"][401][1]
live.add_fill(days[401], "1111", "buy", p1111, 100)
st0 = live.state()
exp_cash = 20000 - p1111 * 100 - max(1, round(p1111 * 100 * 0.001425 * 0.6))
check(abs(st0["cash"] - round(exp_cash)) <= 1, "現金＝入金−買進金額−手續費")
check(len(st0["holdings"]) == 1 and st0["holdings"][0]["days"] == len(days) - 401, "持股與持有天數")
h = st0["holdings"][0]
check(h["stop"] > 0 and h["trail_line"] > 0, "停損價、移動停利線有算出來")
exp_b = 10000 / (20 * 1.0005 ** 400) * (20 * 1.0005 ** 519) + 10000 / (20 * 1.0005 ** 420) * (20 * 1.0005 ** 519)
check(abs(st0["bench_value"] - exp_b) < 5, "ETF 對照：每筆入金照當日價買進")
# 持有到期 → 賣出建議
live.add_fill(days[300], "2222", "buy", series["2222"][300][1], 10)
st1 = live.state()
s2 = next(x for x in st1["sells"] if x["code"] == "2222")
check(s2["reason"] in ("持有到期", "停損", "移動停利"), f"超過 60 天的持股列入明日賣出（{s2['reason']}）")
# 賣出後部位與已實現損益
fid = live.add_fill(days[-1], "2222", "sell", series["2222"][-1][4], 10)
st2 = live.state()
check(all(h["code"] != "2222" for h in st2["holdings"]), "賣出後部位歸零")
db.live_delete("live_fills", fid)
check(any(h["code"] == "2222" for h in live.state()["holdings"]), "刪除成交紀錄可還原")
res = live.scan()
check(res["date"] == db.stats()["last"] and isinstance(res["signals"], list) and live.last_scan()["created"], "每日掃描完成並存檔")
import json as _j
with open(live.SCAN_PATH, "w", encoding="utf-8") as _f:
    _j.dump({"date": "x", "created": "x", "signals": [
        {"code": "3333", "name": "甲", "conc90": 0.05, "raw_close": 40.0, "low": 38.0},
        {"code": "1111", "name": "已持有", "conc90": 0.06, "raw_close": 30.0, "low": 28.0},
        {"code": "4444", "name": "乙", "conc90": 0.07, "raw_close": 20.0, "low": 19.0}]}, _f)
st3 = live.state()
bcodes = [x["code"] for x in st3["buys"]]
check("1111" not in bcodes and (not bcodes or bcodes[0] == "3333"), "買進建議：跳過已持有、集中度小的優先")
check(all(x["limit"] == live.floor_tick(x["close"] * 1.03) and x["amount"] <= max(st3["equity"] / 10, 5000) + 1
          for x in st3["buys"]), "買進建議：限價＝前收×1.03，金額不超過每檔預算")
check(len(bcodes) <= st3["free_slots"], "買進建議不超過空位（含資金不足時減少檔數）")
lines = [l for l in st3["ark_text"].split("\n") if l]
ok_fmt = all(len(l.split(",")) == 5 and l.split(",")[1] in ("buy", "sell") and 0 < int(l.split(",")[2]) < 1000
             and float(l.split(",")[3]) > 0 for l in lines)
check(lines and ok_fmt, "方舟貼上格式：代號,buy/sell,股數(1~999),價格,備註")
check(sum(1 for l in lines if ",sell," in l) >= len(st3["sells"]) and
      all(l.split(",")[0] in {x["code"] for x in st3["sells"]} | {x["code"] for x in st3["buys"]} for l in lines),
      "方舟貼上：包含所有賣出與買進")
big = live.ark_lines([{"code": "1234", "shares": 2100, "last": 50.0, "reason": "停損"}], [], live.cfg())
check(big.split("\n") == ["1234,sell,999,48.5,籌碼峰 停損", "1234,sell,999,48.5,籌碼峰 停損", "1234,sell,102,48.5,籌碼峰 停損"],
      "超過 999 股的賣出自動拆成多筆，賣出限價＝前收×0.97 向上取整")
check(live.floor_tick(123.37) == 123.0 and live.floor_tick(45.33) == 45.3 and live.floor_tick(9.876) == 9.87, "限價依升降單位取整")
print(f"\n全部通過：{ok} 項")
