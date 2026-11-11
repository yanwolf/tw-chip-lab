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
print(f"\n全部通過：{ok} 項")
