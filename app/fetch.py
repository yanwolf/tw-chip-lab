"""證交所（上市）＋櫃買中心（上櫃）每日收盤行情抓取。

- 一天一個請求抓全市場，存進 SQLite，已抓過的日期會跳過（可中斷續抓）。
- 證交所請求太密集會被暫時封鎖，所以每個請求間隔 FETCH_INTERVAL 秒（預設 4）。
- 欄位用「欄名」對應而不是固定位置，交易所改版加減欄位不會錯位；
  找不到需要的欄位會直接報錯停下，不會默默存錯資料。
"""
import json
import logging
import os
import re
import threading
import time
import urllib.request
from datetime import date, datetime, timedelta, timezone

from . import db

log = logging.getLogger("fetch")
TPE = timezone(timedelta(hours=8))
INTERVAL = float(os.environ.get("FETCH_INTERVAL", "4"))
UA = {"User-Agent": "Mozilla/5.0 (X11; Linux x86_64) tw-chip-lab",
      "Accept": "application/json,text/plain,*/*"}

TWSE_URL = ("https://www.twse.com.tw/rwd/zh/afterTrading/MI_INDEX"
            "?date={d:%Y%m%d}&type=ALLBUT0999&response=json")
TPEX_URLS = [
    "https://www.tpex.org.tw/www/zh-tw/afterTrading/dailyQ?date={d:%Y/%m/%d}&type=EW&response=json",
    "https://www.tpex.org.tw/web/stock/aftertrading/daily_close_quotes/stk_quote_result.php"
    "?l=zh-tw&d={roc}/{d:%m/%d}&se=EW&o=json",
]
SHARES_URLS = {
    "TWSE": "https://openapi.twse.com.tw/v1/opendata/t187ap03_L",
    "TPEX": "https://www.tpex.org.tw/openapi/v1/mopsfin_t187ap03_O",
}


def today_tpe():
    return datetime.now(TPE).date()


def http_json(url, timeout=30):
    req = urllib.request.Request(url, headers=UA)
    with urllib.request.urlopen(req, timeout=timeout) as r:
        raw = r.read()
    txt = raw.decode("utf-8", errors="replace").strip()
    if not txt or txt[0] not in "[{":
        raise ValueError("回應不是 JSON（可能被暫時封鎖）")
    return json.loads(txt)


def num(s):
    if s is None:
        return None
    s = re.sub(r"<[^>]*>", "", str(s)).replace(",", "").strip()
    if s in ("", "-", "--", "---", "----", "X"):
        return None
    try:
        return float(s)
    except ValueError:
        return None


def is_stock(code):
    """一般普通股：4 碼數字、非 00 開頭（ETF）、非 91 開頭（TDR）。"""
    return bool(re.fullmatch(r"[1-9]\d{3}", code)) and not code.startswith("91")


def _col(fields, *names):
    clean = [str(f).strip() for f in fields]
    for n in names:                      # 先找完全相同
        if n in clean:
            return clean.index(n)
    for n in names:                      # 再找開頭相同
        for i, f in enumerate(clean):
            if f.startswith(n):
                return i
    return None


def _tables(js):
    out = [(t.get("fields") or [], t.get("data") or []) for t in (js.get("tables") or [])]
    for k in js:                         # 舊格式 fields9 / data9
        if k.startswith("fields") and isinstance(js[k], list):
            out.append((js[k], js.get("data" + k[6:]) or []))
    return out


def _row(code, name, o, h, l, c, v, amt, chg):
    return {"code": code, "name": name.strip(), "open": num(o), "high": num(h),
            "low": num(l), "close": num(c), "volume": num(v), "value": num(amt),
            "change": chg}


def parse_twse(js):
    """回傳 rows；休市回傳 None。"""
    if str(js.get("stat", "")).upper() != "OK":
        return None
    for fields, data in _tables(js):
        if _col(fields, "證券代號") is None or _col(fields, "收盤價") is None:
            continue
        ix = {k: _col(fields, k) for k in
              ("證券代號", "證券名稱", "成交股數", "成交金額", "開盤價", "最高價",
               "最低價", "收盤價", "漲跌(+/-)", "漲跌價差")}
        miss = [k for k, v in ix.items() if v is None]
        if miss:
            raise ValueError(f"證交所欄位缺少：{miss}")
        rows = []
        for r in data:
            code = str(r[ix["證券代號"]]).strip()
            if not is_stock(code):
                continue
            sign_raw = re.sub(r"<[^>]*>", "", str(r[ix["漲跌(+/-)"]])).strip()
            diff = num(r[ix["漲跌價差"]])
            if sign_raw == "X" or diff is None:
                chg = None              # 不比價（新上市、除權息當日部分情況等）
            elif "-" in sign_raw:
                chg = -diff
            else:
                chg = diff
            rows.append(_row(code, r[ix["證券名稱"]], r[ix["開盤價"]], r[ix["最高價"]],
                             r[ix["最低價"]], r[ix["收盤價"]], r[ix["成交股數"]],
                             r[ix["成交金額"]], chg))
        return rows
    raise ValueError("證交所格式不符：找不到個股行情表")


def parse_tpex(js):
    """支援新版 tables 格式與舊版 aaData 格式。休市（無資料）回傳 None。"""
    tabs = _tables(js)
    for fields, data in tabs:
        if _col(fields, "代號") is None or _col(fields, "收盤") is None:
            continue
        ix = {"code": _col(fields, "代號"), "name": _col(fields, "名稱"),
              "close": _col(fields, "收盤"), "chg": _col(fields, "漲跌"),
              "open": _col(fields, "開盤"), "high": _col(fields, "最高"),
              "low": _col(fields, "最低"), "vol": _col(fields, "成交股數"),
              "amt": _col(fields, "成交金額")}
        if not data:
            return None
        miss = [k for k, v in ix.items() if v is None]
        if miss:
            raise ValueError(f"櫃買欄位缺少：{miss}")
        return [_row(str(r[ix["code"]]).strip(), r[ix["name"]], r[ix["open"]], r[ix["high"]],
                     r[ix["low"]], r[ix["close"]], r[ix["vol"]], r[ix["amt"]], num(r[ix["chg"]]))
                for r in data if is_stock(str(r[ix["code"]]).strip())]
    if "aaData" in js:                   # 舊版：固定欄位順序
        data = js.get("aaData") or []
        if not data:
            return None
        return [_row(str(r[0]).strip(), r[1], r[4], r[5], r[6], r[2], r[8], r[9], num(r[3]))
                for r in data if is_stock(str(r[0]).strip())]
    if tabs and all(not d for _, d in tabs):
        return None
    raise ValueError("櫃買格式不符：找不到個股行情表")


def fetch_twse(d):
    return parse_twse(http_json(TWSE_URL.format(d=d)))


def fetch_tpex(d):
    last = None
    for u in TPEX_URLS:
        try:
            return parse_tpex(http_json(u.format(d=d, roc=d.year - 1911)))
        except Exception as e:  # 換下一個網址
            last = e
    raise last


def fetch_shares():
    """目前已發行股數（快照，非歷史）。回傳 {market: 筆數}。"""
    out = {}
    for mkt, url in SHARES_URLS.items():
        try:
            js = http_json(url, timeout=60)
        except Exception as e:
            log.warning("股本抓取失敗 %s: %s", mkt, e)
            out[mkt] = f"失敗：{e}"
            continue
        items = []
        for it in js if isinstance(js, list) else []:
            code = next((str(v).strip() for k, v in it.items()
                         if k in ("公司代號", "SecuritiesCompanyCode", "CompanyCode")), None)
            sh = next((num(v) for k, v in it.items()
                       if ("發行普通股" in k) or k in ("IssueShares", "IssuedShares")), None)
            if code and sh and sh > 0:
                items.append((code, sh))
        db.save_shares(items, mkt)
        out[mkt] = len(items)
    return out


class FetchJob:
    """背景抓取工作；同一時間只跑一個。"""

    def __init__(self):
        self.thread = None
        self.stop_flag = False
        self.state = {"running": False, "msg": "閒置", "done": 0, "total": 0,
                      "current": None, "error": None, "finished": None}

    def running(self):
        return bool(self.thread and self.thread.is_alive())

    def start(self, start_date, end_date=None):
        if self.running():
            return False
        self.stop_flag = False
        self.thread = threading.Thread(target=self._run, args=(start_date, end_date),
                                       daemon=True)
        self.thread.start()
        return True

    def stop(self):
        self.stop_flag = True

    def _get(self, fn, d, label):
        for attempt in range(3):
            try:
                return fn(d)
            except Exception as e:
                log.warning("%s %s 第 %d 次失敗：%s", label, d, attempt + 1, e)
                self.state["msg"] = f"{label} {d} 失敗，{60 * (attempt + 1)} 秒後重試：{e}"
                if attempt == 2:
                    raise
                for _ in range(60 * (attempt + 1)):
                    if self.stop_flag:
                        raise RuntimeError("已停止")
                    time.sleep(1)

    def _run(self, start_date, end_date):
        st = self.state
        st.update(running=True, error=None, done=0, msg="準備中")
        try:
            end_date = end_date or today_tpe()
            # 今天收盤資料約 14:30–15:30 後才會出來
            if end_date == today_tpe() and datetime.now(TPE).hour < 16:
                end_date -= timedelta(days=1)
            got = {"TWSE": db.fetched("TWSE"), "TPEX": db.fetched("TPEX")}
            days = []
            d = start_date
            while d <= end_date:
                if d.weekday() < 5:
                    ds = d.isoformat()
                    for mkt in ("TWSE", "TPEX"):
                        if got[mkt].get(ds) not in ("ok", "holiday"):
                            days.append((d, mkt))
                d += timedelta(days=1)
            st["total"] = len(days)
            if not db.load_shares():
                st["msg"] = "抓取已發行股數"
                st["shares"] = fetch_shares()
            dead = {}
            for d, mkt in days:
                if self.stop_flag:
                    st["msg"] = "已停止"
                    break
                if mkt in dead:
                    st["done"] += 1
                    continue
                st["current"] = f"{d} {mkt}"
                st["msg"] = f"抓取 {d} {'上市' if mkt == 'TWSE' else '上櫃'}"
                fn = fetch_twse if mkt == "TWSE" else fetch_tpex
                try:
                    rows = self._get(fn, d, mkt)
                except Exception as e:
                    if self.stop_flag:
                        st["msg"] = "已停止"
                        break
                    db.save_day([], d.isoformat(), mkt, f"error: {e}"[:200])
                    dead[mkt] = f"{d}: {e}"
                    st["error"] = "；".join(f"{k} 停在 {v}" for k, v in dead.items())
                    if len(dead) == 2:
                        raise RuntimeError(st["error"])
                    continue
                if rows is None:
                    db.save_day([], d.isoformat(), mkt, "holiday")
                else:
                    db.save_day(rows, d.isoformat(), mkt, "ok")
                st["done"] += 1
                time.sleep(INTERVAL)
            else:
                st["msg"] = "完成" if not dead else "部分完成（見錯誤）"
        except Exception as e:
            log.exception("抓取中止")
            st["error"] = str(e)
            st["msg"] = f"中止：{e}"
        finally:
            st["running"] = False
            st["finished"] = db.now()


JOB = FetchJob()


def auto_update_loop():
    """AUTO_UPDATE=1 時：台北時間平日 16:30 後自動補抓到今天。"""
    while True:
        try:
            now = datetime.now(TPE)
            if now.weekday() < 5 and (now.hour, now.minute) >= (16, 30) and not JOB.running():
                if db.fetched("TWSE").get(now.date().isoformat()) not in ("ok", "holiday"):
                    s = db.stats()
                    if s["last"]:
                        JOB.start(date.fromisoformat(s["last"]) + timedelta(days=1))
        except Exception:
            log.exception("auto update")
        time.sleep(600)
