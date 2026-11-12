"""SQLite 資料層。部署時 DATA_DIR 要掛 Volume（預設 /data），否則重啟資料會消失。"""
import json
import os
import sqlite3
import threading
from datetime import datetime

DATA_DIR = os.environ.get("DATA_DIR", "/data")
DB_PATH = os.path.join(DATA_DIR, "chiplab.db")
_lock = threading.Lock()

SCHEMA = """
CREATE TABLE IF NOT EXISTS daily (
  code TEXT, date TEXT, market TEXT, name TEXT,
  open REAL, high REAL, low REAL, close REAL,
  volume REAL, value REAL, change REAL,
  PRIMARY KEY (code, date)
);
CREATE INDEX IF NOT EXISTS idx_daily_date ON daily(date);
CREATE TABLE IF NOT EXISTS fetch_log (
  date TEXT, market TEXT, status TEXT, n INTEGER, ts TEXT,
  PRIMARY KEY (date, market)
);
CREATE TABLE IF NOT EXISTS shares (
  code TEXT PRIMARY KEY, shares REAL, market TEXT, updated TEXT
);
CREATE TABLE IF NOT EXISTS adj_events (
  code TEXT, date TEXT, factor REAL, source TEXT,
  PRIMARY KEY (code, date, source)
);
CREATE TABLE IF NOT EXISTS settings (
  key TEXT PRIMARY KEY, value TEXT
);
CREATE TABLE IF NOT EXISTS live_cash (
  id INTEGER PRIMARY KEY AUTOINCREMENT, date TEXT, amount REAL, note TEXT
);
CREATE TABLE IF NOT EXISTS live_fills (
  id INTEGER PRIMARY KEY AUTOINCREMENT, date TEXT, code TEXT, side TEXT,
  price REAL, shares INTEGER, fee REAL, tax REAL, note TEXT
);
CREATE TABLE IF NOT EXISTS runs (
  id INTEGER PRIMARY KEY AUTOINCREMENT,
  created TEXT, params TEXT, summary TEXT, result TEXT
);
"""


def connect():
    os.makedirs(os.path.dirname(DB_PATH), exist_ok=True)
    con = sqlite3.connect(DB_PATH, timeout=60, check_same_thread=False)
    con.execute("PRAGMA journal_mode=WAL")
    con.execute("PRAGMA synchronous=NORMAL")
    return con


def init():
    with _lock:
        con = connect()
        con.executescript(SCHEMA)
        con.commit()
        con.close()


def now():
    return datetime.now().isoformat(timespec="seconds")


def save_day(rows, date, market, status):
    """rows: list of dict(code,name,open,high,low,close,volume,value,change)"""
    with _lock:
        con = connect()
        con.executemany(
            "INSERT OR REPLACE INTO daily VALUES (?,?,?,?,?,?,?,?,?,?,?)",
            [(r["code"], date, market, r["name"], r["open"], r["high"], r["low"],
              r["close"], r["volume"], r["value"], r["change"]) for r in rows])
        con.execute("INSERT OR REPLACE INTO fetch_log VALUES (?,?,?,?,?)",
                    (date, market, status, len(rows), now()))
        con.commit()
        con.close()


def log_fetch(date, market, status, n):
    with _lock:
        con = connect()
        con.execute("INSERT OR REPLACE INTO fetch_log VALUES (?,?,?,?,?)", (date, market, status, n, now()))
        con.commit()
        con.close()


def save_adj(items, source):
    with _lock:
        con = connect()
        con.executemany("INSERT OR REPLACE INTO adj_events VALUES (?,?,?,?)",
                        [(c, d, f, source) for c, d, f in items])
        con.commit()
        con.close()


def adj_for(code, con=None):
    own = con is None
    con = con or connect()
    rows = con.execute("SELECT date, factor FROM adj_events WHERE code=? ORDER BY date", (code,)).fetchall()
    if own:
        con.close()
    return rows


def adj_stats():
    con = connect()
    rows = con.execute("SELECT source, COUNT(*), MIN(date), MAX(date) FROM adj_events GROUP BY source").fetchall()
    con.close()
    return [{"source": a, "n": b, "first": c, "last": d} for a, b, c, d in rows]


def reset_market(market):
    with _lock:
        con = connect()
        con.execute("DELETE FROM daily WHERE market=?", (market,))
        con.execute("DELETE FROM fetch_log WHERE market=?", (market,))
        con.commit()
        con.execute("VACUUM")
        con.close()


def fetched(market):
    con = connect()
    out = {d: s for d, s in con.execute(
        "SELECT date, status FROM fetch_log WHERE market=?", (market,))}
    con.close()
    return out


def save_shares(items, market):
    with _lock:
        con = connect()
        con.executemany("INSERT OR REPLACE INTO shares VALUES (?,?,?,?)",
                        [(c, s, market, now()) for c, s in items])
        con.commit()
        con.close()


def load_shares():
    con = connect()
    out = {c: s for c, s in con.execute("SELECT code, shares FROM shares")}
    con.close()
    return out


def codes(all_codes=False):
    """預設只回傳一般普通股（4 碼、非 00 開頭）；實盤對照用的 ETF 不進回測母體。"""
    con = connect()
    out = con.execute(
        "SELECT code, MAX(name), MAX(market) FROM daily GROUP BY code").fetchall()
    con.close()
    if all_codes:
        return out
    return [r for r in out if len(r[0]) == 4 and r[0].isdigit() and r[0][0] != "0"]


def series(code, con=None):
    own = con is None
    con = con or connect()
    rows = con.execute(
        "SELECT date, open, high, low, close, volume, value, change, name, market "
        "FROM daily WHERE code=? ORDER BY date", (code,)).fetchall()
    if own:
        con.close()
    return rows


def stats():
    con = connect()
    r = con.execute(
        "SELECT COUNT(DISTINCT code), MIN(date), MAX(date), COUNT(DISTINCT date) FROM daily").fetchone()
    by_mkt = dict(con.execute(
        "SELECT market, COUNT(DISTINCT code) FROM daily GROUP BY market").fetchall())
    ns = con.execute("SELECT COUNT(*) FROM shares").fetchone()[0]
    errs = con.execute(
        "SELECT date, market, status FROM fetch_log WHERE status LIKE 'error%' "
        "ORDER BY ts DESC LIMIT 8").fetchall()
    con.close()
    return {"stocks": r[0], "first": r[1], "last": r[2], "days": r[3],
            "by_market": by_mkt, "shares": ns,
            "errors": [{"date": d, "market": m, "status": s} for d, m, s in errs],
            "adj": adj_stats()}


def save_run(params, summary, result):
    with _lock:
        con = connect()
        cur = con.execute("INSERT INTO runs (created, params, summary, result) VALUES (?,?,?,?)",
                          (now(), json.dumps(params, ensure_ascii=False),
                           json.dumps(summary, ensure_ascii=False),
                           json.dumps(result, ensure_ascii=False)))
        rid = cur.lastrowid
        con.commit()
        con.close()
    return rid


def list_runs(limit=30):
    con = connect()
    rows = con.execute("SELECT id, created, params, summary FROM runs ORDER BY id DESC LIMIT ?",
                       (limit,)).fetchall()
    con.close()
    return [{"id": i, "created": c, "params": json.loads(p), "summary": json.loads(s)}
            for i, c, p, s in rows]


def get_run(rid):
    con = connect()
    r = con.execute("SELECT id, created, params, summary, result FROM runs WHERE id=?",
                    (rid,)).fetchone()
    con.close()
    if not r:
        return None
    return {"id": r[0], "created": r[1], "params": json.loads(r[2]),
            "summary": json.loads(r[3]), **json.loads(r[4])}


def delete_run(rid):
    with _lock:
        con = connect()
        con.execute("DELETE FROM runs WHERE id=?", (rid,))
        con.commit()
        con.close()


def get_setting(key, default=None):
    con = connect()
    r = con.execute("SELECT value FROM settings WHERE key=?", (key,)).fetchone()
    con.close()
    return json.loads(r[0]) if r else default


def set_setting(key, value):
    with _lock:
        con = connect()
        con.execute("INSERT OR REPLACE INTO settings VALUES (?,?)", (key, json.dumps(value, ensure_ascii=False)))
        con.commit()
        con.close()


def live_rows(table):
    con = connect()
    cur = con.execute(f"SELECT * FROM {table} ORDER BY date, id")
    cols = [c[0] for c in cur.description]
    out = [dict(zip(cols, r)) for r in cur.fetchall()]
    con.close()
    return out


def live_insert(table, row):
    with _lock:
        con = connect()
        keys = list(row)
        cur = con.execute(f"INSERT INTO {table} ({','.join(keys)}) VALUES ({','.join('?' * len(keys))})",
                          [row[k] for k in keys])
        rid = cur.lastrowid
        con.commit()
        con.close()
    return rid


def live_delete(table, rid):
    with _lock:
        con = connect()
        con.execute(f"DELETE FROM {table} WHERE id=?", (rid,))
        con.commit()
        con.close()
