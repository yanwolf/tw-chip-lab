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


def codes():
    con = connect()
    out = con.execute(
        "SELECT code, MAX(name), MAX(market) FROM daily GROUP BY code").fetchall()
    con.close()
    return out


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
        "ORDER BY date DESC LIMIT 5").fetchall()
    con.close()
    return {"stocks": r[0], "first": r[1], "last": r[2], "days": r[3],
            "by_market": by_mkt, "shares": ns,
            "errors": [{"date": d, "market": m, "status": s} for d, m, s in errs]}


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
