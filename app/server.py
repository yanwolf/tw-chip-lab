"""tw-chip-lab 網頁服務（純標準庫 http.server）。

環境變數：
  PORT          預設 8000
  DATA_DIR      SQLite 位置，預設 /data（要掛 Volume）
  APP_PASSWORD  設了就需要登入
  AUTO_UPDATE   =1 時平日 16:30 後自動補抓當日資料
  FETCH_INTERVAL 每個請求間隔秒數，預設 4（太短會被證交所封鎖）
"""
import hashlib
import json
import logging
import os
import threading
from datetime import date, timedelta
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import parse_qs, urlparse

from . import backtest, db, diag, fetch

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(message)s")
STATIC = os.path.join(os.path.dirname(__file__), "static")
PASSWORD = os.environ.get("APP_PASSWORD", "")
TOKEN = hashlib.sha256(("tw-chip-lab:" + PASSWORD).encode()).hexdigest() if PASSWORD else ""


class H(BaseHTTPRequestHandler):
    def log_message(self, fmt, *a):
        pass

    def _send(self, code, body, ctype="application/json; charset=utf-8", headers=None):
        if not isinstance(body, (bytes, bytearray)):
            body = json.dumps(body, ensure_ascii=False).encode()
        self.send_response(code)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        for k, v in (headers or {}).items():
            self.send_header(k, v)
        self.end_headers()
        self.wfile.write(body)

    def _authed(self):
        if not PASSWORD:
            return True
        ck = self.headers.get("Cookie", "")
        return any(p.strip() == f"auth={TOKEN}" for p in ck.split(";"))

    def _body(self):
        n = int(self.headers.get("Content-Length") or 0)
        if not n:
            return {}
        try:
            return json.loads(self.rfile.read(n))
        except ValueError:
            return {}

    def do_GET(self):
        u = urlparse(self.path)
        q = {k: v[0] for k, v in parse_qs(u.query).items()}
        if u.path in ("/", "/index.html"):
            with open(os.path.join(STATIC, "index.html"), "rb") as f:
                return self._send(200, f.read(), "text/html; charset=utf-8")
        if u.path == "/healthz":
            return self._send(200, {"ok": True})
        if not u.path.startswith("/api/"):
            return self._send(404, {"error": "not found"})
        if u.path == "/api/auth":
            return self._send(200, {"need": bool(PASSWORD), "ok": self._authed()})
        if not self._authed():
            return self._send(401, {"error": "需要登入"})
        try:
            if u.path == "/api/status":
                return self._send(200, {"data": db.stats(), "fetch": fetch.JOB.state,
                                        "backtest": backtest.JOB.state,
                                        "defaults": backtest.DEFAULTS,
                                        "auto_update": os.environ.get("AUTO_UPDATE") == "1"})
            if u.path == "/api/runs":
                return self._send(200, db.list_runs())
            if u.path.startswith("/api/runs/"):
                r = db.get_run(int(u.path.rsplit("/", 1)[1]))
                return self._send(200 if r else 404, r or {"error": "找不到這次回測"})
            if u.path == "/api/diag":
                return self._send(200, {"state": diag.JOB.state, "result": diag.last_result()})
            if u.path == "/api/chips":
                code = (q.get("code") or "").strip()
                v = backtest.chip_view(code, q.get("date") or None,
                                       float(q.get("decay") or 1.0))
                return self._send(200 if v else 404,
                                  v or {"error": f"{code} 沒有足夠資料（至少 30 個交易日）"})
        except Exception as e:
            logging.exception("GET %s", u.path)
            return self._send(500, {"error": str(e)})
        return self._send(404, {"error": "not found"})

    def do_POST(self):
        u = urlparse(self.path)
        b = self._body()
        if u.path == "/api/login":
            if PASSWORD and b.get("password") == PASSWORD:
                return self._send(200, {"ok": True}, headers={
                    "Set-Cookie": f"auth={TOKEN}; Path=/; Max-Age=2592000; HttpOnly; SameSite=Lax"})
            return self._send(401, {"error": "密碼錯誤"})
        if not self._authed():
            return self._send(401, {"error": "需要登入"})
        try:
            if u.path == "/api/fetch":
                start = date.fromisoformat(b.get("start") or
                                           (date.today() - timedelta(days=365 * 3)).isoformat())
                end = date.fromisoformat(b["end"]) if b.get("end") else None
                ok = fetch.JOB.start(start, end)
                return self._send(200 if ok else 409, {"ok": ok} if ok else {"error": "抓取已在進行中"})
            if u.path == "/api/fetch/stop":
                fetch.JOB.stop()
                return self._send(200, {"ok": True})
            if u.path == "/api/diag":
                ok = diag.JOB.start()
                return self._send(200 if ok else 409, {"ok": ok} if ok else {"error": "檢查已在進行中"})
            if u.path == "/api/shares":
                return self._send(200, fetch.fetch_shares())
            if u.path == "/api/backtest":
                ok = backtest.JOB.start(b)
                return self._send(200 if ok else 409, {"ok": ok} if ok else {"error": "回測已在進行中"})
            if u.path.startswith("/api/runs/") and u.path.endswith("/delete"):
                db.delete_run(int(u.path.split("/")[3]))
                return self._send(200, {"ok": True})
        except Exception as e:
            logging.exception("POST %s", u.path)
            return self._send(400, {"error": str(e)})
        return self._send(404, {"error": "not found"})


def main():
    db.init()
    if os.environ.get("AUTO_UPDATE") == "1":
        threading.Thread(target=fetch.auto_update_loop, daemon=True).start()
    port = int(os.environ.get("PORT", "8000"))
    logging.info("tw-chip-lab listening on :%d, db=%s", port, db.DB_PATH)
    ThreadingHTTPServer(("0.0.0.0", port), H).serve_forever()


if __name__ == "__main__":
    main()
