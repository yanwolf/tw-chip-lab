#!/usr/bin/env bash
# 本機驗證：離線測試 + 啟動伺服器打幾個 API + 前端 JS 語法檢查
set -euo pipefail
cd "$(dirname "$0")"
python3 tests/test_core.py
TMP=$(mktemp -d)
DATA_DIR=$TMP PORT=8765 python3 -m app.server & PID=$!
trap "kill $PID 2>/dev/null; rm -rf $TMP" EXIT
sleep 1.5
curl -sf localhost:8765/healthz >/dev/null && echo "  ✓ healthz"
curl -sf localhost:8765/api/status | python3 -c "import json,sys;d=json.load(sys.stdin);assert 'defaults' in d" && echo "  ✓ /api/status"
[[ "$(curl -sf localhost:8765/)" == *籌碼峰實驗室* ]] && echo "  ✓ 首頁"
if command -v node >/dev/null; then
  python3 -c "import re;s=open('app/static/index.html').read();open('$TMP/a.js','w').write(re.findall(r'<script>(.*?)</script>',s,re.S)[0])"
  node --check "$TMP/a.js" && echo "  ✓ 前端 JS 語法"
fi
echo "verify OK"
