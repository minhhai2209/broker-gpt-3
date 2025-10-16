#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")"/../.. && pwd)"
cd "$ROOT_DIR"

LOG_DIR="out/logs"
mkdir -p "$LOG_DIR"

if [[ -d venv ]]; then
  source venv/bin/activate
fi

echo "[collect] Global factors"
python scripts/data_fetching/collect_global_factors.py | tee "$LOG_DIR/collect_global_factors.log"

echo "[collect] Vietstock fundamentals (VN100 from data/industry_map.csv)"
python scripts/data_fetching/collect_vietstock_fundamentals.py --skip-missing --delay 0.3 --out data/fundamentals_vietstock.csv | tee "$LOG_DIR/collect_vietstock_fundamentals.log"

echo "[collect] Vietstock events calendar (VN100 from data/industry_map.csv)"
python scripts/data_fetching/collect_vietstock_events.py --skip-missing --delay 0.2 --out data/events_calendar.csv | tee "$LOG_DIR/collect_vietstock_events.log"

echo "[collect] Intraday latest snapshot (best-effort)"
python - "$LOG_DIR" << 'PY'
import pandas as pd
from scripts.data_fetching.collect_intraday import ensure_intraday_latest

tickers = pd.read_csv('data/industry_map.csv')['Ticker'].astype(str).str.upper().tolist()
tickers = [t for t in tickers if t not in {'VNINDEX','VN30','VN100'}]
ensure_intraday_latest(tickers, outdir='out/intraday', window_minutes=12*60)
print('Intraday snapshot written to out/intraday/latest.csv (if data available)')
PY

echo "[done] All collectors finished"

