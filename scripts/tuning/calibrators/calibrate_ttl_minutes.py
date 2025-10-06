from __future__ import annotations

"""
Calibrate orders_ui.ttl_minutes based on current volatility regime.

Heuristic (objective, no fill logs needed)
- Use VNINDEX annualized volatility and ATR percentile thresholds from policy
  to set a conservative base and tighter TTLs under higher volatility.
  Mapping: vol ≤25% → 14', ≤40% → 11', else 8'; soft/hard trails base by ~3/5'.

Inputs
- out/prices_history.csv (for volatility and ATR percentile via engine logic), or metrics/session files
- out/orders/policy_overrides.json (market_filter index_atr soft/hard)

Outputs
- orders_ui.ttl_minutes.{base,soft,hard}

Fail-fast
- Missing policy or history files.
"""

from pathlib import Path
import json, re
import numpy as np
import pandas as pd

from scripts.engine.volatility import garman_klass_sigma, percentile_thresholds

BASE_DIR = Path(__file__).resolve().parents[3]
OUT_DIR = BASE_DIR / 'out'
ORDERS_DIR = OUT_DIR / 'orders'


def _load_json(p: Path) -> dict:
    raw = p.read_text(encoding='utf-8')
    raw = re.sub(r"/\*.*?\*/", "", raw, flags=re.S)
    raw = re.sub(r"(^|\s)//.*$", "", raw, flags=re.M)
    raw = re.sub(r"(^|\s)#.*$", "", raw, flags=re.M)
    return json.loads(raw)


def _prepare_vnindex(ph: pd.DataFrame) -> pd.DataFrame:
    ph = ph.copy()
    ph['Ticker'] = ph['Ticker'].astype(str).str.upper()
    vn = ph[ph['Ticker'] == 'VNINDEX'].copy()
    if vn.empty:
        raise SystemExit('prices_history has no VNINDEX rows for TTL calibration')
    vn['Date'] = pd.to_datetime(vn['Date'], errors='coerce')
    vn = vn.dropna(subset=['Date']).sort_values('Date')
    keep_cols = ['Date', 'Open', 'High', 'Low', 'Close']
    for col in keep_cols:
        if col not in vn.columns:
            vn[col] = np.nan
    return vn[keep_cols].reset_index(drop=True)


def _latest_bucket(vn: pd.DataFrame) -> tuple[str, dict, dict]:
    if vn.empty:
        return 'medium', {}, {}
    sigma = garman_klass_sigma(vn)
    if sigma.empty:
        return 'medium', {}, {}
    # Use last ~252 business days when available for robust thresholds
    sigma_clean = sigma.dropna()
    if len(sigma_clean) > 260:
        sigma_clean = sigma_clean.tail(260)
    p75, p95 = percentile_thresholds(sigma_clean, (0.75, 0.95))
    latest = float(sigma_clean.iloc[-1]) if not sigma_clean.empty else float('nan')
    thresholds = {
        'p75': float(p75) if np.isfinite(p75) else float('nan'),
        'p95': float(p95) if np.isfinite(p95) else float('nan'),
        'latest': latest if np.isfinite(latest) else float('nan'),
    }
    bucket = 'medium'
    if np.isfinite(latest) and np.isfinite(p95) and latest >= p95:
        bucket = 'high'
    elif np.isfinite(latest) and np.isfinite(p75) and latest >= p75:
        bucket = 'medium'
    elif np.isfinite(latest):
        bucket = 'low'
    bucket_minutes = {
        'low': {'base': 14, 'soft': 11, 'hard': 8},
        'medium': {'base': 11, 'soft': 9, 'hard': 7},
        'high': {'base': 8, 'soft': 6, 'hard': 5},
    }
    return bucket, bucket_minutes, thresholds


def calibrate(*, write: bool = False) -> tuple[int, int, int]:
    pol_p = ORDERS_DIR / 'policy_overrides.json'
    ph_p = OUT_DIR / 'prices_history.csv'
    if not pol_p.exists() or not ph_p.exists():
        raise SystemExit('Missing policy or prices_history for ttl calibration')
    pol = _load_json(pol_p)
    ph = pd.read_csv(ph_p)
    for c in ('Date','Ticker','Close'):
        if c not in ph.columns:
            raise SystemExit('prices_history.csv missing required columns')
    vn = _prepare_vnindex(ph)
    bucket, bucket_minutes, thresholds = _latest_bucket(vn)
    current = bucket_minutes.get(bucket, bucket_minutes.get('medium', {'base': 11, 'soft': 9, 'hard': 7}))
    base = int(current.get('base', 11))
    soft = int(current.get('soft', max(base - 2, 3)))
    hard = int(current.get('hard', max(soft - 2, 2)))
    baseline = {'base': 12, 'soft': 9, 'hard': 7}
    if write:
        ou = dict(pol.get('orders_ui', {}) or {})
        ttl = dict(current)
        ttl['base'] = int(base)
        ttl['soft'] = int(soft)
        ttl['hard'] = int(hard)
        ou['ttl_minutes'] = ttl
        ou['ttl_bucket_minutes'] = bucket_minutes
        ou['ttl_bucket_thresholds'] = thresholds
        ou['ttl_bucket_state'] = {'current': bucket, 'sigma': thresholds.get('latest')}
        ou['ttl_minutes_baseline'] = baseline
        pol['orders_ui'] = ou
        pol_p.write_text(json.dumps(pol, ensure_ascii=False, indent=2), encoding='utf-8')
    return int(base), int(soft), int(hard)


if __name__ == '__main__':
    b,s,h = calibrate(write=True)
    print(f"ttl base/soft/hard = {b}/{s}/{h}")

