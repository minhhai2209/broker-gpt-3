from __future__ import annotations

"""
Calibrate market_filter.risk_off_breadth_floor from historical breadth distribution.

Definition
- Breadth (daily) = fraction of non-index tickers with Close > MA50 on that date.
  Use available universe in out/prices_history.csv. Ignore tickers with <50 bars
  for MA50 at a date.

Inputs
- out/prices_history.csv (must include Date, Ticker, Close)
- calibration_targets.market_filter.breadth_floor_q (e.g., 0.40)

Output
- Writes market_filter.risk_off_breadth_floor to config.

Fail-fast on missing files/columns or insufficient data.
"""

from pathlib import Path
import json
import re

import numpy as np
import pandas as pd


BASE_DIR = Path(__file__).resolve().parents[2]
OUT_DIR = BASE_DIR / 'out'
ORDERS_PATH = OUT_DIR / 'orders' / 'policy_overrides.json'
CONFIG_PATH = BASE_DIR / 'config' / 'policy_overrides.json'
DEFAULTS_PATH = BASE_DIR / 'config' / 'policy_default.json'


def _strip(s: str) -> str:
    s = re.sub(r"/\*.*?\*/", "", s, flags=re.S)
    s = re.sub(r"(^|\s)//.*$", "", s, flags=re.M)
    s = re.sub(r"(^|\s)#.*$", "", s, flags=re.M)
    return s


def _load_policy() -> dict:
    src = ORDERS_PATH if ORDERS_PATH.exists() else CONFIG_PATH
    if not src.exists():
        raise SystemExit(f'Missing policy file: {src}')
    return json.loads(_strip(src.read_text(encoding='utf-8')))


def _save_policy(obj: dict) -> None:
    target = ORDERS_PATH if ORDERS_PATH.exists() else CONFIG_PATH
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(obj, ensure_ascii=False, indent=2), encoding='utf-8')


def _load_history() -> pd.DataFrame:
    p = OUT_DIR / 'prices_history.csv'
    if not p.exists():
        raise SystemExit('Missing out/prices_history.csv for breadth calibration')
    df = pd.read_csv(p)
    must = {'Date','Ticker','Close'}
    if not must.issubset(df.columns):
        miss = ', '.join(sorted(must - set(df.columns)))
        raise SystemExit(f'prices_history.csv missing columns: {miss}')
    df['Date'] = pd.to_datetime(df['Date'], errors='coerce')
    df = df.dropna(subset=['Date'])
    df['Ticker'] = df['Ticker'].astype(str).str.upper()
    # drop indices
    idx = {'VNINDEX','VN30','VN100'}
    df = df[~df['Ticker'].isin(idx)]
    if df.empty:
        raise SystemExit('No non-index tickers in history for breadth calibration')
    return df


def _compute_breadth_series(df: pd.DataFrame) -> pd.Series:
    # Pivot per ticker time series
    piv = df.pivot(index='Date', columns='Ticker', values='Close').sort_index()
    if piv.empty or piv.shape[1] == 0:
        raise SystemExit('Insufficient data to compute breadth series')
    ma50 = piv.rolling(50, min_periods=50).mean()
    above = (piv > ma50)
    valid = ma50.notna()
    # For each date, breadth = (# above) / (# valid tickers with MA50)
    num = above.sum(axis=1)
    den = valid.sum(axis=1)
    with np.errstate(divide='ignore', invalid='ignore'):
        frac = (num / den).replace([np.inf, -np.inf], np.nan)
    frac = frac.dropna()
    if frac.empty:
        raise SystemExit('Breadth series empty (not enough MA50 data)')
    return frac


def calibrate(write: bool = False) -> float:
    pol = _load_policy()
    tgt = (pol.get('calibration_targets', {}) or {}).get('market_filter', {})
    if 'breadth_floor_q' not in tgt:
        raise SystemExit('Missing calibration_targets.market_filter.breadth_floor_q')
    try:
        q = float(tgt['breadth_floor_q'])
    except Exception as exc:
        raise SystemExit(f'invalid breadth_floor_q: {exc}') from exc
    if not (0.0 <= q <= 1.0):
        raise SystemExit('breadth_floor_q must be in [0,1]')
    hist = _load_history()
    series = _compute_breadth_series(hist)
    floor = float(np.quantile(series.to_numpy(), q))
    if write:
        obj = pol
        mf = dict(obj.get('market_filter', {}) or {})
        mf['risk_off_breadth_floor'] = float(floor)
        obj['market_filter'] = mf
        _save_policy(obj)
    return float(floor)


def main():
    # Always write tuned value to runtime overrides; baseline is updated by nightly merge
    v = calibrate(write=True)
    print(f"[calibrate.breadth] risk_off_breadth_floor={v:.2f}")


if __name__ == '__main__':
    import argparse
    main()
