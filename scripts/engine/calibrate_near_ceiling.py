from __future__ import annotations

"""
Calibrate thresholds.near_ceiling_pct from current market proximity to band ceilings.

Heuristic (objective, non‑intraday)
- For tickers with both current Price and BandCeiling_Tick, compute ratio r = Price / BandCeiling_Tick.
- Set near_ceiling_pct to a high percentile (e.g., 98th) of r, clipped to (0.90..0.999).

Inputs
- out/snapshot.csv (Ticker, Price)
- out/presets_all.csv (BandCeiling_Tick or BandCeilingRaw)
- out/orders/policy_overrides.json

Outputs
- thresholds.near_ceiling_pct in runtime policy.

Fail-fast
- Missing files/columns; insufficient usable pairs => SystemExit.
"""

from pathlib import Path
import json, re
from typing import Dict

import numpy as np
import pandas as pd

BASE_DIR = Path(__file__).resolve().parents[2]
OUT_DIR = BASE_DIR / 'out'
ORDERS_DIR = OUT_DIR / 'orders'


def _load_json(p: Path) -> Dict:
    raw = p.read_text(encoding='utf-8')
    raw = re.sub(r"/\*.*?\*/", "", raw, flags=re.S)
    raw = re.sub(r"(^|\s)//.*$", "", raw, flags=re.M)
    raw = re.sub(r"(^|\s)#.*$", "", raw, flags=re.M)
    return json.loads(raw)


def calibrate(*, write: bool = False) -> float:
    snap_p = OUT_DIR / 'snapshot.csv'
    pre_p = OUT_DIR / 'presets_all.csv'
    pol_p = ORDERS_DIR / 'policy_overrides.json'
    if not (snap_p.exists() and pre_p.exists() and pol_p.exists()):
        raise SystemExit('Missing snapshot/presets/policy files for near_ceiling calibration')
    snap = pd.read_csv(snap_p)
    pre = pd.read_csv(pre_p)
    if 'Ticker' not in snap.columns or 'Price' not in snap.columns:
        raise SystemExit('snapshot.csv missing Ticker/Price')
    if 'Ticker' not in pre.columns:
        raise SystemExit('presets_all.csv missing Ticker')
    pre = pre[['Ticker','BandCeiling_Tick','BandCeilingRaw'] if 'BandCeiling_Tick' in pre.columns else ['Ticker','BandCeilingRaw']]
    df = snap.merge(pre, on='Ticker', how='left')
    if df.empty:
        raise SystemExit('No rows to compute near_ceiling')
    ceil = pd.to_numeric(df.get('BandCeiling_Tick', df.get('BandCeilingRaw')), errors='coerce')
    price = pd.to_numeric(df['Price'], errors='coerce')
    r = (price / ceil).replace([np.inf, -np.inf], np.nan).dropna()
    r = r[r > 0]
    if r.empty:
        raise SystemExit('Insufficient ceiling/price data')
    thr = float(np.quantile(r.to_numpy(), 0.98))
    thr = max(0.90, min(0.999, thr))
    if write:
        pol = _load_json(pol_p)
        th = dict(pol.get('thresholds', {}) or {})
        th['near_ceiling_pct'] = float(thr)
        pol['thresholds'] = th
        pol_p.write_text(json.dumps(pol, ensure_ascii=False, indent=2), encoding='utf-8')
    return float(thr)


if __name__ == '__main__':
    v = calibrate(write=True)
    print(f"near_ceiling_pct={v:.3f}")

