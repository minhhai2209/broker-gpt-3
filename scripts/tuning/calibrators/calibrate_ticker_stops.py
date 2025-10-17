from __future__ import annotations

"""
Calibrator: per‑ticker TP/SL overrides based on long‑term signals.

Inputs (built by pipeline before tuning):
- out/snapshot.csv (Ticker, Price)
- out/metrics.csv (Ticker, ATR14_Pct, MomRet_12_1 [optional])
- out/presets_all.csv (Ticker, MA20, MA50)

Rule of thumb (initial defaults):
- Loose (trend up & ATR percentile <= 0.60 & momentum >= 0): sl_atr_mult=2.5, sl_floor_pct=0.03
- Normal (else, ATR percentile <= 0.90): sl_atr_mult=2.0, sl_floor_pct=0.025
- Tight (ATR percentile > 0.90 OR trend down OR momentum < 0): sl_atr_mult=1.5, sl_floor_pct=0.02

All categories set sl_rule='dynamic_only' and clamp via optional sl_cap_pct (0.10/0.08/0.06).

Output:
- Writes per‑ticker overrides under policy.ticker_overrides.{TICKER} with keys:
  sl_rule, sl_atr_mult, sl_floor_pct, sl_cap_pct

This is conservative: it only shapes SL distance; it does not weaken the hard‑SL exit rule.
"""

from pathlib import Path
from typing import Dict, Optional
import json
import math

import numpy as np
import pandas as pd

from scripts.tuning.calibrators.policy_write import write_policy

BASE_DIR = Path(__file__).resolve().parents[3]
OUT_DIR = BASE_DIR / 'out'
ORDERS_DIR = OUT_DIR / 'orders'


def _load_csv(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise SystemExit(f"Missing required input: {path}")
    df = pd.read_csv(path)
    if df.empty:
        raise SystemExit(f"Empty input: {path}")
    return df


def _atr_percentile(series: pd.Series) -> pd.Series:
    s = pd.to_numeric(series, errors='coerce').fillna(np.nan)
    arr = s.to_numpy(copy=False)
    ranks = np.argsort(np.argsort(np.argsort(np.nan_to_num(arr, nan=np.inf))))  # stable rank but handle NaN
    # Fallback safe if all NaN
    if np.all(~np.isfinite(arr)):
        return pd.Series([0.5] * len(series), index=series.index, dtype=float)
    n = float(np.sum(np.isfinite(arr)))
    pct = np.zeros_like(arr, dtype=float)
    # Use nanpercentile approach: rank among finite values only
    finite_idx = np.isfinite(arr)
    order = np.argsort(arr[finite_idx])
    pct_vals = np.zeros(int(n), dtype=float)
    if n > 1:
        pct_vals = np.linspace(0.0, 1.0, int(n), endpoint=True)
    pct[finite_idx] = pct_vals[np.argsort(order)]
    pct[~finite_idx] = 0.5
    return pd.Series(pct, index=series.index, dtype=float)


def _classify_row(price: float, ma50: Optional[float], atr_pctile: float, mom: Optional[float]) -> str:
    trend_up = (ma50 is not None and price is not None and float(price) >= float(ma50))
    mom_ok = (mom is not None and float(mom) >= 0.0)
    if trend_up and atr_pctile <= 0.60 and mom_ok:
        return 'loose'
    if atr_pctile > 0.90 or (not trend_up) or (mom is not None and float(mom) < 0.0):
        return 'tight'
    return 'normal'


def _profile_params(kind: str) -> Dict[str, float | str]:
    if kind == 'loose':
        return {'sl_rule': 'dynamic_only', 'sl_atr_mult': 2.5, 'sl_floor_pct': 0.03, 'sl_cap_pct': 0.10}
    if kind == 'tight':
        return {'sl_rule': 'dynamic_only', 'sl_atr_mult': 1.5, 'sl_floor_pct': 0.02, 'sl_cap_pct': 0.06}
    return {'sl_rule': 'dynamic_only', 'sl_atr_mult': 2.0, 'sl_floor_pct': 0.025, 'sl_cap_pct': 0.08}


def calibrate(*, write: bool = False,
              snapshot_path: Optional[Path] = None,
              metrics_path: Optional[Path] = None,
              presets_path: Optional[Path] = None,
              explicit_policy_path: Optional[Path] = None) -> Dict[str, Dict[str, object]]:
    snap_p = snapshot_path or (OUT_DIR / 'snapshot.csv')
    met_p = metrics_path or (OUT_DIR / 'metrics.csv')
    pre_p = presets_path or (OUT_DIR / 'presets_all.csv')
    snap = _load_csv(snap_p).set_index('Ticker')
    met = _load_csv(met_p).set_index('Ticker')
    pre = _load_csv(pre_p).set_index('Ticker')
    if 'ATR14_Pct' not in met.columns:
        raise SystemExit("Missing ATR14_Pct in metrics.csv for ticker stop calibration")
    # Join needed columns
    df = pd.DataFrame(index=sorted(set(snap.index) & set(met.index)))
    df['Price'] = pd.to_numeric(snap.reindex(df.index)['Price'], errors='coerce')
    df['MA50'] = pd.to_numeric(pre.reindex(df.index)['MA50'], errors='coerce')
    df['ATR14_Pct'] = pd.to_numeric(met.reindex(df.index)['ATR14_Pct'], errors='coerce')
    mom = None
    if 'MomRet_12_1' in met.columns:
        mom = pd.to_numeric(met.reindex(df.index)['MomRet_12_1'], errors='coerce')
    df['ATR_pctile'] = _atr_percentile(df['ATR14_Pct'])
    overrides: Dict[str, Dict[str, object]] = {}
    for t, row in df.iterrows():
        try:
            kind = _classify_row(float(row.get('Price')), float(row.get('MA50')), float(row.get('ATR_pctile')), None if mom is None else float(mom.loc[t]))
        except Exception:
            kind = 'normal'
        overrides[str(t).upper()] = _profile_params(kind)
    if not write:
        return overrides
    # Persist into policy_overrides.json (runtime copy preferred)
    pol_path = explicit_policy_path or (ORDERS_DIR / 'policy_overrides.json')
    if not pol_path.exists():
        # Fallback to config copy if present
        cfg_p = BASE_DIR / 'config' / 'policy_overrides.json'
        if cfg_p.exists():
            pol_path = cfg_p
        else:
            raise SystemExit('Missing policy_overrides.json to write ticker_overrides')
    try:
        pol = json.loads(pol_path.read_text(encoding='utf-8'))
    except Exception as exc:
        raise SystemExit(f'Invalid JSON in {pol_path}: {exc}') from exc
    cur = dict(pol.get('ticker_overrides', {}) or {})
    cur.update(overrides)
    pol['ticker_overrides'] = cur
    write_policy(calibrator=__name__, policy=pol, explicit_path=pol_path)
    return overrides


if __name__ == '__main__':
    calibrate(write=True)

