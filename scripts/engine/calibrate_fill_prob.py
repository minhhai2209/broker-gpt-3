from __future__ import annotations

"""
Calibrate pricing.fill_prob decay scale from median ATR (in ticks).

Rationale
- Fill probability decay with distance (in ticks) should decay over an order of
  the typical ATR in ticks. We set decay_scale_min_ticks to median(ATR_ticks)
  across the actionable universe; keep other parameters if present.

Inputs
- out/metrics.csv (ATR14_Pct, TickSizeHOSE_Thousand)
- out/snapshot.csv (Price)
- out/orders/policy_overrides.json

Outputs
- pricing.fill_prob.decay_scale_min_ticks (and preserves base/cross/near/min).

Fail-fast
- Missing files/columns or empty series.
"""

from pathlib import Path
import json, re
import numpy as np
import pandas as pd

BASE_DIR = Path(__file__).resolve().parents[2]
OUT_DIR = BASE_DIR / 'out'
ORDERS_DIR = OUT_DIR / 'orders'


def _load_json(p: Path) -> dict:
    raw = p.read_text(encoding='utf-8')
    raw = re.sub(r"/\*.*?\*/", "", raw, flags=re.S)
    raw = re.sub(r"(^|\s)//.*$", "", raw, flags=re.M)
    raw = re.sub(r"(^|\s)#.*$", "", raw, flags=re.M)
    return json.loads(raw)


def calibrate(*, write: bool = False) -> dict:
    met_p = OUT_DIR / 'metrics.csv'
    snap_p = OUT_DIR / 'snapshot.csv'
    pol_p = ORDERS_DIR / 'policy_overrides.json'
    if not (met_p.exists() and snap_p.exists() and pol_p.exists()):
        raise SystemExit('Missing metrics/snapshot/policy files for fill_prob calibration')
    dfm = pd.read_csv(met_p)
    dfs = pd.read_csv(snap_p)
    for c in ('Ticker','ATR14_Pct','TickSizeHOSE_Thousand'):
        if c not in dfm.columns:
            raise SystemExit(f'metrics.csv missing {c}')
    if 'Ticker' not in dfs.columns or 'Price' not in dfs.columns:
        raise SystemExit('snapshot.csv missing Ticker/Price')
    df = dfm[['Ticker','ATR14_Pct','TickSizeHOSE_Thousand']].merge(dfs[['Ticker','Price']], on='Ticker', how='left')
    df['atr_ticks'] = (pd.to_numeric(df['ATR14_Pct'], errors='coerce')/100.0) * pd.to_numeric(df['Price'], errors='coerce') / pd.to_numeric(df['TickSizeHOSE_Thousand'], errors='coerce')
    atr_ticks = df['atr_ticks'].replace([np.inf,-np.inf], np.nan).dropna()
    atr_ticks = atr_ticks[atr_ticks > 0]
    if atr_ticks.empty:
        raise SystemExit('Unable to compute ATR in ticks for any ticker')
    decay = float(np.median(atr_ticks.to_numpy()))
    decay = float(max(3.0, min(20.0, round(decay, 1))))
    pol = _load_json(pol_p)
    pr = dict(pol.get('pricing', {}) or {})
    fp = dict(pr.get('fill_prob', {}) or {})
    # Preserve existing values when present; ensure all keys exist
    base = float(fp.get('base', 0.30) or 0.30)
    cross = float(fp.get('cross', 0.90) or 0.90)
    near = float(fp.get('near_ceiling', 0.05) or 0.05)
    minp = float(fp.get('min', 0.05) or 0.05)
    out = {
        'base': base,
        'cross': cross,
        'near_ceiling': near,
        'min': minp,
        'decay_scale_min_ticks': decay,
    }
    if write:
        fp2 = dict(out)
        pr['fill_prob'] = fp2
        pol['pricing'] = pr
        pol_p.write_text(json.dumps(pol, ensure_ascii=False, indent=2), encoding='utf-8')
    return out


if __name__ == '__main__':
    vals = calibrate(write=True)
    print('[calibrate.fill_prob] ' + ', '.join(f"{k}={v}" for k, v in vals.items()))

