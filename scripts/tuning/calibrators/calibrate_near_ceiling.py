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
from typing import Dict, Tuple

import numpy as np
import pandas as pd

from scripts.tuning.calibrators.policy_write import write_policy

BASE_DIR = Path(__file__).resolve().parents[3]
OUT_DIR = BASE_DIR / 'out'
ORDERS_DIR = OUT_DIR / 'orders'
DEFAULTS_PATH = BASE_DIR / 'config' / 'policy_default.json'
CONFIG_PATH = BASE_DIR / 'config' / 'policy_overrides.json'


def _load_json(p: Path) -> Dict:
    raw = p.read_text(encoding='utf-8')
    raw = re.sub(r"/\*.*?\*/", "", raw, flags=re.S)
    raw = re.sub(r"(^|\s)//.*$", "", raw, flags=re.M)
    raw = re.sub(r"(^|\s)#.*$", "", raw, flags=re.M)
    return json.loads(raw)


def _resolve_policy_paths() -> Tuple[Dict, Path]:
    """Return overlay policy object and path to update, with defaults for targets.

    - Read calibration_targets from policy_default.json (source of truth), and
      allow overlay to override if it provides its own calibration_targets.
    - Choose overlay path in priority order: out/orders/policy_overrides.json,
      else config/policy_overrides.json. Create parent if missing.
    """
    defaults = {}
    if DEFAULTS_PATH.exists():
        try:
            defaults = _load_json(DEFAULTS_PATH)
        except Exception:
            defaults = {}
    # Pick overlay policy path
    pol_path = ORDERS_DIR / 'policy_overrides.json'
    if not pol_path.exists():
        pol_path = CONFIG_PATH
    if pol_path.exists():
        overlay = _load_json(pol_path)
    else:
        overlay = {}
        pol_path.parent.mkdir(parents=True, exist_ok=True)
    # Merge calibration targets shallowly: defaults -> overlay
    def_targets = (defaults.get('calibration_targets') or {}).get('thresholds') or {}
    ov_targets = (overlay.get('calibration_targets') or {}).get('thresholds') or {}
    targets = {**def_targets, **ov_targets}
    overlay.setdefault('calibration_targets', {})
    overlay['calibration_targets'].setdefault('thresholds', {})
    # Store merged targets back for transparency (optional)
    overlay['calibration_targets']['thresholds'].update(targets)
    return overlay, pol_path


def calibrate(*, write: bool = False) -> float:
    snap_p = OUT_DIR / 'snapshot.csv'
    pre_p = OUT_DIR / 'presets_all.csv'
    if not (snap_p.exists() and pre_p.exists()):
        raise SystemExit('Missing snapshot/presets files for near_ceiling calibration')
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
    # Resolve quantile target (default 0.98) from calibration_targets.thresholds.near_ceiling_q
    overlay, pol_p = _resolve_policy_paths()
    try:
        q = float(((overlay.get('calibration_targets') or {}).get('thresholds') or {}).get('near_ceiling_q', 0.98))
    except Exception as exc:
        raise SystemExit(f'invalid calibration_targets.thresholds.near_ceiling_q: {exc}') from exc
    q = max(0.0, min(1.0, q))
    thr = float(np.quantile(r.to_numpy(), q))
    thr = max(0.90, min(0.999, thr))
    if write:
        pol = overlay if overlay else {}
        th = dict(pol.get('thresholds', {}) or {})
        th['near_ceiling_pct'] = float(thr)
        pol['thresholds'] = th
        write_policy(
            calibrator=__name__,
            policy=pol,
            explicit_path=pol_p,
        )
    return float(thr)


if __name__ == '__main__':
    v = calibrate(write=True)
    print(f"near_ceiling_pct={v:.3f}")
