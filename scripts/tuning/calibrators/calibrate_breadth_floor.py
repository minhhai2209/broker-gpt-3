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
import argparse
import json
import re
from typing import Any, Mapping

import numpy as np
import pandas as pd


BASE_DIR = Path(__file__).resolve().parents[3]
OUT_DIR = BASE_DIR / 'out'
ORDERS_PATH = OUT_DIR / 'orders' / 'policy_overrides.json'
CONFIG_PATH = BASE_DIR / 'config' / 'policy_overrides.json'
DEFAULTS_PATH = BASE_DIR / 'config' / 'policy_default.json'


def _strip(s: str) -> str:
    s = re.sub(r"/\*.*?\*/", "", s, flags=re.S)
    s = re.sub(r"(^|\s)//.*$", "", s, flags=re.M)
    s = re.sub(r"(^|\s)#.*$", "", s, flags=re.M)
    return s


def _load_json(path: Path, *, required: bool = True) -> dict[str, Any]:
    if not path.exists():
        if required:
            raise SystemExit(f'Missing policy file: {path}')
        return {}
    try:
        return json.loads(_strip(path.read_text(encoding='utf-8')))
    except json.JSONDecodeError as exc:
        raise SystemExit(f'Policy file {path} is not valid JSON: {exc}') from exc


def _resolve_policy_paths(policy_path: Path | None) -> tuple[dict[str, Any], dict[str, Any], Path]:
    """Return baseline defaults, overlay policy, and the file we will update."""

    defaults = _load_json(DEFAULTS_PATH, required=True)

    if policy_path is not None:
        path = policy_path
        overlay = _load_json(path, required=False)
        if not overlay:
            # Ensure parent exists when the caller points to a new file.
            path.parent.mkdir(parents=True, exist_ok=True)
        return defaults, overlay, path

    for candidate in (ORDERS_PATH, CONFIG_PATH):
        if candidate.exists():
            overlay = _load_json(candidate, required=True)
            return defaults, overlay, candidate

    # Fallback: create overrides file if none exist yet.
    CONFIG_PATH.parent.mkdir(parents=True, exist_ok=True)
    return defaults, {}, CONFIG_PATH


def _save_policy(obj: Mapping[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, ensure_ascii=False, indent=2), encoding='utf-8')


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


def _weighted_quantile(series: pd.Series, q: float, half_life_days: float | None) -> float:
    values = series.to_numpy(dtype=float)
    if values.size == 0:
        raise SystemExit('Breadth series empty (not enough MA50 data)')

    if half_life_days is None or half_life_days <= 0:
        return float(np.quantile(values, q))

    idx = series.index
    if isinstance(idx, pd.DatetimeIndex):
        delta_days = ((idx[-1] - idx) / np.timedelta64(1, 'D')).astype(float)
    else:
        delta_days = (np.arange(len(values))[-1] - np.arange(len(values))).astype(float)

    lam = np.log(2.0) / float(half_life_days)
    weights = np.exp(-lam * delta_days)
    weights = np.asarray(weights, dtype=float)
    mask = np.isfinite(values) & np.isfinite(weights)
    if not mask.any():
        raise SystemExit('No finite data available for weighted quantile')

    values = values[mask]
    weights = weights[mask]
    if not (weights > 0).any():
        raise SystemExit('Invalid weights for weighted quantile (all zero)')

    sorter = np.argsort(values)
    values = values[sorter]
    weights = weights[sorter]

    total = float(weights.sum())
    if not np.isfinite(total) or total <= 0:
        raise SystemExit('Invalid cumulative weight during weighted quantile calculation')
    weights = weights / total
    cum = np.cumsum(weights)

    if q <= 0.0:
        return float(values[0])
    if q >= 1.0:
        return float(values[-1])

    idx_high = int(np.searchsorted(cum, q, side='left'))
    idx_high = min(idx_high, values.size - 1)
    if idx_high == 0:
        return float(values[0])

    idx_low = idx_high - 1
    w_low = cum[idx_low]
    w_high = cum[idx_high]
    if not np.isfinite(w_low):
        w_low = 0.0
    if w_high <= w_low:
        return float(values[idx_high])
    ratio = (q - w_low) / (w_high - w_low)
    return float(values[idx_low] + ratio * (values[idx_high] - values[idx_low]))


def _merge_dict(base: Mapping[str, Any] | None, override: Mapping[str, Any] | None) -> dict[str, Any]:
    result: dict[str, Any] = {}
    if base:
        result.update(base)
    if override:
        result.update(override)
    return result


def _extract_targets(defaults: Mapping[str, Any], overlay: Mapping[str, Any]) -> dict[str, Any]:
    defaults_targets = ((defaults.get('calibration_targets') or {}).get('market_filter') or {})
    overlay_targets = ((overlay.get('calibration_targets') or {}).get('market_filter') or {})
    targets = _merge_dict(defaults_targets, overlay_targets)
    if 'breadth_floor_q' not in targets:
        raise SystemExit('Missing calibration_targets.market_filter.breadth_floor_q')
    return targets


def calibrate(*, write: bool = False, policy_path: Path | None = None) -> float:
    defaults, overlay, target_path = _resolve_policy_paths(policy_path)
    targets = _extract_targets(defaults, overlay)

    try:
        q = float(targets['breadth_floor_q'])
    except Exception as exc:
        raise SystemExit(f'invalid breadth_floor_q: {exc}') from exc
    if not (0.0 <= q <= 1.0):
        raise SystemExit('breadth_floor_q must be in [0,1]')

    def _clean(name: str, bounds: tuple[float, float] | None = None) -> float | None:
        raw = targets.get(name)
        if raw is None:
            return None
        try:
            value = float(raw)
        except Exception as exc:
            raise SystemExit(f'invalid {name}: {exc}') from exc
        if bounds is not None and not (bounds[0] <= value <= bounds[1]):
            raise SystemExit(f'{name} must be within [{bounds[0]}, {bounds[1]}]')
        return value

    half_life = _clean('breadth_floor_half_life_days')
    floor_min = _clean('breadth_floor_min', (0.0, 1.0))
    floor_max = _clean('breadth_floor_max', (0.0, 1.0))

    if half_life is not None and half_life <= 0.0:
        half_life = None

    if floor_min is not None and floor_max is not None and floor_min > floor_max:
        raise SystemExit('breadth_floor_min cannot exceed breadth_floor_max')

    hist = _load_history()
    series = _compute_breadth_series(hist)
    floor = _weighted_quantile(series, q, half_life)
    if floor_min is not None:
        floor = max(floor, floor_min)
    if floor_max is not None:
        floor = min(floor, floor_max)

    if write:
        updated = dict(overlay)
        mf = _merge_dict(overlay.get('market_filter') if overlay else None, None)
        mf['risk_off_breadth_floor'] = float(floor)
        updated['market_filter'] = mf
        _save_policy(updated, target_path)

    return float(floor)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description='Calibrate market breadth guard floor.')
    parser.add_argument(
        '--policy',
        type=Path,
        default=None,
        help='Override path to the policy JSON file (defaults to out/orders/policy_overrides.json, '
             'falling back to config/policy_overrides.json).',
    )
    parser.add_argument(
        '--dry-run',
        action='store_true',
        help='Compute the floor but do not write it back to the policy file.',
    )
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    value = calibrate(write=not args.dry_run, policy_path=args.policy)
    suffix = ' (dry-run)' if args.dry_run else ''
    print(f"[calibrate.breadth] risk_off_breadth_floor={value:.4f}{suffix}")


if __name__ == '__main__':
    main()
