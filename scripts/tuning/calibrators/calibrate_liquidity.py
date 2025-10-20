from __future__ import annotations

"""
Calibrate thresholds.min_liq_norm from ADTV vs planned per-position budget.

Orthodox logic
- Require average daily turnover (ADTV, 20D) to exceed a multiple of the
  expected order size per NEW position: ADTV >= M * ticket_size.
- Convert this condition into the engine's LiqNorm percentile threshold by
  mapping the required ADTV to its empirical CDF over the universe.

Inputs
- out/metrics.csv with column AvgTurnover20D_k
- out/portfolio_pnl_summary.csv for NAV estimation
- config/policy_overrides.json for buy_budget_frac, new_max, sizing.new_share
- calibration_targets.liquidity.adtv_multiple (e.g., 20)

Output
- Writes thresholds.min_liq_norm into config/policy_overrides.json

Fail-fast
- Missing required files/columns/targets or NAV<=0 => SystemExit
"""

from pathlib import Path
import json
import re

import numpy as np
import pandas as pd

from scripts.tuning.calibrators.policy_write import write_policy


BASE_DIR = Path(__file__).resolve().parents[3]
OUT_DIR = BASE_DIR / 'out'
ORDERS_PATH = OUT_DIR / 'orders' / 'policy_overrides.json'
CONFIG_PATH = BASE_DIR / 'config' / 'policy_overrides.json'
DEFAULTS_PATH = BASE_DIR / 'config' / 'policy_default.json'


def _strip_json_comments(text: str) -> str:
    text = re.sub(r"/\*.*?\*/", "", text, flags=re.S)
    text = re.sub(r"(^|\s)//.*$", "", text, flags=re.M)
    text = re.sub(r"(^|\s)#.*$", "", text, flags=re.M)
    return text


def _load_policy() -> dict:
    src = ORDERS_PATH if ORDERS_PATH.exists() else CONFIG_PATH
    if not src.exists():
        raise SystemExit(f"Missing policy file: {src}")
    raw = src.read_text(encoding='utf-8')
    return json.loads(_strip_json_comments(raw))


def _load_nav() -> float:
    path = OUT_DIR / 'portfolio_pnl_summary.csv'
    if not path.exists():
        raise SystemExit('Missing out/portfolio_pnl_summary.csv for NAV estimation')
    df = pd.read_csv(path)
    if df.empty:
        raise SystemExit('portfolio_pnl_summary.csv is empty')
    def _f(col: str) -> float:
        if col in df.columns:
            val = pd.to_numeric(df.iloc[0][col], errors='coerce')
            if pd.notna(val) and float(val) > 0:
                return float(val)
        return 0.0
    nav = max(_f('TotalMarket'), _f('TotalCost'))
    if nav <= 0:
        raise SystemExit('NAV not available (>0) in portfolio_pnl_summary.csv')
    return float(nav)


def _load_metrics() -> pd.DataFrame:
    path = OUT_DIR / 'metrics.csv'
    if not path.exists():
        raise SystemExit('Missing out/metrics.csv required for liquidity calibration')
    df = pd.read_csv(path)
    if 'AvgTurnover20D_k' not in df.columns:
        raise SystemExit('metrics.csv missing AvgTurnover20D_k')
    ser = pd.to_numeric(df['AvgTurnover20D_k'], errors='coerce').replace([np.inf, -np.inf], np.nan).dropna()
    if ser.empty:
        raise SystemExit('AvgTurnover20D_k series is empty after cleaning')
    return ser.to_frame(name='AvgTurnover20D_k')


def _load_baseline_min_liq() -> float:
    if not DEFAULTS_PATH.exists():
        return 0.3
    try:
        raw = DEFAULTS_PATH.read_text(encoding='utf-8')
        base = json.loads(_strip_json_comments(raw))
        thresholds = (base.get('thresholds', {}) or {})
        val = thresholds.get('min_liq_norm')
        if val is not None:
            return float(val)
    except Exception:
        pass
    return 0.3


def _coerce_float(val):
    try:
        return float(val)
    except Exception:
        return None


def calibrate(write: bool = False) -> float:
    pol = _load_policy()
    targets = (pol.get('calibration_targets', {}) or {}).get('liquidity', {})
    if 'adtv_multiple' not in targets:
        raise SystemExit('Missing calibration_targets.liquidity.adtv_multiple')
    try:
        multiple = float(targets['adtv_multiple'])
    except Exception as exc:
        raise SystemExit(f'invalid adtv_multiple: {exc}') from exc
    if multiple <= 0:
        raise SystemExit('adtv_multiple must be > 0')

    thresholds = (pol.get('thresholds', {}) or {})
    fallback_min_liq = _coerce_float(thresholds.get('min_liq_norm'))
    if fallback_min_liq is None:
        fallback_min_liq = _load_baseline_min_liq()

    try:
        bb = float(pol['buy_budget_frac'])
        new_max = int(pol['new_max'])
        sizing = pol.get('sizing', {}) or {}
        new_share = float(sizing['new_share'])
    except Exception as exc:
        raise SystemExit(f'Missing or invalid policy fields for liquidity calibration: {exc}') from exc
    if new_max <= 0:
        if write:
            obj = dict(pol)
            th = dict(thresholds)
            existing = _coerce_float(th.get('min_liq_norm'))
            floor = float(existing if existing is not None else fallback_min_liq)
            th['min_liq_norm'] = floor
            obj['thresholds'] = th
            write_policy(
                calibrator=__name__,
                policy=obj,
                orders_path=ORDERS_PATH,
                config_path=CONFIG_PATH,
            )
        return float(fallback_min_liq)
    if not (0.0 < bb <= 0.30):
        raise SystemExit('buy_budget_frac must be in (0, 0.30]')
    if not (0.0 < new_share <= 1.0):
        raise SystemExit('sizing.new_share must be in (0,1]')

    nav = _load_nav()
    per_new_budget_k = (bb * new_share * nav) / float(new_max)
    required_adtv_k = multiple * per_new_budget_k

    metrics = _load_metrics()
    s = metrics['AvgTurnover20D_k'].sort_values().to_numpy()
    # Empirical CDF at required_adtv_k
    cdf = float((s <= required_adtv_k).sum()) / float(len(s))
    min_liq_norm = max(0.0, min(1.0, cdf))

    if write:
        obj = pol
        th = dict(obj.get('thresholds', {}) or {})
        th['min_liq_norm'] = float(min_liq_norm)
        obj['thresholds'] = th
        write_policy(
            calibrator=__name__,
            policy=obj,
            orders_path=ORDERS_PATH,
            config_path=CONFIG_PATH,
        )
    return float(min_liq_norm)


def main():
    # Always write tuned liquidity threshold to runtime overrides
    val = calibrate(write=True)
    print(f"[calibrate.liq] min_liq_norm = {val:.4f}")


if __name__ == '__main__':
    main()
