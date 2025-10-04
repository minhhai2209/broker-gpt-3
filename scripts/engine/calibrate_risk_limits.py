from __future__ import annotations

"""
Calibrate ATR-based TP/SL multiples to align with static pct levels for typical volatility.

Orthodox approach
- Let median ATR14% across universe be A_med. If static take-profit/stop-loss
  are tp_pct/sl_pct, choose multipliers m_tp, m_sl so that dynamic thresholds
  (m * ATR%) ≈ static at typical volatility: m_tp ≈ tp_pct / A_med, m_sl ≈ sl_pct / A_med.
  Clamp multipliers to a practical band [0.5, 4.0] when non-zero to avoid
  unrealistic extremes when ATR is very high/low.
- Respect optional floors (tp_floor_pct, sl_floor_pct) already in policy.

Inputs
- out/metrics.csv with ATR14_Pct
- config/policy_overrides.json thresholds {tp_pct, sl_pct, tp_floor_pct?, sl_floor_pct?}

Output
- thresholds.tp_atr_mult and thresholds.sl_atr_mult written to config.

Fail-fast if required columns/keys missing or ATR median <= 0.
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


def _load_json() -> dict:
    src = ORDERS_PATH if ORDERS_PATH.exists() else CONFIG_PATH
    raw = src.read_text(encoding='utf-8')
    raw = re.sub(r"/\*.*?\*/", "", raw, flags=re.S)
    raw = re.sub(r"(^|\s)//.*$", "", raw, flags=re.M)
    raw = re.sub(r"(^|\s)#.*$", "", raw, flags=re.M)
    return json.loads(raw)


def _save_json(obj: dict) -> None:
    target = ORDERS_PATH if ORDERS_PATH.exists() else CONFIG_PATH
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(obj, ensure_ascii=False, indent=2), encoding='utf-8')


def _load_metrics() -> pd.DataFrame:
    path = OUT_DIR / 'metrics.csv'
    if not path.exists():
        raise SystemExit('Missing out/metrics.csv for ATR calibration')
    df = pd.read_csv(path)
    if 'ATR14_Pct' not in df.columns:
        raise SystemExit('metrics.csv missing ATR14_Pct')
    s = pd.to_numeric(df['ATR14_Pct'], errors='coerce').replace([np.inf,-np.inf], np.nan).dropna()
    if s.empty:
        raise SystemExit('ATR14_Pct series is empty after cleaning')
    return s


def calibrate(write: bool = False) -> tuple[float, float, float]:
    pol = _load_json()
    th = dict(pol.get('thresholds', {}) or {})
    if 'tp_pct' not in th or 'sl_pct' not in th:
        raise SystemExit('thresholds.tp_pct and thresholds.sl_pct are required for ATR calibration')
    tp_pct = float(th['tp_pct'])
    sl_pct = float(th['sl_pct'])
    tp_floor = float(th.get('tp_floor_pct') or 0.0)
    sl_floor = float(th.get('sl_floor_pct') or 0.0)

    atr_series = _load_metrics()
    atr_med = float(np.median(atr_series)) / 100.0
    if atr_med <= 0.0:
        raise SystemExit('Median ATR14_Pct is non-positive')

    # Choose multiples so dynamic ≈ static at median ATR; floors are enforced downstream
    def _clip_mult(numer: float) -> float:
        if numer <= 0.0:
            return 0.0
        return float(max(0.5, min(4.0, numer / atr_med)))

    tp_mult = _clip_mult(tp_pct if tp_pct > 0 else tp_floor)
    sl_mult = _clip_mult(sl_pct if sl_pct > 0 else sl_floor)

    if write:
        obj = pol
        th2 = dict(obj.get('thresholds', {}) or {})
        th2['tp_atr_mult'] = float(tp_mult)
        th2['sl_atr_mult'] = float(sl_mult)
        obj['thresholds'] = th2
        # Align sizing.default_stop_atr_mult with calibrated SL multiple for consistency
        sz = dict(obj.get('sizing', {}) or {})
        sz['default_stop_atr_mult'] = float(sl_mult)
        obj['sizing'] = sz
        _save_json(obj)
    return float(tp_mult), float(sl_mult), float(atr_med)


def main():
    from argparse import ArgumentParser
    ap = ArgumentParser()
    ap.add_argument('--write', action='store_true', help='write to runtime policy_overrides.json (default behavior)')
    ap.add_argument('--write-defaults', action='store_true', help='write to baseline config/policy_default.json')
    args = ap.parse_args()
    # If user wants baseline update, compute without side effects then write to defaults explicitly
    if args.write_defaults:
        tp_m, sl_m, atr_med = calibrate(write=False)
        # Update defaults file directly (source of truth)
        raw = DEFAULTS_PATH.read_text(encoding='utf-8')
        import re, json as _json
        raw = re.sub(r"/\*.*?\*/", "", raw, flags=re.S)
        raw = re.sub(r"(^|\s)//.*$", "", raw, flags=re.M)
        raw = re.sub(r"(^|\s)#.*$", "", raw, flags=re.M)
        obj = _json.loads(raw)
        th = dict(obj.get('thresholds', {}) or {})
        th['tp_atr_mult'] = float(tp_m)
        th['sl_atr_mult'] = float(sl_m)
        obj['thresholds'] = th
        # Align default stop multiple
        sz = dict(obj.get('sizing', {}) or {})
        sz['default_stop_atr_mult'] = float(sl_m)
        obj['sizing'] = sz
        DEFAULTS_PATH.write_text(_json.dumps(obj, ensure_ascii=False, indent=2), encoding='utf-8')
        print(f"[calibrate.risk] (defaults) tp_atr_mult={tp_m:.3f}, sl_atr_mult={sl_m:.3f} (median ATR={atr_med*100:.2f}%) -> {DEFAULTS_PATH}")
        return
    # Otherwise use legacy behavior (runtime overrides)
    tp_m, sl_m, atr_med = calibrate(write=args.write)
    print(f"[calibrate.risk] tp_atr_mult={tp_m:.3f}, sl_atr_mult={sl_m:.3f} (median ATR={atr_med*100:.2f}%)")


if __name__ == '__main__':
    main()
