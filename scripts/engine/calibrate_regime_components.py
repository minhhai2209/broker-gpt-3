from __future__ import annotations

"""
Calibrate regime_model.components mean/std from recent history.

Orthodox approach
- Keep expert-set component weights; only normalize component distributions so
  z-scores are well-scaled. Uses VNINDEX-based daily composites consistent with
  the engine for: trend, index_return, volatility, drawdown, index_atr_percentile
  (if High/Low available). Optionally computes turnover percentile across the
  universe when Volume is present.

Inputs
- out/prices_history.csv (required), ideally with High/Low for ATR and Volume for turnover
- config/policy_overrides.json with regime_model.components

Output
- Writes updated mean/std for matching components in config.

Notes
- Components present in policy but not computable here are left unchanged.
- Fail-fast on missing core files; skip-only unavailable components.
"""

from pathlib import Path
import json
import re
from typing import Dict, List

import numpy as np
import pandas as pd


BASE_DIR = Path(__file__).resolve().parents[2]
OUT_DIR = BASE_DIR / 'out'
ORDERS_PATH = OUT_DIR / 'orders' / 'policy_overrides.json'
CONFIG_PATH = BASE_DIR / 'config' / 'policy_overrides.json'


def _strip(text: str) -> str:
    text = re.sub(r"/\*.*?\*/", "", text, flags=re.S)
    text = re.sub(r"(^|\s)//.*$", "", text, flags=re.M)
    text = re.sub(r"(^|\s)#.*$", "", text, flags=re.M)
    return text


def _load_policy() -> Dict:
    src = ORDERS_PATH if ORDERS_PATH.exists() else CONFIG_PATH
    if not src.exists():
        raise SystemExit(f"Missing policy file: {src}")
    return json.loads(_strip(src.read_text(encoding='utf-8')))


def _save_policy(obj: Dict) -> None:
    target = ORDERS_PATH if ORDERS_PATH.exists() else CONFIG_PATH
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(obj, ensure_ascii=False, indent=2), encoding='utf-8')


def _load_history() -> pd.DataFrame:
    path = OUT_DIR / 'prices_history.csv'
    if not path.exists():
        raise SystemExit('Missing out/prices_history.csv for regime components calibration')
    df = pd.read_csv(path)
    must = {'Date','Ticker','Close'}
    if not must.issubset(df.columns):
        missing = ', '.join(sorted(must - set(df.columns)))
        raise SystemExit(f'prices_history.csv missing columns: {missing}')
    df['Ticker'] = df['Ticker'].astype(str).str.upper()
    df['Date'] = pd.to_datetime(df['Date'], errors='coerce')
    return df.dropna(subset=['Date'])


def _ema(series: pd.Series, span: int) -> pd.Series:
    return pd.to_numeric(series, errors='coerce').ewm(span=span, adjust=False).mean()


def _atr_wilder(high: pd.Series, low: pd.Series, close: pd.Series, period: int = 14) -> pd.Series:
    h = pd.to_numeric(high, errors='coerce')
    l = pd.to_numeric(low, errors='coerce')
    c = pd.to_numeric(close, errors='coerce')
    prev = c.shift(1)
    tr = pd.concat([(h - l), (h - prev).abs(), (l - prev).abs()], axis=1).max(axis=1)
    return tr.ewm(alpha=1/period, adjust=False).mean()


def _pct_rank(s: pd.Series, win: int = 252) -> pd.Series:
    out = s.copy() * 0.0
    for i in range(len(s)):
        lo = max(0, i - win + 1)
        window = s.iloc[lo:i+1]
        x = s.iloc[i]
        if np.isfinite(x) and window.notna().any():
            out.iloc[i] = (window <= x).mean()
        else:
            out.iloc[i] = np.nan
    return out


def _compute_components(history: pd.DataFrame) -> pd.DataFrame:
    vn = history[history['Ticker']=='VNINDEX'].copy()
    if vn.empty:
        raise SystemExit('prices_history has no VNINDEX; cannot calibrate regime components')
    vn = vn.sort_values('Date')
    close = pd.to_numeric(vn['Close'], errors='coerce')
    ret = close.pct_change()
    ma50 = close.rolling(50).mean()
    ma200 = close.rolling(200).mean()
    trend_strength = (close - ma200) / ma200
    mom63 = close.pct_change(63)
    mom_ratio = np.maximum(0.0, mom63) / 0.12
    mom_ratio = np.clip(mom_ratio, 0.0, 1.0)
    mom_pct = _pct_rank(mom63)
    trend_up = np.clip(np.maximum(trend_strength, 0.0) / 0.10, 0.0, 1.0)
    momentum_norm = 0.5 * mom_pct + 0.5 * mom_ratio
    trend_momentum = 0.5 * trend_up + 0.5 * momentum_norm
    roll_max = close.cummax()
    dd = 1.0 - close / roll_max.replace(0, np.nan)
    dd = dd.replace([np.inf, -np.inf], np.nan).clip(lower=0.0)
    dd_pct = _pct_rank(dd)
    dd_comp = np.minimum(1.0 - (dd / 0.20), 1.0)
    dd_comp = np.minimum(dd_comp, 1.0 - dd_pct)
    vol_ann = ret.ewm(span=20, adjust=False).std() * np.sqrt(252.0)
    vol_pct = _pct_rank(vol_ann)
    vol_comp = np.minimum(np.clip(1.0 - (vol_ann / 0.45), 0.0, 1.0), 1.0 - vol_pct)
    # ATR percentile if Hi/Lo available
    atr_pctile = None
    if {'High','Low'}.issubset(set(vn.columns)):
        atr = _atr_wilder(vn['High'], vn['Low'], close, 14)
        with np.errstate(divide='ignore', invalid='ignore'):
            atr_pct = (atr / close).replace([np.inf, -np.inf], np.nan)
        atr_pctile = _pct_rank(atr_pct.fillna(atr_pct.median()) if atr_pct.notna().any() else pd.Series([0.5]*len(vn)))
    idx_smoothed = (ret * 100.0).rolling(6).mean()
    idx_norm = np.clip(idx_smoothed / 1.5, -1.0, 1.0)
    idx_comp = np.clip(np.maximum(idx_norm, 0.0), 0.0, 1.0)
    out = pd.DataFrame({
        'Date': vn['Date'].values,
        'trend': trend_momentum.values,
        'index_return': idx_comp.values,
        'volatility': vol_comp.values,
        'drawdown': dd_comp.fillna(0.5).values,
    })
    if atr_pctile is not None:
        out['index_atr_percentile'] = atr_pctile.fillna(0.5).values

    # Turnover percentile across non-index if Volume present
    turnover_series = None
    if 'Volume' in history.columns and 'Close' in history.columns:
        h = history.copy()
        non_index = ~h['Ticker'].astype(str).str.upper().isin(['VNINDEX','VN30','VN100'])
        h = h[non_index]
        if not h.empty:
            h['Date'] = pd.to_datetime(h['Date'], errors='coerce')
            h = h.dropna(subset=['Date'])
            h['Close'] = pd.to_numeric(h['Close'], errors='coerce')
            h['Volume'] = pd.to_numeric(h['Volume'], errors='coerce')
            h = h.dropna(subset=['Close','Volume'])
            if not h.empty:
                tv = h.assign(Total=(h['Close']*h['Volume'])).groupby('Date')['Total'].sum().sort_index()
                # Percentile rank over trailing window
                turnover_series = tv.rolling(252, min_periods=20).apply(lambda x: (x<=x.iloc[-1]).mean(), raw=False)
    if turnover_series is not None and not turnover_series.dropna().empty:
        out = out.merge(turnover_series.rename('turnover').reset_index(), on='Date', how='left')

    return out.dropna(subset=['Date']).reset_index(drop=True)


def calibrate(write: bool = False) -> List[str]:
    pol = _load_policy()
    rm = pol.get('regime_model', {}) or {}
    comps = rm.get('components', {}) or {}
    if not comps:
        raise SystemExit('policy_overrides has no regime_model.components to calibrate')
    hist = _load_history()
    comp_df = _compute_components(hist)
    updated: List[str] = []
    # Compute mean/std for overlapping components
    for name in comps.keys():
        if name in comp_df.columns:
            series = pd.to_numeric(comp_df[name], errors='coerce').replace([np.inf,-np.inf], np.nan).dropna()
            if series.empty:
                continue
            mean = float(series.mean())
            std = float(series.std(ddof=0))
            std = max(std, 1e-6)
            comps[name]['mean'] = mean
            comps[name]['std'] = std
            updated.append(name)
    if write and updated:
        obj = pol
        obj.setdefault('regime_model', {}).setdefault('components', comps)
        _save_policy(obj)
    return updated


def main():
    from argparse import ArgumentParser
    ap = ArgumentParser()
    ap.add_argument('--write', action='store_true')
    args = ap.parse_args()
    names = calibrate(write=args.write)
    print('[calibrate.regime_components] updated:', ', '.join(names) if names else '(none)')


if __name__ == '__main__':
    main()
