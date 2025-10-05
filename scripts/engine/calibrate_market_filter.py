from __future__ import annotations

"""
Calibrate market_filter thresholds from VNINDEX history using robust percentiles.

Orthodox approach:
- Use distributional percentiles (quantiles) over a long lookback to set
  hard/soft guards. This avoids ad‑hoc thresholds and follows risk practice
  of gating by volatility and tail moves.

Inputs (required):
- out/prices_history.csv with VNINDEX Close (and High/Low for ATR) columns.
- config/policy_overrides.json containing `calibration_targets.market_filter`:
  {
    "idx_drop_q": 0.10,
    "vol_ann_q": 0.95,
    "trend_floor_q": 0.10,
    "atr_soft_q": 0.80,
    "atr_hard_q": 0.95,
    # Optional: set market_score floors by probability percentiles
    "ms_soft_q": 0.60,
    "ms_hard_q": 0.35,
    # Optional: drawdown floor quantile for disabling leader bypass during severe drawdown
    "dd_floor_q": 0.80
  }

Outputs:
- Writes calibrated values into config/policy_overrides.json under `market_filter`:
  - risk_off_index_drop_pct (percent magnitude)
  - vol_ann_hard_ceiling (annualized volatility)
  - trend_norm_hard_floor (normalized trend floor)
  - index_atr_soft_pct, index_atr_hard_pct (percentiles of ATR14%)
  - market_score_soft_floor, market_score_hard_floor (optional if targets provided)

Fail-fast:
- Missing VNINDEX history or required columns => SystemExit with clear message.
- Missing calibration targets => SystemExit.
"""

from pathlib import Path
from typing import Dict, Tuple, List
import json
import re

import numpy as np
import pandas as pd


BASE_DIR = Path(__file__).resolve().parents[2]
OUT_DIR = BASE_DIR / 'out'
ORDERS_PATH = OUT_DIR / 'orders' / 'policy_overrides.json'
CONFIG_PATH = BASE_DIR / 'config' / 'policy_overrides.json'
DEFAULTS_PATH = BASE_DIR / 'config' / 'policy_default.json'


def _strip_json_comments(text: str) -> str:
    text = re.sub(r"/\*.*?\*/", "", text, flags=re.S)
    text = re.sub(r"(^|\s)//.*$", "", text, flags=re.M)
    text = re.sub(r"(^|\s)#.*$", "", text, flags=re.M)
    return text


def _load_policy() -> Dict:
    src = ORDERS_PATH if ORDERS_PATH.exists() else CONFIG_PATH
    if not src.exists():
        raise SystemExit(f"Missing policy file: {src}")
    raw = src.read_text(encoding='utf-8')
    js = json.loads(_strip_json_comments(raw))
    return js


def _save_policy(obj: Dict) -> None:
    target = ORDERS_PATH if ORDERS_PATH.exists() else CONFIG_PATH
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(obj, ensure_ascii=False, indent=2), encoding='utf-8')


def _load_vnindex_history(min_days: int = 400) -> pd.DataFrame:
    path = OUT_DIR / 'prices_history.csv'
    if not path.exists():
        raise SystemExit("Missing out/prices_history.csv — build pipeline first.")
    df = pd.read_csv(path)
    must = {'Date', 'Ticker', 'Close'}
    if not must.issubset(df.columns):
        missing = ', '.join(sorted(must - set(df.columns)))
        raise SystemExit(f"prices_history.csv missing columns: {missing}")
    df['Ticker'] = df['Ticker'].astype(str).str.upper()
    vn = df[df['Ticker'] == 'VNINDEX'].copy()
    if vn.empty:
        raise SystemExit('prices_history has no VNINDEX; cannot calibrate market_filter')
    vn['Date'] = pd.to_datetime(vn['Date'], errors='coerce')
    vn = vn.dropna(subset=['Date']).sort_values('Date')
    if len(vn) < min_days:
        # Attempt a targeted refill of VNINDEX cache, then retry once
        try:
            from scripts.fetch_ticker_data import ensure_and_load_history_df as _ensure_hist
            refill = _ensure_hist(['VNINDEX'], outdir=str(OUT_DIR / 'data'), min_days=max(min_days, 700), resolution='D')
            if not refill.empty:
                refill['Ticker'] = refill['Ticker'].astype(str).str.upper()
                vn2 = refill[refill['Ticker'] == 'VNINDEX'].copy()
                vn2['Date'] = pd.to_datetime(vn2['Date'], errors='coerce')
                vn2 = vn2.dropna(subset=['Date']).sort_values('Date')
                if len(vn2) >= min_days:
                    return vn2
        except Exception:
            # Fall through to fail‑fast below
            pass
        # Allow shorter window but surface a clear failure for the caller
        raise SystemExit(f'VNINDEX history too short ({len(vn)}< {min_days}) for stable calibration')
    return vn


def _ewm_vol_ann(ret: pd.Series, span: int = 20) -> pd.Series:
    vol = pd.to_numeric(ret, errors='coerce').ewm(span=span, adjust=False).std()
    return vol * np.sqrt(252.0)


def _wilder_atr(high: pd.Series, low: pd.Series, close: pd.Series, period: int = 14) -> pd.Series:
    h = pd.to_numeric(high, errors='coerce')
    l = pd.to_numeric(low, errors='coerce')
    c = pd.to_numeric(close, errors='coerce')
    prev = c.shift(1)
    tr = pd.concat([(h - l), (h - prev).abs(), (l - prev).abs()], axis=1).max(axis=1)
    return tr.ewm(alpha=1/period, adjust=False).mean()


def _quantile(series: pd.Series, q: float) -> float:
    s = pd.to_numeric(series, errors='coerce').replace([np.inf, -np.inf], np.nan).dropna()
    if s.empty:
        raise SystemExit('Calibration series is empty after cleaning')
    if not (0.0 <= float(q) <= 1.0):
        raise SystemExit('Quantile must be in [0,1]')
    return float(np.quantile(s.to_numpy(), q))


def _compute_series(vn: pd.DataFrame) -> Tuple[pd.Series, pd.Series, pd.Series, pd.Series]:
    close = pd.to_numeric(vn['Close'], errors='coerce')
    ret = close.pct_change().fillna(0.0)
    # Smoothed daily % change over 6 days (percent units)
    daily_pct = ret * 100.0
    smoothed_pct = daily_pct.rolling(6).mean().dropna()

    # Annualized EWM vol (span=20)
    vol_ann = _ewm_vol_ann(ret)

    # Trend normalized vs MA200
    ma200 = close.rolling(200).mean()
    trend_strength = (close - ma200) / ma200
    trend_norm = (trend_strength / 0.05).clip(-1.0, 1.0)
    trend_norm = trend_norm.replace([np.inf, -np.inf], np.nan).dropna()

    # ATR14% if High/Low present
    if {'High','Low'}.issubset(set(vn.columns)):
        high = pd.to_numeric(vn['High'], errors='coerce')
        low = pd.to_numeric(vn['Low'], errors='coerce')
        atr = _wilder_atr(high, low, close, 14)
        atr_pct = (atr / close).replace([np.inf, -np.inf], np.nan)
    else:
        raise SystemExit('prices_history.csv missing High/Low required for ATR calibration')
    return smoothed_pct.dropna(), vol_ann.dropna(), trend_norm.dropna(), atr_pct.dropna()


def calibrate(write: bool = False) -> Dict[str, float]:
    pol = _load_policy()
    targets = (pol.get('calibration_targets', {}) or {}).get('market_filter', {})
    required = ['idx_drop_q', 'vol_ann_q', 'trend_floor_q', 'atr_soft_q', 'atr_hard_q']
    missing = [k for k in required if k not in targets]
    if missing:
        raise SystemExit('Missing calibration_targets.market_filter: ' + ', '.join(missing))

    for k in required:
        try:
            targets[k] = float(targets[k])
        except Exception as exc:
            raise SystemExit(f'Invalid calibration_targets.market_filter["{k}"]: {exc}') from exc

    vn = _load_vnindex_history(min_days=400)
    smoothed_pct, vol_ann, trend_norm, atr_pct = _compute_series(vn)

    # Compute thresholds
    q_idx = float(targets['idx_drop_q'])
    q_vol = float(targets['vol_ann_q'])
    q_trend = float(targets['trend_floor_q'])
    q_soft = float(targets['atr_soft_q'])
    q_hard = float(targets['atr_hard_q'])

    idx_drop = abs(_quantile(smoothed_pct, q_idx))  # percent magnitude
    vol_ceiling = _quantile(vol_ann, q_vol)
    trend_floor = _quantile(trend_norm, q_trend)
    # Engine compares current ATR percentile (0..1) with these thresholds,
    # so we must calibrate on the distribution of ATR percentiles, not raw ATR%.
    atr_rank = pd.to_numeric(atr_pct, errors='coerce').rank(pct=True)
    atr_rank = atr_rank.replace([np.inf, -np.inf], np.nan).dropna()
    atr_soft = float(np.quantile(atr_rank.to_numpy(), q_soft))
    atr_hard = float(np.quantile(atr_rank.to_numpy(), q_hard))

    out = {
        'risk_off_index_drop_pct': float(idx_drop),
        # Use the same negative‑tail magnitude for hard drop guard on smoothed index change
        'idx_chg_smoothed_hard_drop': float(idx_drop),
        'vol_ann_hard_ceiling': float(vol_ceiling),
        'trend_norm_hard_floor': float(trend_floor),
        'index_atr_soft_pct': float(atr_soft),
        'index_atr_hard_pct': float(atr_hard),
    }

    # Optional: calibrate market_score floors by regime probability percentiles
    try:
        ms_soft_q = targets.get('ms_soft_q', None)
        ms_hard_q = targets.get('ms_hard_q', None)
        dd_floor_q = targets.get('dd_floor_q', None)
        if ms_soft_q is not None and ms_hard_q is not None:
            ms_soft_q = float(ms_soft_q); ms_hard_q = float(ms_hard_q)
            # Build components consistent with regime_model and compute probability series
            rm = (pol.get('regime_model', {}) or {})
            comps_conf = dict(rm.get('components', {}) or {})
            intercept = float(rm.get('intercept', 0.0))
            # Prepare component frame similar to calibrate_regime
            # Reuse series already computed where possible
            # Build DF with required columns if available
            comp_df = pd.DataFrame({'Date': vn['Date'].values})
            # 'trend' composite requires more context; approximate with normalized trend_norm mapped to [0..1]
            # but to stay consistent, derive using the same approach as calibrator: use positive trend vs MA200 and momentum proxy
            # Minimal proxy: map trend_norm [-1..1] -> [0..1]
            comp_df['trend'] = ((trend_norm - trend_norm.min()) / max((trend_norm.max() - trend_norm.min()), 1e-9)).reindex(comp_df.index).fillna(0.5).values
            comp_df['index_return'] = np.clip(np.maximum(smoothed_pct / 1.5, 0.0), 0.0, 1.0).reindex(comp_df.index, fill_value=0.0).values
            # Volatility component ~ 1 - vol percentile and cushion
            vol_pct = (vol_ann.rank(pct=True)).reindex(comp_df.index, fill_value=0.5)
            vol_cushion = np.clip(1.0 - (vol_ann / max(vol_ann.quantile(0.99), 1e-9)), 0.0, 1.0)
            comp_df['volatility'] = np.minimum(vol_cushion, 1.0 - vol_pct).astype(float).values
            # Drawdown component ~ 1 - dd percentile
            close = pd.to_numeric(vn['Close'], errors='coerce')
            roll_max = close.cummax()
            dd = 1.0 - close / roll_max.replace(0, np.nan)
            dd = dd.replace([np.inf, -np.inf], np.nan).clip(lower=0.0)
            dd_pct = (dd.rank(pct=True)).reindex(comp_df.index, fill_value=0.5)
            dd_comp = np.minimum(1.0 - (dd / max(dd.quantile(0.99), 1e-9)), 1.0)
            comp_df['drawdown'] = np.minimum(dd_comp, 1.0 - dd_pct).astype(float).values
            # ATR percentile if available
            if {'High','Low'}.issubset(set(vn.columns)):
                high = pd.to_numeric(vn['High'], errors='coerce')
                low = pd.to_numeric(vn['Low'], errors='coerce')
                atr = _wilder_atr(high, low, close, 14)
                atr_pct = (atr / close).replace([np.inf,-np.inf], np.nan)
                atr_pctile = (atr_pct.rank(pct=True)).reindex(comp_df.index, fill_value=0.5)
                comp_df['index_atr_percentile'] = atr_pctile.astype(float).values

            # Build Z and weights from comps_conf
            names: List[str] = []
            Z_cols: List[np.ndarray] = []
            w: List[float] = []
            for name, conf in comps_conf.items():
                mean = float(conf.get('mean', 0.0))
                std = float(conf.get('std', 1.0)) or 1.0
                weight = float(conf.get('weight', 0.0))
                if name in comp_df.columns:
                    z = (pd.to_numeric(comp_df[name], errors='coerce') - mean) / std
                    Z_cols.append(z.to_numpy(dtype=float))
                else:
                    Z_cols.append(np.zeros(len(comp_df), dtype=float))
                names.append(name)
                w.append(weight)
            if Z_cols:
                Z = np.column_stack(Z_cols)
                w_arr = np.array(w, dtype=float)
                z_lin = Z @ w_arr + float(intercept)
                # Avoid overflow in exp by clipping linear term
                z_lin = np.clip(z_lin, -20.0, 20.0)
                prob = 1.0 / (1.0 + np.exp(-z_lin))
                # Drop NaNs/Infs before quantiles; require sufficient data
                prob_finite = prob[np.isfinite(prob)]
                if prob_finite.size == 0:
                    raise RuntimeError('no finite probabilities for market_score floors')
                ms_soft = float(np.quantile(prob_finite, ms_soft_q))
                ms_hard = float(np.quantile(prob_finite, ms_hard_q))
                # Ensure soft >= hard
                soft, hard = (ms_soft, ms_hard) if ms_soft >= ms_hard else (ms_hard, ms_soft)
                out['market_score_soft_floor'] = soft
                out['market_score_hard_floor'] = hard
        # Optional: drawdown floor from quantile
        if dd_floor_q is not None:
            dd_floor_q = float(dd_floor_q)
            close = pd.to_numeric(vn['Close'], errors='coerce')
            roll_max = close.cummax()
            dd = (1.0 - close / roll_max.replace(0, np.nan)).replace([np.inf, -np.inf], np.nan).dropna()
            if not dd.empty:
                out['risk_off_drawdown_floor'] = float(np.quantile(dd.to_numpy(), dd_floor_q))
    except Exception as _exc_ms:
        # Do not write ms_* floors when insufficient/unstable; print explicit warning upstack
        # (the caller prints a consolidated message; baseline values remain in effect)
        out.pop('market_score_soft_floor', None)
        out.pop('market_score_hard_floor', None)

    if write:
        obj = pol
        mf = dict(obj.get('market_filter', {}) or {})
        mf.update(out)
        obj['market_filter'] = mf
        _save_policy(obj)
    return out


def main():
    # Always write tuned thresholds to runtime overrides; baseline updated via nightly merge
    vals = calibrate(write=True)
    for k, v in vals.items():
        print(f"[calibrate.mf] {k} = {v}")


if __name__ == '__main__':
    main()
