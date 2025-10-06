from __future__ import annotations

"""
Calibrate thresholds.q_add and thresholds.q_new to target top‑K adds/news.

Orthodox approach
- Use current cross‑section to estimate pool sizes before quantile gates, then
  set quantiles following the canonical q = 1 - K / N formula. Guardrails only
  apply at the extremes: q → 0.0 when K ≥ N (keep full pool) and q → 0.995 when
  K ≤ 0 or pools are empty. This keeps the calibration faithful to theory while
  still preventing pathological tails.

Inputs (required)
- out/portfolio_clean.csv (holdings)
- out/snapshot.csv, out/metrics.csv (features), out/presets_all.csv (optional)
- out/sector_strength.csv, out/session_summary.csv (for regime diagnostics)
- data/industry_map.csv or metrics.Sector for sector mapping
- config/policy_overrides.json for weights/thresholds/etc

Output
- Writes thresholds.q_add and thresholds.q_new into config/policy_overrides.json

Fail‑fast on missing files/columns or insufficient data.
"""

from pathlib import Path
import json
import re
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd

from scripts.orders.order_engine import (
    get_market_regime,
    compute_features,
    conviction_score,
    classify_action,
)


BASE_DIR = Path(__file__).resolve().parents[3]
OUT_DIR = BASE_DIR / 'out'
DATA_DIR = BASE_DIR / 'data'
ORDERS_PATH = OUT_DIR / 'orders' / 'policy_overrides.json'
CONFIG_PATH = BASE_DIR / 'config' / 'policy_overrides.json'


def _read_json_config() -> Dict:
    src = ORDERS_PATH if ORDERS_PATH.exists() else CONFIG_PATH
    if not src.exists():
        raise SystemExit(f'Missing policy file: {src}')
    raw = src.read_text(encoding='utf-8')
    raw = re.sub(r"/\*.*?\*/", "", raw, flags=re.S)
    raw = re.sub(r"(^|\s)//.*$", "", raw, flags=re.M)
    raw = re.sub(r"(^|\s)#.*$", "", raw, flags=re.M)
    return json.loads(raw)


def _write_json_config(obj: Dict) -> None:
    target = ORDERS_PATH if ORDERS_PATH.exists() else CONFIG_PATH
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(obj, ensure_ascii=False, indent=2), encoding='utf-8')


def _load_csv(name: str) -> pd.DataFrame:
    path = OUT_DIR / name
    if not path.exists():
        raise SystemExit(f'Missing out/{name}')
    df = pd.read_csv(path)
    if df is None:
        raise SystemExit(f'Unable to read out/{name}')
    return df


def _build_industry_df(metrics: pd.DataFrame) -> pd.DataFrame:
    # Prefer data/industry_map.csv if present; else use Sector in metrics
    src = DATA_DIR / 'industry_map.csv'
    if src.exists():
        df = pd.read_csv(src)
        if not {'Ticker', 'Sector'}.issubset(df.columns):
            raise SystemExit('industry_map.csv must have Ticker,Sector')
        return df[['Ticker', 'Sector']]
    if 'Sector' in metrics.columns:
        return metrics[['Ticker', 'Sector']]
    # Fallback to empty with required columns
    return pd.DataFrame({'Ticker': [], 'Sector': []})


def _build_normalizers(metrics: pd.DataFrame) -> Dict[str, Dict[str, float]]:
    normalizers: Dict[str, Dict[str, float]] = {}
    def _collect(col: str) -> None:
        if col not in metrics.columns:
            return
        s = pd.to_numeric(metrics[col], errors='coerce').replace([np.inf, -np.inf], np.nan).dropna()
        if s.empty:
            return
        std = float(s.std(ddof=0))
        if std > 0 and np.isfinite(std):
            normalizers[col] = {'mean': float(s.mean()), 'std': float(max(std, 1e-6))}
    for c in ("RSI14", "LiqNorm", "MomRetNorm", "ATR14_Pct", "Beta60D", "Fund_ROE", "Fund_EarningsYield", "RS_Trend50"):
        _collect(c)
    return normalizers


def _ensure_liqnorm(metrics: pd.DataFrame) -> pd.DataFrame:
    if 'LiqNorm' in metrics.columns:
        return metrics
    if 'AvgTurnover20D_k' in metrics.columns:
        tmp = metrics[['Ticker', 'AvgTurnover20D_k']].dropna().sort_values('AvgTurnover20D_k')
        if not tmp.empty:
            n = float(len(tmp))
            rank_map = {t: i / max(n - 1.0, 1.0) for i, t in enumerate(tmp['Ticker'].tolist())}
            metrics = metrics.copy()
            metrics['LiqNorm'] = metrics['Ticker'].map(rank_map).fillna(0.0)
    return metrics


def _target_q(K: int, N: int) -> float:
    if N <= 0:
        return 0.995
    keep = int(K)
    if keep <= 0:
        return 0.995
    if keep >= N:
        return 0.0
    base = 1.0 - (float(keep) / float(N))
    return float(min(0.995, max(0.0, base)))


def calibrate(write: bool = False) -> Tuple[float, float, int, int]:
    pol = _read_json_config()
    portfolio = _load_csv('portfolio_clean.csv')
    snapshot = _load_csv('snapshot.csv')
    metrics = _load_csv('metrics.csv')
    presets = _load_csv('presets_all.csv') if (OUT_DIR / 'presets_all.csv').exists() else pd.DataFrame()
    sector_strength = _load_csv('sector_strength.csv')
    session_summary = _load_csv('session_summary.csv')

    metrics = _ensure_liqnorm(metrics)
    industry = _build_industry_df(metrics)

    # Build tuning dict (same schema as engine expects)
    tuning = pol
    # Regime needed for scores (sector boosts etc.)
    regime = get_market_regime(session_summary, sector_strength, tuning)

    snap = snapshot.set_index('Ticker')
    met = metrics.set_index('Ticker')
    pre = presets.set_index('Ticker') if not presets.empty else pd.DataFrame()
    normalizers = _build_normalizers(metrics)

    held = set(str(t).strip().upper() for t in portfolio['Ticker'].astype(str).tolist()) if not portfolio.empty else set()
    universe = set(industry['Ticker'].astype(str).tolist()) if not industry.empty else set(metrics['Ticker'].astype(str).tolist())
    sector_map = {str(row['Ticker']): row.get('Sector') for _, row in industry.iterrows()}

    def _feat_score(t: str) -> Tuple[Dict[str, float], float]:
        if t not in snap.index:
            # if not in snapshot, skip
            return {}, 0.0
        srow = snap.loc[t]
        mrow = met.loc[t] if t in met.index else None
        feats = compute_features(t, srow, mrow, normalizers)
        sc = conviction_score(feats, sector_map.get(t, ''), regime, t)
        return feats, sc

    # Estimate pre-quantile pools
    add_pool_scores: List[float] = []
    new_pool_scores: List[float] = []
    th = dict(regime.thresholds)
    for _, row in portfolio.iterrows():
        t = str(row['Ticker']).strip().upper()
        if t not in snap.index:
            continue
        feats, sc = _feat_score(t)
        if not feats:
            continue
        a = classify_action(True, sc, feats, regime, thresholds_override=None, ticker=t)
        if a == 'add':
            add_pool_scores.append(sc)

    for t in universe:
        if t in held or t not in snap.index:
            continue
        feats, sc = _feat_score(t)
        if not feats:
            continue
        a = classify_action(False, sc, feats, regime, thresholds_override=None, ticker=t)
        if a == 'new':
            new_pool_scores.append(sc)

    n_add = len(add_pool_scores)
    n_new = len(new_pool_scores)
    try:
        add_max = int(pol['add_max'])
        new_max = int(pol['new_max'])
    except Exception as exc:
        raise SystemExit(f'Invalid add_max/new_max: {exc}') from exc

    q_add = _target_q(add_max, n_add)
    q_new = _target_q(new_max, n_new)

    if write:
        obj = pol
        th2 = dict(obj.get('thresholds', {}) or {})
        th2['q_add'] = float(q_add)
        th2['q_new'] = float(q_new)
        obj['thresholds'] = th2
        _write_json_config(obj)
    return float(q_add), float(q_new), int(n_add), int(n_new)


def main():
    # Always write tuned quantile gates to runtime overrides
    q_add, q_new, n_add, n_new = calibrate(write=True)
    print(f"[calibrate.topk] q_add={q_add:.3f} (pool={n_add}), q_new={q_new:.3f} (pool={n_new})")


if __name__ == '__main__':
    main()
