from __future__ import annotations

"""
Calibrate sizing.softmax_tau to target effective number of positions (ENP).

Orthodox basis
- Softmax temperature controls concentration. Define weights w_i ∝ exp(score_i / τ)
  within each bucket (adds/news). Effective number ENP = 1 / Σ w_i^2 increases
  monotonically with τ. Solve τ by bisection to match target ENP.

Inputs
- out/{portfolio_clean, snapshot, metrics, presets_all, sector_strength, session_summary}.csv
- data/industry_map.csv (or metrics.Sector)
- config/policy_overrides.json for weights/thresholds and targets:
  calibration_targets.sizing.{enp_target_add, enp_target_new} (or enp_target)

Output
- sizing.softmax_tau written to config.

Fail-fast: missing files/columns/targets => SystemExit.
"""

from pathlib import Path
import json
import re
from typing import Dict, List, Tuple, Optional

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


def _load_json() -> Dict:
    src = ORDERS_PATH if ORDERS_PATH.exists() else CONFIG_PATH
    raw = src.read_text(encoding='utf-8')
    raw = re.sub(r"/\*.*?\*/", "", raw, flags=re.S)
    raw = re.sub(r"(^|\s)//.*$", "", raw, flags=re.M)
    raw = re.sub(r"(^|\s)#.*$", "", raw, flags=re.M)
    return json.loads(raw)


def _save_json(obj: Dict) -> None:
    target = ORDERS_PATH if ORDERS_PATH.exists() else CONFIG_PATH
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(obj, ensure_ascii=False, indent=2), encoding='utf-8')


def _load(name: str) -> pd.DataFrame:
    path = OUT_DIR / name
    if not path.exists():
        raise SystemExit(f'Missing out/{name}')
    return pd.read_csv(path)


def _industry_df(metrics: pd.DataFrame) -> pd.DataFrame:
    src = DATA_DIR / 'industry_map.csv'
    if src.exists():
        df = pd.read_csv(src)
        if not {'Ticker','Sector'}.issubset(df.columns):
            raise SystemExit('industry_map.csv must contain Ticker,Sector')
        return df[['Ticker','Sector']]
    return metrics[['Ticker','Sector']] if 'Sector' in metrics.columns else pd.DataFrame({'Ticker': [], 'Sector': []})


def _normalizers(metrics: pd.DataFrame) -> Dict[str, Dict[str, float]]:
    out: Dict[str, Dict[str, float]] = {}
    for c in ("RSI14","LiqNorm","MomRetNorm","ATR14_Pct","Beta60D","Fund_ROE","Fund_EarningsYield","RS_Trend50"):
        if c not in metrics.columns:
            continue
        s = pd.to_numeric(metrics[c], errors='coerce').replace([np.inf,-np.inf], np.nan).dropna()
        if s.empty:
            continue
        std = float(s.std(ddof=0))
        if std > 0 and np.isfinite(std):
            out[c] = {'mean': float(s.mean()), 'std': float(max(std, 1e-6))}
    return out


def _candidate_scores(tuning: Dict, portfolio: pd.DataFrame, snapshot: pd.DataFrame, metrics: pd.DataFrame, presets: pd.DataFrame, sector_strength: pd.DataFrame, session_summary: pd.DataFrame, industry: pd.DataFrame) -> Tuple[List[float], List[float]]:
    regime = get_market_regime(session_summary, sector_strength, tuning)
    snap = snapshot.set_index('Ticker')
    met = metrics.set_index('Ticker')
    pre = presets.set_index('Ticker') if not presets.empty else pd.DataFrame()
    norm = _normalizers(metrics)
    held = set(portfolio['Ticker'].astype(str).tolist()) if not portfolio.empty else set()
    uni = set(industry['Ticker'].astype(str).tolist()) if not industry.empty else set(metrics['Ticker'].astype(str).tolist())
    secmap = {str(r['Ticker']): r.get('Sector') for _, r in industry.iterrows()}

    def feat_score(t: str) -> Tuple[Optional[Dict[str,float]], float]:
        if t not in snap.index:
            return None, 0.0
        srow = snap.loc[t]
        mrow = met.loc[t] if t in met.index else None
        feats = compute_features(t, srow, mrow, norm)
        sc = conviction_score(feats, secmap.get(t, ''), regime, t)
        return feats, sc

    add_scores: List[float] = []
    new_scores: List[float] = []
    for _, r in portfolio.iterrows():
        t = str(r['Ticker']).strip()
        if t not in snap.index:
            continue
        feats, sc = feat_score(t)
        if feats is None:
            continue
        a = classify_action(True, sc, feats, regime, thresholds_override=None, ticker=t)
        if a == 'add':
            add_scores.append(sc)
    for t in uni:
        if t in held or t not in snap.index:
            continue
        feats, sc = feat_score(t)
        if feats is None:
            continue
        a = classify_action(False, sc, feats, regime, thresholds_override=None, ticker=t)
        if a == 'new':
            new_scores.append(sc)
    return add_scores, new_scores


def _enp_for_tau(scores: List[float], tau: float) -> float:
    if not scores:
        return 0.0
    arr = np.array(scores, dtype=float)
    m = float(np.max(arr))
    z = np.exp((arr - m) / max(tau, 1e-6))
    w = z / float(np.sum(z))
    return float(1.0 / float(np.sum(w * w)))


def _solve_tau(scores: List[float], target_enp: float, lo: float = 0.05, hi: float = 5.0, tol: float = 1e-3, max_iter: int = 50) -> float:
    if not scores or target_enp <= 0:
        return lo
    # Clamp achievable range
    n = len(scores)
    target = max(1.0, min(float(n), float(target_enp)))
    a, b = float(lo), float(hi)
    for _ in range(max_iter):
        mid = 0.5 * (a + b)
        enp_mid = _enp_for_tau(scores, mid)
        if abs(enp_mid - target) <= tol:
            return mid
        if enp_mid < target:
            a = mid
        else:
            b = mid
    return 0.5 * (a + b)


def calibrate(write: bool = False) -> Tuple[float, int, float, int, float]:
    pol = _load_json()
    targets = (pol.get('calibration_targets', {}) or {}).get('sizing', {})
    if not targets:
        raise SystemExit('Missing calibration_targets.sizing (need enp_target or enp_target_add/new)')
    try:
        enp_add = float(targets.get('enp_target_add', targets.get('enp_target')))
        enp_new = float(targets.get('enp_target_new', targets.get('enp_target')))
    except Exception as exc:
        raise SystemExit(f'Invalid sizing ENP targets: {exc}') from exc

    portfolio = _load('portfolio_clean.csv')
    snapshot = _load('snapshot.csv')
    metrics = _load('metrics.csv')
    presets = _load('presets_all.csv') if (OUT_DIR / 'presets_all.csv').exists() else pd.DataFrame()
    sector_strength = _load('sector_strength.csv')
    session_summary = _load('session_summary.csv')
    industry = _industry_df(metrics)

    add_scores, new_scores = _candidate_scores(pol, portfolio, snapshot, metrics, presets, sector_strength, session_summary, industry)
    n_add, n_new = len(add_scores), len(new_scores)
    if n_add <= 1 and n_new <= 1:
        raise SystemExit('Insufficient candidates to calibrate softmax_tau')

    tau_add = _solve_tau(add_scores, enp_add) if n_add > 1 else None
    tau_new = _solve_tau(new_scores, enp_new) if n_new > 1 else None

    # Combine into a single tau via budget-share weighting if available
    add_share = float((pol.get('sizing', {}) or {}).get('add_share', 0.5) or 0.5)
    new_share = float((pol.get('sizing', {}) or {}).get('new_share', 0.5) or 0.5)
    if tau_add is not None and tau_new is not None:
        tau = (add_share * tau_add + new_share * tau_new) / max(add_share + new_share, 1e-9)
    else:
        tau = float(tau_add if tau_add is not None else tau_new)

    if write:
        obj = pol
        sz = dict(obj.get('sizing', {}) or {})
        sz['softmax_tau'] = float(tau)
        obj['sizing'] = sz
        _save_json(obj)
    return float(tau), int(n_add), float(enp_add), int(n_new), float(enp_new)


def main():
    # Always write tuned softmax_tau to runtime overrides
    tau, n_add, enp_add, n_new, enp_new = calibrate(write=True)
    print(f"[calibrate.tau] softmax_tau={tau:.4f} (add_pool={n_add}, target_enp_add={enp_add:.2f}; new_pool={n_new}, target_enp_new={enp_new:.2f})")


if __name__ == '__main__':
    main()
