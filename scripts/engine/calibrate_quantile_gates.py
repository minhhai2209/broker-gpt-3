from __future__ import annotations

"""
Calibrate thresholds.q_add and thresholds.q_new from current score distributions.

Method (objective, lightweight)
- Build candidate pools using the same feature/score/action logic as the engine:
  holders classified as 'add' and non-holders classified as 'new' using base gates
  (classify_action uses thresholds.base_add/base_new; quantile gates are applied later).
- For each pool S, choose q = 1 - K / N exactly (when 0 < K < N) so that
  quantile(S, q) yields ~K items above the gate. Guardrails apply only at the
  extremes: q becomes 0.0 when K ≥ N (keep everything) and 0.995 when K ≤ 0 or
  pools are empty/noisy. This keeps calibration aligned with the orthodox
  1 - K/N formula while still preventing pathological tails.

Inputs
- out/{portfolio_clean,snapshot,metrics,sector_strength,session_summary}.csv
- data/industry_map.csv
- out/orders/policy_overrides.json (runtime policy) for base thresholds, add_max/new_max.

Outputs
- thresholds.q_add, thresholds.q_new (written to runtime policy) and (q_add, q_new, pool sizes).

Fail-fast
- Missing artifacts; empty pools (<=1) -> SystemExit with a clear message.
"""

from pathlib import Path
import json, re
from typing import Dict, Tuple, List

import pandas as pd

BASE_DIR = Path(__file__).resolve().parents[2]
OUT_DIR = BASE_DIR / 'out'
ORDERS_DIR = OUT_DIR / 'orders'


def _load_json(path: Path) -> Dict:
    raw = path.read_text(encoding='utf-8')
    raw = re.sub(r"/\*.*?\*/", "", raw, flags=re.S)
    raw = re.sub(r"(^|\s)//.*$", "", raw, flags=re.M)
    raw = re.sub(r"(^|\s)#.*$", "", raw, flags=re.M)
    return json.loads(raw)


def _load_df(name: str) -> pd.DataFrame:
    p = OUT_DIR / name
    if not p.exists():
        raise SystemExit(f'Missing out/{name} for quantile gates calibration')
    df = pd.read_csv(p)
    if df is None or df.empty:
        raise SystemExit(f'out/{name} is empty')
    return df


def _industry_df(metrics: pd.DataFrame) -> pd.DataFrame:
    src = BASE_DIR / 'data' / 'industry_map.csv'
    if src.exists():
        d = pd.read_csv(src)
        if not {'Ticker','Sector'}.issubset(d.columns):
            raise SystemExit('industry_map.csv must contain Ticker,Sector')
        return d[['Ticker','Sector']]
    return metrics[['Ticker','Sector']] if 'Sector' in metrics.columns else pd.DataFrame({'Ticker': [], 'Sector': []})


def _candidate_scores(policy: Dict, portfolio: pd.DataFrame, snapshot: pd.DataFrame, metrics: pd.DataFrame, presets: pd.DataFrame, sector_strength: pd.DataFrame, session_summary: pd.DataFrame, industry: pd.DataFrame) -> Tuple[List[float], List[float]]:
    from scripts.order_engine import get_market_regime, compute_features, conviction_score, classify_action
    regime = get_market_regime(session_summary, sector_strength, policy)
    snap = snapshot.set_index('Ticker')
    met = metrics.set_index('Ticker')
    pre = presets.set_index('Ticker') if not presets.empty else pd.DataFrame()
    held = set(portfolio['Ticker'].astype(str).tolist()) if not portfolio.empty else set()
    uni = set(industry['Ticker'].astype(str).tolist()) if not industry.empty else set(metrics['Ticker'].astype(str).tolist())
    secmap = {str(r['Ticker']).upper(): r.get('Sector') for _, r in industry.iterrows()}

    def score_of(t: str) -> Tuple[bool, float]:
        if t not in snap.index:
            return False, 0.0
        srow = snap.loc[t]
        mrow = met.loc[t] if t in met.index else None
        feats = compute_features(t, srow, mrow, normalizers=None)
        sc = conviction_score(feats, secmap.get(t, ''), regime, t)
        is_add = classify_action(True, sc, feats, regime, ticker=t) == 'add'
        is_new = classify_action(False, sc, feats, regime, ticker=t) == 'new'
        return is_add or is_new, sc if (is_add or is_new) else 0.0

    add_scores: List[float] = []
    for t in held:
        ok, sc = score_of(t)
        if ok:
            # ok indicates add or new; for holders we only accept add
            srow = snap.loc[t]
            mrow = met.loc[t] if t in met.index else None
            feats = compute_features(t, srow, mrow, None)
            if classify_action(True, sc, feats, regime, ticker=t) == 'add':
                add_scores.append(sc)
    new_scores: List[float] = []
    for t in uni:
        if t in held or t not in snap.index:
            continue
        ok, sc = score_of(t)
        if ok:
            srow = snap.loc[t]
            mrow = met.loc[t] if t in met.index else None
            feats = compute_features(t, srow, mrow, None)
            if classify_action(False, sc, feats, regime, ticker=t) == 'new':
                new_scores.append(sc)
    return add_scores, new_scores


def _solve_q(n: int, target: int) -> float:
    """Return the orthodox quantile gate for keeping ``target`` of ``n``."""

    if n <= 0:
        return 0.995
    keep = int(target)
    if keep <= 0:
        return 0.995
    if keep >= n:
        return 0.0
    base = 1.0 - (float(keep) / float(n))
    return float(min(0.995, max(0.0, base)))


def calibrate(*, write: bool = False) -> Tuple[float, float, int, int]:
    pol_path = ORDERS_DIR / 'policy_overrides.json'
    if not pol_path.exists():
        raise SystemExit('Missing out/orders/policy_overrides.json for quantile gates calibration')
    pol = _load_json(pol_path)

    try:
        add_max = int(pol.get('add_max'))
        new_max = int(pol.get('new_max'))
    except Exception as exc:
        raise SystemExit(f'invalid add_max/new_max: {exc}') from exc
    if add_max < 0 or new_max < 0:
        raise SystemExit('add_max/new_max must be non-negative')

    portfolio = _load_df('portfolio_clean.csv')
    snapshot = _load_df('snapshot.csv')
    metrics = _load_df('metrics.csv')
    presets = pd.read_csv(OUT_DIR / 'presets_all.csv') if (OUT_DIR / 'presets_all.csv').exists() else pd.DataFrame()
    sector_strength = _load_df('sector_strength.csv')
    session_summary = _load_df('session_summary.csv')
    industry = _industry_df(metrics)

    add_scores, new_scores = _candidate_scores(pol, portfolio, snapshot, metrics, presets, sector_strength, session_summary, industry)
    n_add = len(add_scores)
    n_new = len(new_scores)
    if n_add <= 1 and n_new <= 1:
        raise SystemExit('Insufficient candidates to calibrate q_add/q_new')

    q_add = _solve_q(n_add, add_max)
    q_new = _solve_q(n_new, new_max)

    if write:
        obj = pol
        th = dict(obj.get('thresholds', {}) or {})
        th['q_add'] = float(q_add)
        th['q_new'] = float(q_new)
        obj['thresholds'] = th
        pol_path.write_text(json.dumps(obj, ensure_ascii=False, indent=2), encoding='utf-8')
    return float(q_add), float(q_new), n_add, n_new


if __name__ == '__main__':
    v = calibrate(write=True)
    print(f"q_add={v[0]:.3f}, q_new={v[1]:.3f} (pools add={v[2]}, new={v[3]})")

