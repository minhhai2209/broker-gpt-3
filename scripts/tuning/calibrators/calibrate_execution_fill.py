from __future__ import annotations

"""
Deterministic calibration for execution.fill parameters.

Goals
- Set a realistic acceptance target (target_prob) from the current NEW candidate
  distribution using the engine’s probability model, not a fixed constant.
- Adapt horizon/window and microstructure knobs smoothly with volatility/turnover.

Method
1) Build regime and candidate list using conviction scores (pre-quantile gate) so we
   can rank prospective NEWs without relying on previous hard filters.
2) Evaluate the top slice by conviction (bounded window) and compute best POF across
   0..max_chase_ticks steps with no-cross at the inside, using the same primitives as
   the engine.
3) With K = new_max from policy, set target_prob = quantile(P, 1 - K/|P|),
   clamp to [0.20, 0.75], and apply a distribution-aware floor so at least one
   candidate can satisfy the acceptance test when microstructure allows.
4) Set horizon/window from index_atr_percentile; joiner_factor from turnover percentile.

Inputs
- out/orders/policy_overrides.json (or config/policy_overrides.json)
- out/portfolio_clean.csv, out/snapshot.csv, out/metrics.csv, out/presets_all.csv (optional)
- out/session_summary.csv, out/sector_strength.csv

Outputs
- Writes execution.fill {horizon_s, window_sigma_s, window_vol_s, target_prob,
  max_chase_ticks, cancel_ratio_per_min, joiner_factor, no_cross}

Fail‑fast
- Missing core CSVs -> SystemExit. When NEW pool is empty, skip target_prob update (not an error).
"""

from pathlib import Path
from typing import Dict, List, Optional
import json
import math
import re
import numpy as np
import pandas as pd

from scripts.tuning.calibrators.policy_write import write_policy

BASE_DIR = Path(__file__).resolve().parents[3]
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
        raise SystemExit(f'Missing policy file: {src}')
    return json.loads(_strip(src.read_text(encoding='utf-8')))


def _load_csv(name: str) -> pd.DataFrame:
    path = OUT_DIR / name
    if not path.exists():
        raise SystemExit(f'Missing out/{name}')
    df = pd.read_csv(path)
    if df is None or df.empty:
        raise SystemExit(f'out/{name} is empty')
    return df


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


def _clip(v: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, float(v)))


def calibrate(*, write: bool = True) -> Dict[str, float]:
    pol = _load_policy()
    # Artifacts
    snapshot = _load_csv('snapshot.csv').set_index('Ticker')
    metrics = _load_csv('metrics.csv').set_index('Ticker')
    presets = _load_csv('presets_all.csv') if (OUT_DIR / 'presets_all.csv').exists() else pd.DataFrame()
    pre_idx = presets.set_index('Ticker') if not presets.empty else pd.DataFrame()
    session = _load_csv('session_summary.csv')
    sectors = _load_csv('sector_strength.csv')
    portfolio = _load_csv('portfolio_clean.csv')
    if 'Ticker' not in portfolio.columns:
        raise SystemExit('portfolio_clean.csv missing Ticker')
    held = set(str(t).strip().upper() for t in portfolio['Ticker'].astype(str))

    try:
        from scripts.orders.order_engine import (
            get_market_regime,
            compute_features,
            conviction_score,
            classify_action,
            pick_limit_price,
            _estimate_new_buy_fill_env,
            _pof_new_buy,
        )
    except Exception as exc:
        raise SystemExit(f'Unable to import engine primitives: {exc}') from exc

    regime = get_market_regime(session, sectors, pol)
    normalizers = _build_normalizers(metrics.reset_index())

    # Candidate pool before quantile gates
    new_candidates: List[str] = []
    scores: Dict[str, float] = {}
    for t in snapshot.index.astype(str):
        if t in held:
            continue
        mrow = metrics.loc[t] if t in metrics.index else None
        feats = compute_features(t, snapshot.loc[t], mrow, normalizers)
        sc = float(conviction_score(feats, sector='', regime=regime, ticker=t))
        # Use conviction ranking directly (pre‑gate) to form evaluation pool
        new_candidates.append(t)
        scores[t] = sc

    # Fill config template — will be adapted below
    exec_conf = dict(pol.get('execution', {}) or {})
    fill_conf = dict(exec_conf.get('fill', {}) or {})
    max_chase_ticks = int(float(fill_conf.get('max_chase_ticks', 2) or 2))
    no_cross = bool(fill_conf.get('no_cross', True))
    cancel_ratio_per_min = float(fill_conf.get('cancel_ratio_per_min', 0.15) or 0.15)

    # Vol‑adaptive horizon/window
    atr_pctile = float(getattr(regime, 'index_atr_percentile', 0.5) or 0.5)
    mf = dict(pol.get('market_filter', {}) or {})
    soft = float(mf.get('index_atr_soft_pct', 0.90) or 0.90)
    hard = float(mf.get('index_atr_hard_pct', 0.97) or 0.97)
    if hard <= soft:
        hard = max(soft + 0.01, 0.96)
    vol_norm = 1.0 - _clip((atr_pctile - soft) / (hard - soft), 0.0, 1.0)  # 1=calm, 0=stressed
    horizon_s = int(_clip(round(60 + 120 * vol_norm), 45, 180))
    window_sigma_s = int(_clip(round(horizon_s * 0.8), 30, 240))
    window_vol_s = int(_clip(round(horizon_s * 1.2), 60, 300))

    # Turnover‑adaptive joiner
    t_pct = float(getattr(regime, 'turnover_percentile', 0.5) or 0.5)
    joiner_factor = float(_clip(0.05 + 0.20 * t_pct, 0.05, 0.50))

    # Compute best POF for each NEW candidate using the engine’s primitives
    lot = int(float((pol.get('sizing', {}) or {}).get('min_lot', 100) or 100))
    fill_eval = {
        'horizon_s': horizon_s,
        'window_sigma_s': window_sigma_s,
        'window_vol_s': window_vol_s,
        'cancel_ratio_per_min': cancel_ratio_per_min,
        'joiner_factor': joiner_factor,
        'no_cross': no_cross,
        'target_prob': float(fill_conf.get('target_prob', 0.55) or 0.55),  # placeholder during eval
        'max_chase_ticks': max_chase_ticks,
    }

    # Focus on top-N by score (closer to actual selection) to set acceptance fairly
    new_max = int(float(pol.get('new_max', 0) or 0))
    top_n = int(max(5, min(20, (new_max if new_max > 0 else 3) * 3)))
    top_pool = sorted(new_candidates, key=lambda x: scores.get(x, 0.0), reverse=True)[:top_n]

    pofs: List[float] = []
    for t in top_pool:
        srow = snapshot.loc[t]
        mrow = metrics.loc[t] if t in metrics.index else None
        prow = pre_idx.loc[t] if not pre_idx.empty and t in pre_idx.index else None
        try:
            base_limit = float(pick_limit_price(t, 'BUY', srow, prow, mrow, regime))
        except Exception:
            # Fallback: inside bid/market
            price = srow.get('Price') if 'Price' in srow.index else srow.get('P')
            base_limit = float(price) if pd.notna(price) else 0.0
        env = _estimate_new_buy_fill_env(srow, mrow, lot, fill_eval)
        if env is None:
            continue
        tick = env.get('tick', 0.0)
        if not tick or tick <= 0:
            continue
        bid = env.get('bid', 0.0)
        market = float(srow.get('Price') if 'Price' in srow.index else srow.get('P')) if 'Price' in srow.index or 'P' in srow.index else None
        start = min(base_limit, bid)
        best = 0.0
        for step in range(max_chase_ticks + 1):
            cand = start + step * tick
            if market is not None:
                cand = min(cand, float(market))
            if no_cross and cand >= env['ask']:
                if step == 0:
                    cand = min(start, env['bid'])
                else:
                    break
            # pof at candidate
            pof, _, _, _, _ = _pof_new_buy(cand, env)
            best = max(best, float(pof))
        pofs.append(best)

    # Compute target_prob quantile from best POFs if pool non‑empty
    target_prob = float(fill_conf.get('target_prob', 0.0) or 0.0)
    if pofs:
        p = np.array([_clip(x, 0.0, 1.0) for x in pofs], dtype=float)
        n = int(p.size)
        keep = int(max(0, min(new_max, n)))
        if keep <= 0:
            q = 0.995
        else:
            q = float(1.0 - (keep / float(n)))
            q = float(_clip(q, 0.0, 0.995))
        _ = float(np.quantile(p, q))  # not used when disabling guard
        # Disable fill-probability acceptance guard per operator policy
        target_prob = 0.0
    # else: keep 0.0 target_prob (no NEW pool today)

    out = {
        'horizon_s': horizon_s,
        'window_sigma_s': window_sigma_s,
        'window_vol_s': window_vol_s,
        'target_prob': target_prob,
        'max_chase_ticks': max_chase_ticks,
        'cancel_ratio_per_min': cancel_ratio_per_min,
        'joiner_factor': joiner_factor,
        'no_cross': True,
    }

    if write:
        obj = pol
        ex = dict(obj.get('execution', {}) or {})
        fill = dict(ex.get('fill', {}) or {})
        fill.update(out)
        ex['fill'] = fill
        obj['execution'] = ex
        write_policy(
            calibrator=__name__,
            policy=obj,
            orders_path=ORDERS_PATH,
            config_path=CONFIG_PATH,
        )
    return out


if __name__ == '__main__':
    vals = calibrate(write=True)
    print('[calibrate.execution_fill] ' + ', '.join(f"{k}={v}" for k, v in vals.items()))
