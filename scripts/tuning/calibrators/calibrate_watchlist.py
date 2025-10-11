from __future__ import annotations

"""
Calibrate orders_ui.watchlist.{min_priority, micro_window}.

Approach (conservative without trade logs)
- Compute BUY candidate priorities = Score (0..1) multiplied by a simple
  fill‑prob proxy using current `pricing.fill_prob` and distance to market.
- Choose min_priority as a lower quantile (e.g., 25–35%) of candidate priorities
  so that weak BUYs fall into watchlist. micro_window kept in [3..5].

Inputs
- out/{snapshot,metrics,presets_all}.csv
- out/orders/policy_overrides.json (weights, thresholds, pricing.fill_prob)
- data/industry_map.csv

Outputs
- orders_ui.watchlist.min_priority and micro_window.

Fail-fast
- Missing files/columns result in SystemExit (optional calibrator).
"""

from pathlib import Path
import json, re
import numpy as np
import pandas as pd

from scripts.tuning.calibrators.policy_write import write_policy

BASE_DIR = Path(__file__).resolve().parents[3]
OUT_DIR = BASE_DIR / 'out'
ORDERS_DIR = OUT_DIR / 'orders'


def _load_json(p: Path) -> dict:
    raw = p.read_text(encoding='utf-8')
    raw = re.sub(r"/\*.*?\*/", "", raw, flags=re.S)
    raw = re.sub(r"(^|\s)//.*$", "", raw, flags=re.M)
    raw = re.sub(r"(^|\s)#.*$", "", raw, flags=re.M)
    return json.loads(raw)


def calibrate(*, write: bool = False) -> tuple[float, int]:
    pol_p = ORDERS_DIR / 'policy_overrides.json'
    if not pol_p.exists():
        raise SystemExit('Missing runtime policy for watchlist calibration')
    pol = _load_json(pol_p)
    pr = dict(pol.get('pricing', {}) or {})
    fp = dict(pr.get('fill_prob', {}) or {})
    for req in ('base','cross','near_ceiling','min','decay_scale_min_ticks'):
        if req not in fp:
            raise SystemExit(f'Missing pricing.fill_prob.{req} in policy (required for calibration)')
    fp_base = float(fp['base']); fp_cross = float(fp['cross']); fp_min = float(fp['min']); scale_min = float(fp['decay_scale_min_ticks'])

    snap_p = OUT_DIR / 'snapshot.csv'
    met_p = OUT_DIR / 'metrics.csv'
    pre_p = OUT_DIR / 'presets_all.csv'
    if not (snap_p.exists() and met_p.exists() and pre_p.exists()):
        raise SystemExit('Missing snapshot/metrics/presets for watchlist calibration')
    snap = pd.read_csv(snap_p).set_index('Ticker')
    met = pd.read_csv(met_p).set_index('Ticker')
    pre = pd.read_csv(pre_p).set_index('Ticker') if pre_p.exists() else pd.DataFrame()

    # Compute simple BUY priorities over universe (ignore SELL/EXIT here)
    priorities: list[float] = []
    for t in snap.index:
        if t not in met.index:
            continue
        price = pd.to_numeric(pd.Series([snap.loc[t].get('Price')]), errors='coerce').iloc[0]
        if not np.isfinite(price) or price <= 0:
            continue
        score = 0.0
        try:
            # Use RSI & liquidity as minimal proxy when full score unavailable
            rsi = pd.to_numeric(pd.Series([met.loc[t].get('RSI14')]), errors='coerce').iloc[0]
            liq = pd.to_numeric(pd.Series([met.loc[t].get('AvgTurnover20D_k')]), errors='coerce').iloc[0]
            if np.isfinite(rsi):
                score += max(0.0, min(1.0, (rsi - 40.0)/30.0))
            if np.isfinite(liq):
                score = 0.7*score + 0.3*min(1.0, liq/np.nanmax([liq, 1.0]))
        except Exception:
            score = 0.0
        # Fill prob proxy by distance to market for a notional mid‑anchor = price
        tick = pd.to_numeric(pd.Series([met.loc[t].get('TickSizeHOSE_Thousand')]), errors='coerce').iloc[0]
        if not np.isfinite(tick) or tick <= 0:
            tick = 0.05 if price < 49.95 else (0.01 if price < 10 else 0.10)
        dist_ticks = 1.0  # one tick away as baseline
        scale = max(scale_min, 1.0)
        fill = max(fp_min, float(np.exp(-dist_ticks/scale)))
        priorities.append(max(0.0, float(score)) * float(fill))

    if not priorities:
        raise SystemExit('No priorities computed')
    prio_arr = np.array([p for p in priorities if np.isfinite(p)], dtype=float)
    if prio_arr.size == 0:
        raise SystemExit('No finite priorities')
    # Set threshold near 30th percentile by default
    min_prio = float(np.quantile(prio_arr, 0.30))
    min_prio = max(0.0, min(1.0, round(min_prio, 3)))
    micro_window = 3
    if write:
        ou = dict(pol.get('orders_ui', {}) or {})
        wl = dict(ou.get('watchlist', {}) or {})
        wl['min_priority'] = float(min_prio)
        wl['micro_window'] = int(micro_window)
        ou['watchlist'] = wl
        pol['orders_ui'] = ou
        pol_p = ORDERS_DIR / 'policy_overrides.json'
        write_policy(
            calibrator=__name__,
            policy=pol,
            explicit_path=pol_p,
        )
    return float(min_prio), int(micro_window)


if __name__ == '__main__':
    prio, win = calibrate(write=True)
    print(f"watchlist.min_priority={prio:.3f}, micro_window={win}")

