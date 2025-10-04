from __future__ import annotations

"""
Calibrate leader_min_rsi and leader_min_mom_norm from cross‑sectional quantiles.

Orthodox basis
- Define "leader" via top quantiles of RSI14 and momentum percentile (MomRetNorm).
- leader_min_rsi = quantile(RSI14, q_rsi)
- leader_min_mom_norm = quantile(MomRetNorm, q_mom)

Inputs
- out/metrics.csv (requires RSI14; for momentum either MomRetNorm or MomRet_12_1 to derive percentile ranks)
- config/policy_overrides.json calibration_targets.market_filter.{leader_rsi_q, leader_mom_q}

Outputs
- Writes market_filter.leader_min_rsi and market_filter.leader_min_mom_norm to config.

Fail‑fast on missing inputs/columns/targets.
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


def _strip(text: str) -> str:
    text = re.sub(r"/\*.*?\*/", "", text, flags=re.S)
    text = re.sub(r"(^|\s)//.*$", "", text, flags=re.M)
    text = re.sub(r"(^|\s)#.*$", "", text, flags=re.M)
    return text


def _load_policy() -> dict:
    src = ORDERS_PATH if ORDERS_PATH.exists() else CONFIG_PATH
    if not src.exists():
        raise SystemExit(f'Missing policy file: {src}')
    return json.loads(_strip(src.read_text(encoding='utf-8')))


def _save_policy(obj: dict) -> None:
    target = ORDERS_PATH if ORDERS_PATH.exists() else CONFIG_PATH
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(obj, ensure_ascii=False, indent=2), encoding='utf-8')


def _load_metrics() -> pd.DataFrame:
    path = OUT_DIR / 'metrics.csv'
    if not path.exists():
        raise SystemExit('Missing out/metrics.csv for leader gates calibration')
    df = pd.read_csv(path)
    required = {'Ticker', 'RSI14'}
    if not required.issubset(df.columns):
        miss = ', '.join(sorted(required - set(df.columns)))
        raise SystemExit(f'metrics.csv missing columns: {miss}')
    return df


def calibrate(write: bool = False) -> tuple[float, float]:
    pol = _load_policy()
    tgt = (pol.get('calibration_targets', {}) or {}).get('market_filter', {})
    if 'leader_rsi_q' not in tgt:
        raise SystemExit('Missing calibration_targets.market_filter.leader_rsi_q')
    if 'leader_mom_q' not in tgt:
        raise SystemExit('Missing calibration_targets.market_filter.leader_mom_q')
    try:
        q_rsi = float(tgt['leader_rsi_q'])
        q_mom = float(tgt['leader_mom_q'])
    except Exception as exc:
        raise SystemExit(f'Invalid leader quantiles: {exc}') from exc
    for q in (q_rsi, q_mom):
        if not (0.0 <= q <= 1.0):
            raise SystemExit('leader quantiles must be in [0,1]')

    df = _load_metrics()
    rsi = pd.to_numeric(df['RSI14'], errors='coerce').replace([np.inf, -np.inf], np.nan).dropna()
    if rsi.empty:
        raise SystemExit('RSI14 series empty after cleaning')
    leader_min_rsi = float(np.quantile(rsi.to_numpy(), q_rsi))

    if 'MomRetNorm' in df.columns:
        mom = pd.to_numeric(df['MomRetNorm'], errors='coerce').replace([np.inf, -np.inf], np.nan).dropna()
        if mom.empty:
            raise SystemExit('MomRetNorm series empty after cleaning')
        leader_min_mom = float(np.quantile(mom.to_numpy(), q_mom))
    elif 'MomRet_12_1' in df.columns:
        base = pd.to_numeric(df['MomRet_12_1'], errors='coerce').replace([np.inf, -np.inf], np.nan).dropna()
        if base.empty:
            raise SystemExit('MomRet_12_1 series empty after cleaning')
        # Convert to percentile ranks in [0..1]
        s = base.sort_values()
        rank_map = {t: i / max(len(s) - 1.0, 1.0) for i, t in enumerate(s.to_list())}
        # Map via ranks by value is not stable with duplicates; use percentile on the vector instead
        mom_norm = base.rank(pct=True)
        leader_min_mom = float(np.quantile(mom_norm.to_numpy(), q_mom))
    elif 'RS_Trend50' in df.columns:
        # Fallback: use RS slope 50d as momentum proxy and convert to percentile ranks
        rs = pd.to_numeric(df['RS_Trend50'], errors='coerce').replace([np.inf, -np.inf], np.nan).dropna()
        if rs.empty:
            raise SystemExit('RS_Trend50 series empty after cleaning')
        rs_pct = rs.rank(pct=True)
        leader_min_mom = float(np.quantile(rs_pct.to_numpy(), q_mom))
    else:
        raise SystemExit('metrics.csv missing MomRetNorm, MomRet_12_1 and RS_Trend50 for leader momentum calibration')

    if write:
        obj = pol
        mf = dict(obj.get('market_filter', {}) or {})
        mf['leader_min_rsi'] = float(leader_min_rsi)
        mf['leader_min_mom_norm'] = float(leader_min_mom)
        obj['market_filter'] = mf
        _save_policy(obj)
    return float(leader_min_rsi), float(leader_min_mom)


def main():
    from argparse import ArgumentParser
    ap = ArgumentParser()
    ap.add_argument('--write', action='store_true')
    args = ap.parse_args()
    rsi, mom = calibrate(write=args.write)
    print(f"[calibrate.leaders] leader_min_rsi={rsi:.2f}, leader_min_mom_norm={mom:.2f}")


if __name__ == '__main__':
    main()
