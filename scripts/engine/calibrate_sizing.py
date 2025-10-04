from __future__ import annotations

"""
Calibrate sizing.cov_reg via Ledoit–Wolf shrinkage intensity towards identity.

Rationale (orthodox):
- Sample covariance is noisy in finite windows. Ledoit–Wolf (2004) proposes
  optimal shrinkage to a scaled identity target F = mu*I, with closed-form
  shrinkage intensity. Our engine uses ridge regularization (cov + reg*I),
  which we map approximately via reg = lambda* * mu.

Inputs
- out/prices_history.csv with daily Close for universe tickers.
- config/policy_overrides.json for sizing.cov_lookback_days.

Output
- Writes sizing.cov_reg (non-negative scalar) back to config.

Fail-fast on missing files/columns or insufficient data.
"""

from pathlib import Path
import json
import re
from typing import Tuple

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


def _load_returns(lookback: int) -> pd.DataFrame:
    path = OUT_DIR / 'prices_history.csv'
    if not path.exists():
        raise SystemExit('Missing out/prices_history.csv for covariance calibration')
    df = pd.read_csv(path)
    for c in ('Date', 'Ticker', 'Close'):
        if c not in df.columns:
            raise SystemExit(f'prices_history.csv missing {c}')
    df['Ticker'] = df['Ticker'].astype(str).str.upper()
    df['Date'] = pd.to_datetime(df['Date'], errors='coerce')
    df = df.dropna(subset=['Date'])
    # exclude indices for cross-section stability
    idx_excl = {'VNINDEX', 'VN30', 'VN100'}
    df = df[~df['Ticker'].isin(idx_excl)]
    if df.empty:
        raise SystemExit('No non-index tickers found for covariance calibration')
    piv = df.pivot(index='Date', columns='Ticker', values='Close').sort_index()
    rets = piv.pct_change().dropna(how='all')
    rets = rets.dropna(axis=1, how='all')
    if rets.empty:
        raise SystemExit('Returns matrix empty after cleaning')
    if lookback > 0 and len(rets) > lookback:
        rets = rets.tail(lookback)
    # drop columns with any NaN in the window for stability
    rets = rets.dropna(axis=1, how='any')
    if rets.shape[1] < 2 or rets.shape[0] < 20:
        raise SystemExit('Insufficient returns data (need >=20 days and >=2 tickers)')
    return rets


def _ledoit_wolf_shrink_to_identity(X: np.ndarray) -> Tuple[float, float, float]:
    """Return (lambda_star, mu, gamma) for shrinkage towards mu*I.

    X: T x N matrix of demeaned returns.
    Implements phi_hat = (1/T) sum_t ||x_t x_t' - S||_F^2
    gamma_hat = ||S - mu I||_F^2
    lambda* = min(1, max(0, phi_hat / gamma_hat))
    """
    T, N = X.shape
    X = X - X.mean(axis=0, keepdims=True)
    S = (X.T @ X) / float(T)
    mu = float(np.trace(S)) / float(N)
    # phi_hat
    phi = 0.0
    for t in range(T):
        xt = X[t:t+1, :]  # 1 x N
        outer = xt.T @ xt  # N x N
        delta = outer - S
        phi += float(np.sum(delta * delta))
    phi /= float(T)
    # gamma_hat
    F = np.eye(N) * mu
    diff = S - F
    gamma = float(np.sum(diff * diff))
    if gamma <= 1e-12:
        lam = 1.0
    else:
        lam = max(0.0, min(1.0, phi / gamma))
    return lam, mu, gamma


def calibrate(write: bool = False) -> float:
    pol = _load_policy()
    try:
        sizing = pol.get('sizing', {}) or {}
        lookback = int(float(sizing.get('cov_lookback_days', 90)))
    except Exception as exc:
        raise SystemExit(f'Invalid sizing.cov_lookback_days: {exc}') from exc
    rets = _load_returns(lookback)
    X = rets.to_numpy(dtype=float)
    lam, mu, _ = _ledoit_wolf_shrink_to_identity(X)
    cov_reg = float(max(0.0, lam * mu))
    if write:
        obj = pol
        sz = dict(obj.get('sizing', {}) or {})
        sz['cov_reg'] = cov_reg
        obj['sizing'] = sz
        _save_policy(obj)
    return cov_reg


def main():
    from argparse import ArgumentParser
    ap = ArgumentParser()
    ap.add_argument('--write', action='store_true')
    args = ap.parse_args()
    v = calibrate(write=args.write)
    print(f"[calibrate.sizing] cov_reg = {v:.6f}")


if __name__ == '__main__':
    main()
