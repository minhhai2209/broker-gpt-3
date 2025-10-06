from __future__ import annotations

from pathlib import Path
from typing import Optional, Tuple

import pandas as pd
import numpy as np


# Resolve repo root (this file lives under repo_root/scripts/data_fetching)
ROOT_DIR = Path(__file__).resolve().parents[2]
DATA_DIR = ROOT_DIR / 'data'
OUT_DIR = ROOT_DIR / 'out'


def _read_csv_allow(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Missing global factors CSV: {path}")
    df = pd.read_csv(path)
    if df.empty:
        raise ValueError(f"Global factors CSV is empty: {path}")
    return df


def _prepare_df(df: pd.DataFrame) -> pd.DataFrame:
    # Expected columns (case-insensitive):
    #   - Date (required)
    #   - SPX_Close (or SPX) — optional
    #   - US_EPU (or EPU_US) — optional
    #   - DXY (or DXY_Close) — optional
    #   - Brent (or Brent_Close) — optional
    cols = {c.lower(): c for c in df.columns}
    date_col = cols.get('date')
    if not date_col:
        raise ValueError("global_factors: required column 'Date' not found")
    def _find_one(names):
        for n in names:
            if n.lower() in cols:
                return cols[n.lower()]
        return None
    spx_col = _find_one(['SPX_Close', 'SPX', 'S&P500', 'SP500'])
    epu_col = _find_one(['US_EPU', 'EPU_US', 'EPU'])
    dxy_col = _find_one(['DXY', 'DXY_Close'])
    brent_col = _find_one(['Brent', 'Brent_Close'])

    out = pd.DataFrame()
    out['Date'] = pd.to_datetime(df[date_col], errors='coerce')
    out = out.dropna(subset=['Date']).sort_values('Date')
    if spx_col and df[spx_col].notna().any():
        out['SPX_Close'] = pd.to_numeric(df[spx_col], errors='coerce')
    else:
        out['SPX_Close'] = np.nan
    if epu_col and df[epu_col].notna().any():
        out['US_EPU'] = pd.to_numeric(df[epu_col], errors='coerce')
    else:
        out['US_EPU'] = np.nan
    if dxy_col and df[dxy_col].notna().any():
        out['DXY'] = pd.to_numeric(df[dxy_col], errors='coerce')
    else:
        out['DXY'] = np.nan
    if brent_col and df[brent_col].notna().any():
        out['Brent'] = pd.to_numeric(df[brent_col], errors='coerce')
    else:
        out['Brent'] = np.nan
    return out


def _percentile_rank(series: pd.Series, value: float | None) -> float:
    arr = pd.to_numeric(series, errors='coerce').dropna().to_numpy()
    if arr.size == 0 or value is None or pd.isna(value):
        return float('nan')
    less = float(np.sum(arr < value))
    equal = float(np.sum(arr == value))
    rank = (less + 0.5 * equal) / float(arr.size)
    return max(0.0, min(1.0, rank))


def compute_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    # SPX drawdown vs rolling max over full series (proxy for medium-term stress)
    if 'SPX_Close' in df.columns and df['SPX_Close'].notna().any():
        px = pd.to_numeric(df['SPX_Close'], errors='coerce')
        roll_max = px.cummax().replace(0, np.nan)
        dd = 1.0 - (px / roll_max)
        df['SPX_Drawdown_Pct'] = dd.clip(lower=0).astype(float)
        # 63d momentum (ratio - 1)
        shifted = px.shift(63)
        df['SPX_Mom_63d'] = (px / shifted - 1.0).replace([np.inf, -np.inf], np.nan).astype(float)
    else:
        df['SPX_Drawdown_Pct'] = np.nan
        df['SPX_Mom_63d'] = np.nan

    # US EPU percentile over its history
    if 'US_EPU' in df.columns and df['US_EPU'].notna().any():
        df['US_EPU_Percentile'] = df['US_EPU'].expanding().apply(lambda s: _percentile_rank(s, s.iloc[-1]), raw=False)
    else:
        df['US_EPU_Percentile'] = np.nan

    # DXY percentile (higher = stronger USD)
    if 'DXY' in df.columns and df['DXY'].notna().any():
        df['DXY_Percentile'] = df['DXY'].expanding().apply(lambda s: _percentile_rank(s, s.iloc[-1]), raw=False)
    else:
        df['DXY_Percentile'] = np.nan

    # Brent 63d momentum (ratio - 1)
    if 'Brent' in df.columns and df['Brent'].notna().any():
        px = pd.to_numeric(df['Brent'], errors='coerce')
        df['Brent_Mom_63d'] = (px / px.shift(63) - 1.0).replace([np.inf, -np.inf], np.nan)
    else:
        df['Brent_Mom_63d'] = np.nan
    return df[['Date', 'SPX_Close', 'SPX_Drawdown_Pct', 'SPX_Mom_63d', 'US_EPU', 'US_EPU_Percentile', 'DXY', 'DXY_Percentile', 'Brent', 'Brent_Mom_63d']]


def ensure_global_factors(out_dir: str | Path = OUT_DIR, data_path: Optional[str | Path] = None) -> Tuple[pd.DataFrame, pd.DataFrame]:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    src = Path(data_path) if data_path else (DATA_DIR / 'global_factors.csv')
    try:
        raw = _read_csv_allow(src)
        base = _prepare_df(raw)
        feats = compute_features(base)
        feats.to_csv(out_dir / 'global_factors_features.csv', index=False)
        # Snapshot latest row for convenience
        if not feats.empty:
            feats.tail(1).to_csv(out_dir / 'global_factors_snapshot.csv', index=False)
        print(f"Wrote global factors features to {out_dir / 'global_factors_features.csv'}")
        return base, feats
    except FileNotFoundError:
        print(f"[info] Optional global factors file not found: {src}")
        return pd.DataFrame(), pd.DataFrame()
    except Exception as exc:
        # Optional feature: do not fail pipeline, but record issue for diagnostics
        print(f"[warn] Failed to prepare global factors: {exc}")
        return pd.DataFrame(), pd.DataFrame()


if __name__ == '__main__':
    ensure_global_factors()
