from __future__ import annotations

import os
from datetime import datetime, timedelta, timezone
from typing import List, Dict, Optional, Set, Tuple
from pathlib import Path
from scripts.utils import detect_session_phase_now_vn

import pandas as pd
import numpy as np
import requests

VN_TZ = timezone(timedelta(hours=7))


def load_universe_from_industry_map() -> Set[str]:
    cands: Set[str] = set()
    p = Path('data/industry_map.csv')
    if p.exists():
        df = pd.read_csv(p)
        if 'Ticker' in df.columns:
            cands.update(df['Ticker'].astype(str).str.upper().tolist())
    return cands


def fetch_dchart_history(symbol: str, resolution: str, frm: int, to: int) -> Optional[Dict]:
    url = f"https://dchart-api.vndirect.com.vn/dchart/history?symbol={symbol}&resolution={resolution}&from={frm}&to={to}"
    headers = {
        'User-Agent': 'Mozilla/5.0',
        'Referer': 'https://dchart.vndirect.com.vn/'
    }
    last_exc: Exception | None = None
    for attempt in range(3):
        try:
            r = requests.get(url, timeout=10, headers=headers)
            r.raise_for_status()
            if not r.text or not r.text.strip():
                raise ValueError('empty body')
            data = r.json()
            if data.get('s') == 'ok' and data.get('t'):
                return data
            return None
        except Exception as exc:
            last_exc = exc
            import time as _time
            _time.sleep(0.5 * (attempt + 1))
            continue
    # Give up silently; snapshot builder will fallback to daily close
    print(f"[warn] snapshot dchart fetch failed for {symbol}: {type(last_exc).__name__}: {last_exc}")
    return None


def latest_price_from_dchart(symbol: str) -> Optional[float]:
    now = int(datetime.now(VN_TZ).timestamp())
    data = fetch_dchart_history(symbol, '1', now-3600*12, now)
    if data and data.get('c'):
        return float(data['c'][-1])
    return None


def _load_intraday_latest_map(latest_path: Path = Path('out/intraday/latest.csv')) -> Tuple[Dict[str, float], Optional[pd.Timestamp]]:
    """Load intraday latest.csv to a ticker->price map and return (map, max_ts) for staleness checks.
    Best-effort: returns ({} , None) if file missing or malformed.
    """
    try:
        if not latest_path.exists():
            return {}, None
        df = pd.read_csv(latest_path)
        if df.empty or 'Ticker' not in df.columns or 'Price' not in df.columns:
            return {}, None
        ts_col = None
        for cand in ('Ts','ts','Timestamp'):
            if cand in df.columns:
                ts_col = cand; break
        max_ts = None
        if ts_col:
            try:
                max_ts = pd.to_datetime(pd.to_numeric(df[ts_col], errors='coerce'), unit='s', utc=True).max()
            except Exception:
                max_ts = None
        mp = {str(t).upper(): float(p) for t, p in zip(df['Ticker'].astype(str), pd.to_numeric(df['Price'], errors='coerce')) if pd.notna(p)}
        return mp, max_ts
    except Exception:
        return {}, None

def _load_price_overrides(path: Path = Path('config/price_overrides.csv')) -> Dict[str, float]:
    try:
        if not path.exists():
            return {}
        df = pd.read_csv(path)
        if df.empty or 'Ticker' not in df.columns:
            return {}
        # Accept either 'Price' or 'Price_k' (thousand VND units)
        col = 'Price' if 'Price' in df.columns else ('Price_k' if 'Price_k' in df.columns else None)
        if not col:
            return {}
        return {str(t).upper(): float(p) for t, p in zip(df['Ticker'].astype(str), pd.to_numeric(df[col], errors='coerce')) if pd.notna(p)}
    except Exception:
        return {}


def build_snapshot(portfolio_csv: str, data_dir: str, out_path: str):
    dfp = pd.read_csv(portfolio_csv)
    # Expect columns: Ticker, Quantity, AvgCost (CostValue optional)
    req = {'Ticker', 'Quantity', 'AvgCost'}
    if not req.issubset(dfp.columns):
        raise RuntimeError('portfolio CSV missing required columns')
    pf_tickers = dfp['Ticker'].astype(str).str.upper().tolist()
    universe = load_universe_from_industry_map()
    # Always include key VN benchmarks for context
    index_benchmarks = {"VNINDEX", "VN30", "VN100"}
    tickers = sorted((set(pf_tickers) | universe | index_benchmarks))
    out: List[Dict[str, object]] = []
    # Prefer intraday only when session is active; else use EOD/overrides to avoid stale ticks
    intraday_map, _ = _load_intraday_latest_map()
    phase = detect_session_phase_now_vn().lower()
    if phase in ('pre', 'post'):
        intraday_map = {}
    overrides = _load_price_overrides()
    for t in tickers:
        # Use daily history for price fallback
        hist = None
        p = os.path.join(data_dir, f"{t}_daily.csv")
        has_hist = os.path.exists(p)
        if has_hist:
            hist = pd.read_csv(p)
        row: Dict[str, object] = {'Ticker': t}
        # 0) Operator override
        price = overrides.get(t)
        # 1) Intraday latest (best-effort)
        if price is None:
            price = intraday_map.get(t)
        # 2) Live 1-min dchart
        if price is None:
            price = latest_price_from_dchart(t)
        if price is not None and not np.isfinite(price):
            price = None
        if price is None:
            if hist is not None and not hist.empty and 'close' in hist.columns:
                closes = pd.to_numeric(hist['close'], errors='coerce').dropna()
                if not closes.empty:
                    price = float(closes.iloc[-1])
        row['Price'] = price if price is not None else ''
        out.append(row)

    op = Path(out_path)
    df_out = pd.DataFrame(out)
    cols = ['Ticker', 'Price']
    for c in cols:
        if c not in df_out.columns:
            df_out[c] = ''
    df_out = df_out[cols]
    df_out.to_csv(op, index=False)
    print(f"Wrote {out_path}")


def build_snapshot_df(portfolio_df: pd.DataFrame, data_dir: str) -> pd.DataFrame:
    pf_tickers = portfolio_df['Ticker'].astype(str).str.upper().tolist() if not portfolio_df.empty else []
    universe = load_universe_from_industry_map()
    index_benchmarks = {"VNINDEX", "VN30", "VN100"}
    tickers = sorted((set(pf_tickers) | universe | index_benchmarks))
    out: List[Dict[str, object]] = []
    intraday_map, _ = _load_intraday_latest_map()
    phase = detect_session_phase_now_vn().lower()
    if phase in ('pre', 'post'):
        intraday_map = {}
    overrides = _load_price_overrides()
    for t in tickers:
        hist = None
        p = os.path.join(data_dir, f"{t}_daily.csv")
        has_hist = os.path.exists(p)
        if has_hist:
            hist = pd.read_csv(p)
        row: Dict[str, object] = {'Ticker': t}
        price = overrides.get(t)
        if price is None:
            price = intraday_map.get(t)
        if price is None:
            price = latest_price_from_dchart(t)
        if price is not None and not np.isfinite(price):
            price = None
        if price is None:
            if hist is not None and not hist.empty and 'close' in hist.columns:
                closes = pd.to_numeric(hist['close'], errors='coerce').dropna()
                if not closes.empty:
                    price = float(closes.iloc[-1])
        row['Price'] = price if price is not None else ''
        out.append(row)
    df_out = pd.DataFrame(out)
    cols = ['Ticker', 'Price']
    for c in cols:
        if c not in df_out.columns:
            df_out[c] = ''
    return df_out[cols]
