from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List

import pandas as pd
import numpy as np


def load_snapshot(path: Path) -> pd.DataFrame:
    if path.suffix.lower() == '.csv':
        df = pd.read_csv(path)
    else:
        df = pd.read_json(path)
    for col in ['Ticker','Price']:
        if col not in df.columns:
            df[col] = pd.NA
    df['Ticker'] = df['Ticker'].astype(str).str.upper()
    df['Price'] = pd.to_numeric(df['Price'], errors='coerce')
    return df


def _indicators_from_history(ph_path: Path) -> pd.DataFrame:
    cols = ['Date','Ticker','Close','High','Low']
    ph = pd.read_csv(ph_path, usecols=cols)
    if ph.empty:
        raise ValueError(f"Prices history CSV has no data: {ph_path}")
    ph['Ticker'] = ph['Ticker'].astype(str).str.upper()
    ph['Date'] = pd.to_datetime(ph['Date'], errors='coerce')
    ph = ph.dropna(subset=['Date'])
    ph = ph.sort_values(['Ticker','Date'])
    from scripts.indicators import ma, rsi_wilder, atr_wilder
    rows: List[Dict[str, float]] = []
    for t, g in ph.groupby('Ticker'):
        close = pd.to_numeric(g['Close'], errors='coerce')
        ma20 = ma(close, 20).iloc[-1] if len(close) else np.nan
        ma50 = ma(close, 50).iloc[-1] if len(close) else np.nan
        ma200 = ma(close, 200).iloc[-1] if len(close) else np.nan
        high = pd.to_numeric(g['High'], errors='coerce')
        low = pd.to_numeric(g['Low'], errors='coerce')
        atr14 = atr_wilder(high, low, close, 14).iloc[-1] if len(close) else np.nan
        rsi14 = rsi_wilder(close, 14).iloc[-1] if len(close) else np.nan
        rows.append({'Ticker': t,
                     'MA20': float(ma20) if np.isfinite(ma20) else np.nan,
                     'MA50': float(ma50) if np.isfinite(ma50) else np.nan,
                     'MA200': float(ma200) if np.isfinite(ma200) else np.nan,
                     'RSI14': float(rsi14) if np.isfinite(rsi14) else np.nan,
                     'ATR14': float(atr14) if np.isfinite(atr14) else np.nan})
    return pd.DataFrame(rows)


def compute_metrics(df: pd.DataFrame) -> Dict[str, float]:
    n = len(df)
    if n == 0:
        return {}
    def pct(cond: pd.Series) -> float:
        k = int(cond.sum())
        return round(k / n * 100.0, 2)
    above_ma20 = pct((df['MA20'].notna()) & (pd.to_numeric(df['Price'], errors='coerce') > pd.to_numeric(df['MA20'], errors='coerce')))
    above_ma50 = pct((df['MA50'].notna()) & (pd.to_numeric(df['Price'], errors='coerce') > pd.to_numeric(df['MA50'], errors='coerce')))
    above_ma200 = pct((df.get('MA200').notna() if 'MA200' in df.columns else pd.Series([False]*n)) & (
        pd.to_numeric(df['Price'], errors='coerce') > pd.to_numeric(df.get('MA200'), errors='coerce')
    )) if 'MA200' in df.columns else 0.0
    rsi_avg = float(pd.to_numeric(df['RSI14'], errors='coerce').dropna().mean()) if df['RSI14'].notna().any() else 0.0
    atr_pct = (pd.to_numeric(df['ATR14'], errors='coerce') / pd.to_numeric(df['Price'], errors='coerce')).replace([pd.NA, float('inf')], 0).dropna()
    atr_avg_pct = float(atr_pct.mean()) if not atr_pct.empty else 0.0
    dist_to_s1 = 0.0
    dist_to_r1 = 0.0
    return {
        'breadth_above_ma20_pct': above_ma20,
        'breadth_above_ma50_pct': above_ma50,
        'breadth_above_ma200_pct': above_ma200,
        'avg_rsi14': round(rsi_avg, 2),
        'avg_atr_pct': round(atr_avg_pct * 100.0, 2),
        'avg_dist_to_s1_pct': round(dist_to_s1, 2),
        'avg_dist_to_r1_pct': round(dist_to_r1, 2),
    }


def load_industry_map(path: Path) -> Dict[str, str]:
    import csv
    out: Dict[str, str] = {}
    with path.open(encoding='utf-8') as f:
        for row in csv.DictReader(f):
            t = row.get('Ticker','').strip().upper()
            s = row.get('Sector','').strip()
            if t:
                out[t] = s
    return out


def _validate_sector_mapping_or_die(tickers: List[str], mp: Dict[str, str]) -> None:
    idx_labels = {"VNINDEX", "VN30", "VN100"}
    unknown = []
    for t in tickers:
        t = str(t).upper()
        if t in idx_labels:
            continue
        s = mp.get(t)
        if not s:
            unknown.append(t)
    if unknown:
        missing = ", ".join(sorted(set(unknown)))
        raise SystemExit(f"Missing sector classification for: {missing}. Please update data/industry_map.csv")


def compute_sector_strength(snapshot: str = 'out/snapshot.csv', industry_map: str = 'data/industry_map.csv', out: str = 'out/sector_strength.csv') -> None:
    snap = load_snapshot(Path(snapshot))
    mp = load_industry_map(Path(industry_map))
    _validate_sector_mapping_or_die(snap['Ticker'].astype(str).str.upper().tolist(), mp)
    ind = _indicators_from_history(Path('out/prices_history.csv'))
    df = snap.merge(ind, on='Ticker', how='left')
    if 'Price' not in df.columns or df['Price'].isna().all():
        df['Price'] = df['MA20']
    index_labels = {"VNINDEX", "VN30", "VN100"}
    df['Sector'] = df['Ticker'].map(lambda t: 'Index' if str(t).upper() in index_labels else mp.get(str(t).upper()))
    rows = []
    for sector, g in df.groupby('Sector', dropna=False):
        m = compute_metrics(g)
        m['sector'] = str(sector)
        rows.append(m)
    overall = compute_metrics(df)
    overall['sector'] = 'Tất cả'
    rows.append(overall)
    import csv as _csv
    outp = Path(out)
    outp.parent.mkdir(parents=True, exist_ok=True)
    cols = ['sector','breadth_above_ma20_pct','breadth_above_ma50_pct','breadth_above_ma200_pct','avg_rsi14','avg_atr_pct','avg_dist_to_s1_pct','avg_dist_to_r1_pct']
    with outp.open('w', newline='', encoding='utf-8') as f:
        w = _csv.DictWriter(f, fieldnames=cols)
        w.writeheader()
        for r in rows:
            w.writerow({k: r.get(k, '') for k in cols})
    print(f"Wrote CSV {out}")


def compute_sector_strength_df(snapshot_df: pd.DataFrame, industry_map_df: pd.DataFrame, prices_history_df: pd.DataFrame) -> pd.DataFrame:
    mp = {}
    if not industry_map_df.empty and {'Ticker','Sector'}.issubset(industry_map_df.columns):
        tickers = industry_map_df['Ticker'].astype(str)
        sectors = industry_map_df['Sector'].astype(str).str.strip()
        mp = {str(t).upper(): s for t, s in zip(tickers, sectors)}
    # Validate mapping completeness for all non-index tickers
    _validate_sector_mapping_or_die(snapshot_df['Ticker'].astype(str).str.upper().tolist(), mp)
    ind = _indicators_from_history(Path('out/prices_history.csv')) if prices_history_df is None else None
    if prices_history_df is not None and not prices_history_df.empty:
        # Recompute indicators from provided history df
        ph = prices_history_df[['Date','Ticker','Close','High','Low']].copy()
        ph['Ticker'] = ph['Ticker'].astype(str).str.upper()
        ph['Date'] = pd.to_datetime(ph['Date'], errors='coerce')
        ph = ph.dropna(subset=['Date']).sort_values(['Ticker','Date'])
        from scripts.indicators import ma, rsi_wilder, atr_wilder
        rows: List[Dict[str, float]] = []
        for t, g in ph.groupby('Ticker'):
            close = pd.to_numeric(g['Close'], errors='coerce')
            ma20 = ma(close, 20).iloc[-1] if len(close) else np.nan
            ma50 = ma(close, 50).iloc[-1] if len(close) else np.nan
            high = pd.to_numeric(g['High'], errors='coerce')
            low = pd.to_numeric(g['Low'], errors='coerce')
            atr14 = atr_wilder(high, low, close, 14).iloc[-1] if len(close) else np.nan
            rsi14 = rsi_wilder(close, 14).iloc[-1] if len(close) else np.nan
            rows.append({'Ticker': t, 'MA20': float(ma20) if np.isfinite(ma20) else np.nan, 'MA50': float(ma50) if np.isfinite(ma50) else np.nan, 'RSI14': float(rsi14) if np.isfinite(rsi14) else np.nan, 'ATR14': float(atr14) if np.isfinite(atr14) else np.nan})
        ind = pd.DataFrame(rows)
    df = snapshot_df.merge(ind, on='Ticker', how='left') if ind is not None else snapshot_df.copy()
    if 'Price' not in df.columns or df['Price'].isna().all():
        df['Price'] = df['MA20']
    index_labels = {"VNINDEX", "VN30", "VN100"}
    df['Sector'] = df['Ticker'].map(lambda t: 'Index' if str(t).upper() in index_labels else mp.get(str(t).upper()))
    df['Sector'] = df['Sector'].astype(str).str.strip()
    rows = []
    for sector, g in df.groupby('Sector', dropna=False):
        m = compute_metrics(g); m['sector'] = str(sector); rows.append(m)
    overall = compute_metrics(df); overall['sector'] = 'Tất cả'; rows.append(overall)
    return pd.DataFrame(rows)[['sector','breadth_above_ma20_pct','breadth_above_ma50_pct','breadth_above_ma200_pct','avg_rsi14','avg_atr_pct','avg_dist_to_s1_pct','avg_dist_to_r1_pct']]
