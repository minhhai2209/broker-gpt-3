"""
Precompute only indicators required by presets (used by Order Engine):
- MA10, MA20, MA50, MA200, ATR14, BB20Upper, BB20Lower
"""
from __future__ import annotations

from pathlib import Path
import json
import pandas as pd
from scripts.indicators import ma, atr_wilder, bollinger_bands


def compute_for_ticker(csv_path: Path) -> dict:
    df = pd.read_csv(csv_path)
    if df.empty:
        raise ValueError(f"Indicator source CSV is empty: {csv_path}")
    if not set({'close','high','low'}).issubset(df.columns):
        raise ValueError(f"Indicator source CSV missing columns in {csv_path}")
    df = df.copy()
    for c in ['high','low','close']:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors='coerce')
    out: dict = {}
    close = df['close']
    if len(close) >= 10:
        ma10 = ma(close, 10).iloc[-1]
        if pd.notna(ma10):
            out['MA10'] = float(ma10)
    if len(close) >= 20:
        ma20 = ma(close, 20).iloc[-1]
        if pd.notna(ma20):
            out['MA20'] = float(ma20)
    if len(close) >= 50:
        ma50 = ma(close, 50).iloc[-1]
        if pd.notna(ma50):
            out['MA50'] = float(ma50)
    if len(close) >= 200:
        ma200 = ma(close, 200).iloc[-1]
        if pd.notna(ma200):
            out['MA200'] = float(ma200)
    if len(close) >= 14:
        atr_series = atr_wilder(df['high'], df['low'], close, 14)
        if len(atr_series):
            atr_val = atr_series.iloc[-1]
            if pd.notna(atr_val):
                out['ATR14'] = float(atr_val)
    if len(close) >= 20:
        upper, mid, lower = bollinger_bands(close, window=20, n_std=2.0)
        if upper is not None:
            out['BB20Upper'] = float(upper)
        if lower is not None:
            out['BB20Lower'] = float(lower)
        if mid is not None:
            out['BB20Mid'] = float(mid)
    return out


def precompute_indicators(data_dir: str = 'out/data', out: str = 'out/precomputed_indicators.csv') -> None:
    data_dir = Path(data_dir)
    outp = Path(out)
    rows = []
    result = {}
    for p in sorted(data_dir.glob('*_daily.csv')):
        t = p.stem.replace('_daily','').upper()
        metrics = compute_for_ticker(p)
        result[t] = metrics
        r = {'Ticker': t}
        r.update(metrics)
        rows.append(r)
    if outp.suffix.lower() == '.csv':
        import csv as _csv
        cols = sorted({k for r in rows for k in r.keys()}, key=lambda x: (x!='Ticker', x))
        with outp.open('w', newline='', encoding='utf-8') as f:
            w = _csv.DictWriter(f, fieldnames=cols)
            w.writeheader()
            for r in rows:
                w.writerow(r)
    else:
        outp.write_text(json.dumps(result))
    print(f"Wrote {out}")


def precompute_indicators_from_history_df(ph_df: pd.DataFrame) -> pd.DataFrame:
    """Compute MA10/20/50/200, ATR14, BB20 bands from a merged prices_history DataFrame.
    Returns a DataFrame per ticker with the latest values.
    """
    import numpy as np
    rows = []
    if ph_df is None or ph_df.empty:
        raise ValueError('prices_history_df must contain data for precomputation')
    ph = ph_df.copy()
    ph['Ticker'] = ph['Ticker'].astype(str).str.upper()
    ph['Date'] = pd.to_datetime(ph['Date'], errors='coerce')
    ph = ph.dropna(subset=['Date']).sort_values(['Ticker','Date'])
    for t, g in ph.groupby('Ticker'):
        close = pd.to_numeric(g['Close'], errors='coerce')
        high = pd.to_numeric(g['High'], errors='coerce')
        low = pd.to_numeric(g['Low'], errors='coerce')
        ma10_series = ma(close, 10) if len(close) >= 10 else pd.Series(dtype=float)
        ma20_series = ma(close, 20) if len(close) >= 20 else pd.Series(dtype=float)
        ma50_series = ma(close, 50) if len(close) >= 50 else pd.Series(dtype=float)
        ma200_series = ma(close, 200) if len(close) >= 200 else pd.Series(dtype=float)
        ma10 = float(ma10_series.iloc[-1]) if len(ma10_series) and pd.notna(ma10_series.iloc[-1]) else np.nan
        ma20 = float(ma20_series.iloc[-1]) if len(ma20_series) and pd.notna(ma20_series.iloc[-1]) else np.nan
        ma50 = float(ma50_series.iloc[-1]) if len(ma50_series) and pd.notna(ma50_series.iloc[-1]) else np.nan
        ma200 = float(ma200_series.iloc[-1]) if len(ma200_series) and pd.notna(ma200_series.iloc[-1]) else np.nan
        atr = np.nan
        if len(close) >= 14:
            atr_series = atr_wilder(high, low, close, 14)
            if len(atr_series):
                atr_val = atr_series.iloc[-1]
                if pd.notna(atr_val):
                    atr = float(atr_val)
        upper = mid = lower = None
        if len(close) >= 20:
            upper, mid, lower = bollinger_bands(close, window=20, n_std=2.0)
        rows.append({'Ticker': t, 'MA10': ma10, 'MA20': ma20, 'MA50': ma50, 'MA200': ma200, 'ATR14': atr, 'BB20Upper': upper, 'BB20Lower': lower, 'BB20Mid': mid})
    return pd.DataFrame(rows)
