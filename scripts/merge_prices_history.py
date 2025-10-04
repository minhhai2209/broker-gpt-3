"""
Merge per-ticker daily CSVs in out/data into a single out/prices_history.csv
with columns: Date, Ticker, Open, High, Low, Close, Volume, t.
"""
from __future__ import annotations

from pathlib import Path
import pandas as pd
from datetime import timezone, timedelta

VN_TZ = timezone(timedelta(hours=7))


def merge_prices_history(data_dir: str = 'out/data', out_path: str = 'out/prices_history.csv') -> None:
    data_dir = Path(data_dir)
    out_path = Path(out_path)
    rows = []
    if not data_dir.exists():
        raise FileNotFoundError('Missing data_dir: out/data')
    for p in sorted(data_dir.glob('*_daily.csv')):
        ticker = p.stem.replace('_daily', '').upper()
        df = pd.read_csv(p)
        if df.empty:
            raise ValueError(f"Empty daily CSV: {p}")
        # Ensure required columns exist
        for c in ['t', 'open', 'high', 'low', 'close', 'volume']:
            if c not in df.columns:
                df[c] = pd.NA
        # Build Date column from date_vn if present, else convert t
        if 'date_vn' in df.columns and df['date_vn'].notna().any():
            date = df['date_vn'].astype(str)
        else:
            ts = pd.to_datetime(df['t'], unit='s', utc=True).dt.tz_convert(VN_TZ)
            date = ts.dt.strftime('%Y-%m-%d')
        out = pd.DataFrame({
            'Date': date,
            'Ticker': ticker,
            'Open': df['open'],
            'High': df['high'],
            'Low': df['low'],
            'Close': df['close'],
            'Volume': df['volume'],
            't': df['t'],
        })
        rows.append(out)
    if not rows:
        raise FileNotFoundError('No *_daily.csv files found in out/data')
    all_df = pd.concat(rows, ignore_index=True)
    # Sort by Ticker then time
    if 't' in all_df.columns:
        all_df = all_df.sort_values(['Ticker', 't']).reset_index(drop=True)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    all_df.to_csv(out_path, index=False)
    print(f'Wrote {out_path}')

