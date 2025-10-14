from __future__ import annotations

"""Issuer events calendar loader.

Reads a simple CSV if present and exposes a helper to test event windows.
Schema (flexible): Ticker, Date, Type
If no feed is available, functions return empty/no-op.
"""

from datetime import date, timedelta
from pathlib import Path
from typing import Optional

import pandas as pd


def load_events(path: Path | None = None) -> pd.DataFrame:
    path = path or Path('data/events_calendar.csv')
    if not path.exists():
        return pd.DataFrame(columns=['Ticker','Date','Type'])
    df = pd.read_csv(path)
    if 'Ticker' not in df.columns or 'Date' not in df.columns:
        return pd.DataFrame(columns=['Ticker','Date','Type'])
    df = df[['Ticker','Date']].copy()
    df['Ticker'] = df['Ticker'].astype(str).str.upper()
    df['Date'] = pd.to_datetime(df['Date'], errors='coerce').dt.date
    df = df.dropna(subset=['Ticker','Date'])
    return df


def in_event_window(ticker: str, today: date, t_minus: int = 1, t_plus: int = 1, df: Optional[pd.DataFrame] = None) -> bool:
    if df is None or df.empty:
        return False
    t = str(ticker).upper()
    try:
        start = today - timedelta(days=int(t_minus))
        end = today + timedelta(days=int(t_plus))
    except Exception:
        start = today
        end = today
    sub = df[df['Ticker'] == t]
    if sub.empty:
        return False
    for d in sub['Date']:
        if d is None:
            continue
        if start <= d <= end:
            return True
    return False

