from __future__ import annotations

import pandas as pd


def _ema(series: pd.Series, span: int) -> pd.Series:
    return pd.to_numeric(series, errors="coerce").ewm(span=span, adjust=False).mean()


def macd_hist(close: pd.Series, fast: int = 12, slow: int = 26, signal: int = 9) -> pd.Series:
    c = pd.to_numeric(close, errors="coerce")
    macd = _ema(c, fast) - _ema(c, slow)
    sig = macd.ewm(span=signal, adjust=False).mean()
    hist = macd - sig
    return hist

