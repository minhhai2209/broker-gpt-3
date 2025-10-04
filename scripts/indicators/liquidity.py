from __future__ import annotations

import pandas as pd


def avg_turnover_k(close: pd.Series, volume: pd.Series, window: int = 20) -> pd.Series:
    c = pd.to_numeric(close, errors="coerce")
    v = pd.to_numeric(volume, errors="coerce")
    turn = c * v
    return turn.rolling(window).mean()

