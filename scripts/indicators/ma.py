from __future__ import annotations

import pandas as pd


def ma(series: pd.Series, window: int) -> pd.Series:
    s = pd.to_numeric(series, errors="coerce")
    return s.rolling(window).mean()

