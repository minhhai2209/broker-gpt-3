from __future__ import annotations

import pandas as pd


def rsi_wilder(close: pd.Series, period: int = 14) -> pd.Series:
    c = pd.to_numeric(close, errors="coerce")
    delta = c.diff()
    up = delta.clip(lower=0)
    down = -delta.clip(upper=0)
    roll_up = up.ewm(alpha=1/period, adjust=False).mean()
    roll_down = down.ewm(alpha=1/period, adjust=False).mean()
    rs = roll_up / roll_down.replace(0, pd.NA)
    rsi = 100 - (100 / (1 + rs))
    return rsi

