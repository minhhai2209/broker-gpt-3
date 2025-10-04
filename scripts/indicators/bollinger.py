from __future__ import annotations

import pandas as pd
import numpy as np


def bollinger_bands(close: pd.Series, window: int = 20, n_std: float = 2.0) -> tuple[float | None, float | None, float | None]:
    c = pd.to_numeric(close, errors="coerce")
    ma = c.rolling(window).mean()
    std = c.rolling(window).std(ddof=0)
    upper = ma + n_std * std
    lower = ma - n_std * std
    u = float(upper.iloc[-1]) if len(upper) and pd.notna(upper.iloc[-1]) else None
    m = float(ma.iloc[-1]) if len(ma) and pd.notna(ma.iloc[-1]) else None
    l = float(lower.iloc[-1]) if len(lower) and pd.notna(lower.iloc[-1]) else None
    return u, m, l
