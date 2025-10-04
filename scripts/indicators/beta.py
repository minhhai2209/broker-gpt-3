from __future__ import annotations

import pandas as pd
import numpy as np


def beta_rolling(ret: pd.Series, mkt_ret: pd.Series, window: int = 60) -> float | None:
    df = pd.concat([
        pd.to_numeric(ret, errors="coerce").rename("ret"),
        pd.to_numeric(mkt_ret, errors="coerce").rename("mkt")
    ], axis=1).dropna()
    if df.empty:
        return None
    tail = df.tail(window)
    if tail.empty:
        return None
    cov_matrix = np.cov(tail["ret"], tail["mkt"])
    cov = float(cov_matrix[0, 1])
    var_m = float(np.var(tail["mkt"]))
    if var_m == 0:
        return None
    return cov / var_m
