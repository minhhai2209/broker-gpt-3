from __future__ import annotations

from __future__ import annotations

"""Volatility helpers shared across calibrators and runtime pipeline."""

from typing import Iterable, Tuple

import numpy as np
import pandas as pd


def garman_klass_sigma(df: pd.DataFrame) -> pd.Series:
    """Compute Garman–Klass volatility (daily sigma) from OHLC data.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame with columns ``High``/``Low``/``Close`` and optionally ``Open``.
        Missing ``Open`` values fall back to previous ``Close``. All prices are
        expected to be positive (in any consistent unit).

    Returns
    -------
    pd.Series
        Daily sigma estimates aligned with ``df`` index. Rows lacking sufficient
        data yield ``NaN``.
    """

    if df.empty:
        return pd.Series(dtype=float)
    cols = {c: pd.to_numeric(df.get(c), errors="coerce") for c in ("High", "Low", "Close", "Open")}
    high = cols["High"]
    low = cols["Low"]
    close = cols["Close"]
    open_ = cols["Open"]
    if open_ is None or open_.isna().all():
        open_ = close.shift(1)
    else:
        open_ = open_.fillna(close.shift(1))
    with np.errstate(divide="ignore", invalid="ignore"):
        log_hl = np.log(high / low)
        log_co = np.log(close / open_)
        term1 = 0.5 * (log_hl ** 2)
        term2 = (2.0 * np.log(2.0) - 1.0) * (log_co ** 2)
        sigma2 = term1 - term2
    sigma2 = sigma2.replace([np.inf, -np.inf], np.nan)
    sigma2 = sigma2.clip(lower=0.0)
    sigma = np.sqrt(sigma2)
    return sigma


def percentile_thresholds(series: pd.Series, percentiles: Iterable[float]) -> Tuple[float, ...]:
    """Return tuple of percentile values for a clean numeric series.

    Percentiles should be specified in [0, 1]. NaNs are ignored.
    """

    arr = pd.to_numeric(series, errors="coerce").replace([np.inf, -np.inf], np.nan).dropna()
    if arr.empty:
        return tuple(float("nan") for _ in percentiles)
    pct_vals = []
    for p in percentiles:
        pct = max(0.0, min(1.0, float(p))) * 100.0
        pct_vals.append(float(np.nanpercentile(arr.to_numpy(), pct)))
    return tuple(pct_vals)
