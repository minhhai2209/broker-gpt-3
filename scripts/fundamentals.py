from __future__ import annotations

import re
from pathlib import Path
from typing import Dict, Iterable

import numpy as np
import pandas as pd

FUNDAMENTAL_MAP: Dict[str, Dict[str, str | float]] = {
    "Thu nhập trên mỗi cổ phần của 4 quý gần nhất (EPS)": {
        "column": "Fund_EPS_TTM",
    },
    "Tỷ suất lợi nhuận trên vốn chủ sở hữu bình quân (ROEA)": {
        "column": "Fund_ROE",
        "scale": 0.01,
    },
    "Chỉ số giá thị trường trên thu nhập (P/E)": {
        "column": "Fund_PE",
    },
    "Chỉ số giá thị trường trên giá trị sổ sách (P/B)": {
        "column": "Fund_PB",
    },
}

PERIOD_PATTERN = re.compile(r"Q(\d)/(\d{4})")


def _period_order(period: str) -> int:
    match = PERIOD_PATTERN.match(str(period).strip())
    if match:
        quarter = int(match.group(1))
        year = int(match.group(2))
        return year * 4 + quarter
    # Fallback: attempt to extract year at end
    tail = str(period).strip().split("/")[-1]
    return int(tail) if tail.isdigit() else 0


def load_latest_fundamentals(path: Path | str) -> pd.DataFrame:
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Missing fundamentals CSV: {path}")
    df = pd.read_csv(path)
    if df.empty:
        raise ValueError(f"Fundamentals CSV is empty: {path}")
    # Normalise ticker
    df["ticker"] = df["ticker"].astype(str).str.upper()
    df = df[df["metric"].isin(FUNDAMENTAL_MAP.keys())].copy()
    if df.empty:
        raise ValueError(f"Fundamentals CSV missing required metrics: {path}")
    df["period_order"] = df["period"].apply(_period_order)
    df = df.sort_values(["ticker", "metric", "period_order"])

    rows = []
    for ticker, sub in df.groupby("ticker"):
        rec = {"Ticker": ticker}
        for metric, spec in FUNDAMENTAL_MAP.items():
            m = sub[sub["metric"] == metric]
            if m.empty:
                continue
            series = m["value"].dropna()
            if series.empty:
                continue
            val = float(series.iloc[-1])
            scale = float(spec.get("scale", 1.0))
            rec[spec["column"]] = val * scale
        rows.append(rec)
    fund_df = pd.DataFrame(rows)
    if fund_df.empty:
        return fund_df

    # Derived metrics
    if "Fund_PE" in fund_df.columns:
        fund_df["Fund_EarningsYield"] = np.where(
            fund_df["Fund_PE"].notna() & (fund_df["Fund_PE"] > 0),
            1.0 / fund_df["Fund_PE"],
            np.nan,
        )
    return fund_df


def merge_fundamentals(metrics_df: pd.DataFrame, fundamentals_df: pd.DataFrame) -> pd.DataFrame:
    if fundamentals_df is None or fundamentals_df.empty:
        return metrics_df
    return metrics_df.merge(fundamentals_df, on="Ticker", how="left")
