from __future__ import annotations

"""Minimal slippage model calibration.

Inputs: fills.csv with columns [order_size, adv, spread_bps, vol_d, realized_bps]
Output: artifacts/calibration/slippage_model.json with linear coefficients
"""

import json
from pathlib import Path
import sys

import pandas as pd
from sklearn.linear_model import LinearRegression

Xcols = ["size_adv", "spread_bps", "vol_daily"]


def main(in_csv: str, out_json: str) -> None:
    df = pd.read_csv(in_csv).dropna()
    if df.empty:
        raise SystemExit("No data rows in fills.csv for slippage calibration")
    if "order_size" not in df.columns or "adv" not in df.columns:
        raise SystemExit("fills.csv must include order_size and adv columns")
    df["size_adv"] = df["order_size"] / (df["adv"].clip(lower=1.0))
    # backwards-compatible aliases
    if "vol_d" in df.columns and "vol_daily" not in df.columns:
        df["vol_daily"] = df["vol_d"]
    for col in ["spread_bps", "vol_daily", "realized_bps"]:
        if col not in df.columns:
            raise SystemExit(f"fills.csv missing required column: {col}")
    X = df[Xcols].values
    y = df["realized_bps"].values
    model = LinearRegression().fit(X, y)
    coefs = {
        "alpha_bps": float(model.intercept_),
        "beta_size": float(model.coef_[0]),
        "beta_spread": float(model.coef_[1]),
        "beta_vol": float(model.coef_[2]),
    }
    Path(out_json).parent.mkdir(parents=True, exist_ok=True)
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(coefs, f, indent=2)


if __name__ == "__main__":
    if len(sys.argv) < 3:
        raise SystemExit("Usage: calibrate_slippage_model.py <fills.csv> <artifacts/calibration/slippage_model.json>")
    main(sys.argv[1], sys.argv[2])

