#!/usr/bin/env python3
"""Score tickers using the latest ML calibrator model."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

from .ml_config import load_config
from .ml_runtime import (
    append_ml_log,
    build_feature_panel,
    compute_regime_adjustments,
    extract_risk_on_probability,
    feature_columns,
    latest_feature_slice,
    load_input_frames,
    load_model_bundle,
    predict_proba,
    write_empty_patch,
    write_patch,
    write_predictions,
)


def score(cfg_path: Path | None = None) -> None:
    cfg = load_config(cfg_path)
    frames = load_input_frames(cfg)
    horizon = int(cfg.raw.get("train", {}).get("horizon_days", 3) or 3)
    feature_panel = build_feature_panel(frames["prices"], horizon)
    latest_date, latest_slice = latest_feature_slice(feature_panel)
    features = list(feature_columns())
    latest_slice = latest_slice.dropna(subset=features)
    if latest_slice.empty:
        patch = write_empty_patch(cfg, "no_features")
        append_ml_log({"event": "score", "status": "empty", "patch_gate_size": 0})
        return

    bundle = load_model_bundle()
    preds = predict_proba(bundle, latest_slice)
    latest_slice = latest_slice.assign(p_succ=np.clip(preds, 0.0, 1.0))
    preds_df = latest_slice[["ticker", "p_succ"]]
    write_predictions(preds_df)

    risk_on_prob = extract_risk_on_probability(frames["session"])
    budget, limits, bias = compute_regime_adjustments(frames["session"], frames["sector_strength"], cfg, risk_on_prob)
    patch = write_patch(preds_df, cfg, budget, limits, bias, risk_on_prob, bundle.metadata)
    append_ml_log(
        {
            "event": "score",
            "model": bundle.metadata,
            "preview_date": latest_date.isoformat(),
            "patch_gate_size": len(patch.get("gate", {})),
            "risk_on_prob": risk_on_prob,
        }
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Score tickers with ML calibrator")
    parser.add_argument("--config", type=Path, default=None, help="Path to ml_config.yaml")
    args = parser.parse_args()
    try:
        score(args.config)
    except FileNotFoundError as exc:
        cfg = load_config(args.config)
        write_empty_patch(cfg, f"missing_input:{exc}")
        append_ml_log({"event": "score_error", "error": str(exc)})
    except Exception as exc:
        cfg = load_config(args.config)
        write_empty_patch(cfg, "error")
        append_ml_log({"event": "score_error", "error": str(exc)})


if __name__ == "__main__":
    main()
