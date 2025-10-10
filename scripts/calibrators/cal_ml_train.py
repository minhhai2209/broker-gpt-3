#!/usr/bin/env python3
"""Train the ML calibrator model and publish daily preview scores."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
import pandas as pd

from .ml_config import load_config
from .ml_runtime import (
    append_ml_log,
    attach_labels,
    build_feature_panel,
    compute_regime_adjustments,
    extract_risk_on_probability,
    feature_columns,
    latest_feature_slice,
    load_input_frames,
    save_model_bundle,
    write_metrics,
    write_patch,
    write_predictions,
)


def _roc_auc(y_true: np.ndarray, y_score: np.ndarray) -> float:
    order = np.argsort(y_score)
    y_true = y_true[order]
    y_score = y_score[order]
    cum_pos = np.cumsum(y_true[::-1])[::-1]
    cum_neg = np.cumsum(1 - y_true[::-1])[::-1]
    tp = cum_pos[::-1]
    fp = cum_neg[::-1]
    tp = np.insert(tp, 0, 0)
    fp = np.insert(fp, 0, 0)
    tpr = tp / (tp[-1] if tp[-1] > 0 else 1.0)
    fpr = fp / (fp[-1] if fp[-1] > 0 else 1.0)
    auc = np.trapz(tpr, fpr)
    return float(max(0.0, min(1.0, auc)))


def _precision_at_k(df: pd.DataFrame, k: int) -> float:
    if df.empty or k <= 0:
        return 0.0
    top = df.sort_values("p_succ", ascending=False).head(k)
    if top.empty:
        return 0.0
    return float(top["label"].mean())


def _spearman_ic(df: pd.DataFrame) -> float:
    if df.empty:
        return 0.0
    x = df["p_succ"].to_numpy()
    y = df["excess_future"].to_numpy()
    if np.allclose(x.std(), 0) or np.allclose(y.std(), 0):
        return 0.0
    rx = pd.Series(x).rank().to_numpy()
    ry = pd.Series(y).rank().to_numpy()
    return float(np.corrcoef(rx, ry)[0, 1])


def _train_model(train_df: pd.DataFrame, features: List[str], seed: int) -> Any:
    import lightgbm as lgb  # type: ignore

    params: Dict[str, Any] = {
        "objective": "binary",
        "metric": ["auc"],
        "verbosity": -1,
        "boosting_type": "gbdt",
        "seed": seed,
    }
    params.update(train_df.attrs.get("model_params", {}))
    num_boost_round = int(params.pop("n_estimators", 320))
    dataset = lgb.Dataset(train_df[features], label=train_df["label"], free_raw_data=False)

    # Simple time-based split for validation
    unique_dates = sorted(train_df["date"].unique())
    valid_set = None
    valid_dates = []
    if len(unique_dates) > 40:
        split_idx = int(len(unique_dates) * 0.8)
        valid_dates = unique_dates[split_idx:]
        train_dates = unique_dates[:split_idx]
        train_subset = train_df[train_df["date"].isin(train_dates)]
        valid_subset = train_df[train_df["date"].isin(valid_dates)]
        dataset = lgb.Dataset(train_subset[features], label=train_subset["label"], free_raw_data=False)
        if not valid_subset.empty:
            valid_set = lgb.Dataset(valid_subset[features], label=valid_subset["label"], reference=dataset, free_raw_data=False)

    valid_sets = [dataset]
    if valid_set is not None:
        valid_sets.append(valid_set)
    model = lgb.train(
        params,
        dataset,
        num_boost_round=num_boost_round,
        valid_sets=valid_sets,
        verbose_eval=False,
    )
    model.best_iteration = model.best_iteration or model.current_iteration()
    model.__dict__["_valid_dates"] = valid_dates  # type: ignore[attr-defined]
    return model


def main() -> None:
    parser = argparse.ArgumentParser(description="Train ML calibrator")
    parser.add_argument("--config", type=Path, default=None, help="Path to ml_config.yaml")
    args = parser.parse_args()
    cfg = load_config(args.config)
    frames = load_input_frames(cfg)
    horizon = int(cfg.raw.get("train", {}).get("horizon_days", 3) or 3)
    feature_panel = build_feature_panel(frames["prices"], horizon)
    labelled = attach_labels(feature_panel, cfg, horizon)
    if labelled.empty:
        raise SystemExit("Insufficient training data for ML calibrator")

    features = list(feature_columns())
    missing = [col for col in features if col not in labelled.columns]
    if missing:
        raise SystemExit(f"Training data missing feature columns: {', '.join(missing)}")
    labelled = labelled.dropna(subset=features + ["label"])
    seed = int(cfg.raw.get("train", {}).get("seed", 42) or 42)
    params = dict(cfg.raw.get("model", {}).get("params", {}) or {})
    labelled.attrs["model_params"] = params

    model = _train_model(labelled, features, seed)
    preds = model.predict(labelled[features])  # type: ignore[attr-defined]
    labelled = labelled.assign(p_succ=np.clip(preds, 0.0, 1.0))

    y_true = labelled["label"].to_numpy(dtype=float)
    auc = _roc_auc(y_true, labelled["p_succ"].to_numpy(dtype=float))
    precision20 = _precision_at_k(labelled, 20)
    ic = _spearman_ic(labelled)
    topk_return = float(
        labelled.sort_values("p_succ", ascending=False)
        .head(20)
        .get("excess_future", pd.Series(dtype=float))
        .mean()
    )
    metrics = {
        "auc": auc,
        "precision_at_20": precision20,
        "ic_spearman": ic,
        "train_rows": int(labelled.shape[0]),
        "topk_excess_return": topk_return,
    }
    write_metrics(metrics)

    bundle_meta = {
        "name": str(cfg.raw.get("model", {}).get("type", "lgbm")),
        "version": "1.0.0",
        "seed": seed,
        "features": features,
        "trained_at": datetime.now(timezone.utc).isoformat(),
        "metrics": metrics,
    }
    model_dir = save_model_bundle(model, bundle_meta)

    latest_date, latest_slice = latest_feature_slice(feature_panel)
    latest_slice = latest_slice.dropna(subset=features)
    latest_preds = np.clip(model.predict(latest_slice[features]), 0.0, 1.0)  # type: ignore[attr-defined]
    latest_slice = latest_slice.assign(p_succ=latest_preds)
    latest_slice = latest_slice[["ticker", "p_succ"]]
    write_predictions(latest_slice)

    risk_on_prob = extract_risk_on_probability(frames["session"])
    budget, limits, bias = compute_regime_adjustments(frames["session"], frames["sector_strength"], cfg, risk_on_prob)
    patch = write_patch(
        latest_slice,
        cfg,
        budget,
        limits,
        bias,
        risk_on_prob,
        {"path": str(model_dir), **bundle_meta},
    )

    append_ml_log(
        {
            "event": "train",
            "model_dir": str(model_dir),
            "metrics": metrics,
            "risk_on_prob": risk_on_prob,
            "preview_date": latest_date.isoformat(),
            "patch_gate_size": len(patch.get("gate", {})),
        }
    )


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        append_ml_log({"event": "train_error", "error": str(exc)})
        raise
