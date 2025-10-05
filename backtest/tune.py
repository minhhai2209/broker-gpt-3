"""Parameter tuning workflows for backtests."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional

import pandas as pd

from .metrics import MetricsBundle, score_objective
from .runner import BacktestRunner
from .utils import (
    ConfigError,
    expand_params_grid,
    load_config,
    params_hash,
)


def apply_params(base_config: Mapping[str, Any], params: Mapping[str, Any]) -> Dict[str, Any]:
    conf = json.loads(json.dumps(base_config))
    config_section = conf.setdefault("config", {})
    for key, value in params.items():
        parts = key.split(".")
        node = config_section
        for part in parts[:-1]:
            node = node.setdefault(part, {})
        node[parts[-1]] = value
    return conf


def run_window(config: Mapping[str, Any], start: pd.Timestamp, end: pd.Timestamp, out_dir: Path) -> MetricsBundle:
    runner = BacktestRunner(dict(config), out_dir=out_dir, start=start.date(), end=end.date())
    bundle = runner.run()
    return bundle


def grid_search(
    base_config: Mapping[str, Any],
    grid_params: Mapping[str, Iterable[Any]],
    train_start: pd.Timestamp,
    train_end: pd.Timestamp,
    val_start: pd.Timestamp,
    val_end: pd.Timestamp,
    out_dir: Path,
    *,
    objective: str,
    top_k: int = 3,
) -> Dict[str, Any]:
    combos = expand_params_grid(grid_params)
    leaderboard: List[Dict[str, Any]] = []
    for params in combos:
        cfg = apply_params(base_config, params)
        phash = params_hash(params)
        run_out = out_dir / phash
        run_out.mkdir(parents=True, exist_ok=True)
        train_bundle = run_window(cfg, train_start, train_end, run_out / "train")
        score_train = score_objective(train_bundle.summary, objective)
        if not train_bundle.gates_passed:
            score_train = float("-inf")
        leaderboard.append(
            {
                "params_hash": phash,
                "params": json.dumps(params, sort_keys=True),
                "score_train": score_train,
                "train_total_return": train_bundle.summary.get("total_return", 0.0),
                "train_sharpe": train_bundle.summary.get("sharpe", 0.0),
                "train_fill_rate": train_bundle.summary.get("fill_rate", 0.0),
            }
        )
    leaderboard_df = pd.DataFrame(leaderboard).sort_values(by="score_train", ascending=False)
    top = leaderboard_df.head(top_k)
    best_overall = None
    best_score = float("-inf")
    validation_rows: List[Dict[str, Any]] = []
    for _, row in top.iterrows():
        params = json.loads(row["params"])
        phash = row["params_hash"]
        cfg = apply_params(base_config, params)
        run_out = out_dir / phash
        val_bundle = run_window(cfg, val_start, val_end, run_out / "val")
        score_val = score_objective(val_bundle.summary, objective)
        if not val_bundle.gates_passed:
            score_val = float("-inf")
        validation_rows.append(
            {
                "params_hash": phash,
                "score_val": score_val,
                "val_total_return": val_bundle.summary.get("total_return", 0.0),
                "val_sharpe": val_bundle.summary.get("sharpe", 0.0),
                "val_fill_rate": val_bundle.summary.get("fill_rate", 0.0),
            }
        )
        if score_val > best_score:
            best_score = score_val
            best_overall = {
                "params": params,
                "params_hash": phash,
                "score_val": score_val,
            }
    leaderboard_full = leaderboard_df.merge(pd.DataFrame(validation_rows), on="params_hash", how="left")
    return {
        "leaderboard": leaderboard_full,
        "best": best_overall,
    }


def walk_forward(
    base_config: Mapping[str, Any],
    grid_params: Mapping[str, Iterable[Any]],
    start: pd.Timestamp,
    end: pd.Timestamp,
    train_days: int,
    val_days: int,
    out_dir: Path,
    *,
    objective: str,
) -> pd.DataFrame:
    window_start = start
    records: List[Dict[str, Any]] = []
    while window_start < end:
        train_end = window_start + pd.Timedelta(days=train_days - 1)
        val_end = train_end + pd.Timedelta(days=val_days)
        if val_end > end:
            break
        result = grid_search(
            base_config,
            grid_params,
            window_start,
            train_end,
            train_end + pd.Timedelta(days=1),
            val_end,
            out_dir / f"wf_{window_start:%Y%m%d}",
            objective=objective,
            top_k=1,
        )
        best = result["best"] or {}
        records.append(
            {
                "train_start": window_start.date(),
                "train_end": train_end.date(),
                "val_start": (train_end + pd.Timedelta(days=1)).date(),
                "val_end": val_end.date(),
                "params_hash": best.get("params_hash"),
                "score_val": best.get("score_val"),
            }
        )
        window_start = window_start + pd.Timedelta(days=val_days)
    return pd.DataFrame.from_records(records)


def parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Grid search and walk-forward tuning")
    parser.add_argument("--grid", required=True, help="Parameter grid YAML path")
    parser.add_argument("--base-config", required=True, help="Base replay configuration path")
    parser.add_argument("--out", required=True, help="Output directory for tuning results")
    parser.add_argument("--train-start")
    parser.add_argument("--train-end")
    parser.add_argument("--val-start")
    parser.add_argument("--val-end")
    parser.add_argument("--wf-start")
    parser.add_argument("--wf-end")
    parser.add_argument("--wf-train-days", type=int)
    parser.add_argument("--wf-val-days", type=int)
    parser.add_argument("--top-k", type=int, default=3)
    return parser.parse_args(argv)


def main(argv: Optional[List[str]] = None) -> int:
    args = parse_args(argv)
    base_config = load_config(args.base_config)
    grid = load_config(args.grid)
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    objective = str((base_config.get("config") or {}).get("objective", "sharpe"))
    summary_dir = out_dir / "summary"
    summary_dir.mkdir(parents=True, exist_ok=True)
    if args.train_start and args.train_end and args.val_start and args.val_end:
        result = grid_search(
            base_config,
            grid,
            pd.Timestamp(args.train_start),
            pd.Timestamp(args.train_end),
            pd.Timestamp(args.val_start),
            pd.Timestamp(args.val_end),
            out_dir,
            objective=objective,
            top_k=args.top_k,
        )
        leaderboard = result["leaderboard"]
        best = result["best"] or {}
        leaderboard.to_csv(summary_dir / "leaderboard.csv", index=False)
        (summary_dir / "best_params.json").write_text(json.dumps(best, indent=2), encoding="utf-8")
    if args.wf_start and args.wf_end and args.wf_train_days and args.wf_val_days:
        report = walk_forward(
            base_config,
            grid,
            pd.Timestamp(args.wf_start),
            pd.Timestamp(args.wf_end),
            args.wf_train_days,
            args.wf_val_days,
            out_dir,
            objective=objective,
        )
        report.to_csv(summary_dir / "walk_forward_report.csv", index=False)
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
