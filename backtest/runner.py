"""Command line interface to replay historical windows."""

from __future__ import annotations

import argparse
import json
from datetime import date
from pathlib import Path
from typing import Any, Dict, List, Optional

import pandas as pd

from .engine_wrapper import EngineWrapper
from .fill_sim import execute_orders
from .loader import BacktestLoader
from .metrics import MetricsTracker
from .portfolio import FeeConfig, Portfolio
from .utils import (
    ensure_directory,
    load_config,
    normalize_liquidity_model,
    normalize_slippage_model,
    seed_everything,
)


class BacktestRunner:
    def __init__(self, config: Dict[str, Any], *, out_dir: Path, start: date, end: date):
        self.config = config
        self.out_dir = ensure_directory(out_dir)
        self.start = start
        self.end = end
        self.loader = BacktestLoader.from_dict(config)
        self.run_id = str(config.get("run_id") or self.out_dir.name)
        self.params_hash = config.get("params_hash")
        general_cfg = config.get("config") or {}
        self.initial_cash = float(general_cfg.get("initial_cash", 0.0) or 0.0)
        self.default_ttl = int(general_cfg.get("ttl_days", 3) or 3)
        fees_cfg = general_cfg.get("fees", {})
        self.fee_config = FeeConfig.from_dict(fees_cfg)
        self.liquidity_model = normalize_liquidity_model(general_cfg.get("liquidity_model", {"type": "touch_full"}))
        self.slippage_model = normalize_slippage_model(general_cfg.get("slippage_model", {"type": "none"}))
        self.horizons = list(general_cfg.get("horizon_eval_days", [3, 5, 10]))
        self.quality_gates = dict(general_cfg.get("quality_gates", {}))
        self.objective = str(general_cfg.get("objective", "sharpe"))
        flags = dict(general_cfg.get("engine_flags", {}))
        self.engine = EngineWrapper(flags)
        self.metrics = MetricsTracker(self.horizons, self.quality_gates)
        seed = int(general_cfg.get("seed", 7) or 7)
        seed_everything(seed)
        self.logs: List[Dict[str, Any]] = []

    def run(self):
        trading_days = self.loader.get_trading_days(self.start, self.end)
        portfolio = Portfolio(
            initial_cash=self.initial_cash,
            positions=self.loader.initial_portfolio.to_dict("records") if not self.loader.initial_portfolio.empty else None,
            fee_config=self.fee_config,
        )
        replay_dir = ensure_directory(self.out_dir / "replay_day")
        metrics_dir = ensure_directory(self.out_dir / "metrics")
        summary_dir = ensure_directory(self.out_dir / "summary")
        logs_dir = ensure_directory(self.out_dir / "logs")
        for day in trading_days:
            context = self.loader.get_day_context(day)
            orders = self.engine.build_orders_for_day(day, context)
            self.metrics.register_orders([o.to_dict() for o in orders])
            self._log_orders(day, orders)
            portfolio.submit_orders(day, orders, default_ttl=self.default_ttl)
            pending_orders = portfolio.pending_for_day()
            day_prices = self.loader.get_day_prices(day)
            fills, _ = execute_orders(
                day,
                pending_orders,
                day_prices,
                self.liquidity_model,
                self.slippage_model,
            )
            self._log_fills(day, fills)
            portfolio.apply_fills(day, fills)
            expired = portfolio.expire_orders(day)
            self._log_expired(day, expired)
            state = portfolio.mark_to_market(day, day_prices)
            self.metrics.update(day, state, fills)
            self._log_nav(day, state.nav)
            self._write_day_artifacts(replay_dir, day, orders, fills, state)
        self.metrics.add_trades(portfolio.trades_frame())
        bundle = self.metrics.finalize()
        self._write_summary(summary_dir, bundle)
        self._write_metrics(metrics_dir, bundle)
        self._write_logs(logs_dir)
        return bundle

    # ------------------------------------------------------------------
    # Logging & artifacts
    # ------------------------------------------------------------------

    def _write_day_artifacts(self, out_dir: Path, day: date, orders, fills, state) -> None:
        ts = day.strftime("%Y%m%d")
        orders_df = pd.DataFrame([o.to_dict() for o in orders])
        fills_df = pd.DataFrame([
            {
                "symbol": f.order.symbol,
                "qty": f.executed_qty,
                "price": f.executed_price,
                "status": f.status,
            }
            for f in fills
        ])
        state_df = pd.DataFrame(
            [
                {
                    "symbol": pos.symbol,
                    "qty": pos.quantity,
                    "avg_cost": pos.avg_cost,
                }
                for pos in state.positions.values()
            ]
        )
        orders_df.to_csv(out_dir / f"orders_{ts}.csv", index=False)
        fills_df.to_csv(out_dir / f"fills_{ts}.csv", index=False)
        state_df.to_csv(out_dir / f"portfolio_{ts}.csv", index=False)

    def _write_summary(self, out_dir: Path, bundle) -> None:
        summary_path = out_dir / "summary.json"
        summary_path.write_text(json.dumps(bundle.summary, indent=2), encoding="utf-8")

    def _write_metrics(self, out_dir: Path, bundle) -> None:
        bundle.daily.to_csv(out_dir / "daily.csv", index=False)
        bundle.trades.to_csv(out_dir / "trades.csv", index=False)
        bundle.fills.to_csv(out_dir / "fills.csv", index=False)

    def _write_logs(self, out_dir: Path) -> None:
        path = out_dir / "events.ndjson"
        lines = [json.dumps(event) for event in self.logs]
        path.write_text("\n".join(lines), encoding="utf-8")

    def _log_orders(self, day: date, orders) -> None:
        for idx, order in enumerate(orders):
            self.logs.append(
                {
                    "ts": day.isoformat(),
                    "date": day.isoformat(),
                    "symbol": order.symbol,
                    "event": "ORDER_CREATED",
                    "order_id": f"{day:%Y%m%d}_{idx}",
                    "qty": order.qty,
                    "limit_price": order.limit_price,
                    "notes": order.meta,
                    "run_id": self.run_id,
                    "params_hash": self.params_hash,
                }
            )

    def _log_fills(self, day: date, fills) -> None:
        for fill in fills:
            self.logs.append(
                {
                    "ts": day.isoformat(),
                    "date": day.isoformat(),
                    "symbol": fill.order.symbol,
                    "event": "FILLED",
                    "qty": fill.executed_qty,
                    "exec_price": fill.executed_price,
                    "status": fill.status,
                    "run_id": self.run_id,
                    "params_hash": self.params_hash,
                }
            )

    def _log_expired(self, day: date, expired) -> None:
        for pending in expired:
            self.logs.append(
                {
                    "ts": day.isoformat(),
                    "date": day.isoformat(),
                    "symbol": pending.order.symbol,
                    "event": "TTL_EXPIRED",
                    "qty": pending.order.qty,
                    "run_id": self.run_id,
                    "params_hash": self.params_hash,
                }
            )

    def _log_nav(self, day: date, nav: float) -> None:
        self.logs.append(
            {
                "ts": day.isoformat(),
                "date": day.isoformat(),
                "event": "PNL_SNAPSHOT",
                "nav": nav,
                "run_id": self.run_id,
                "params_hash": self.params_hash,
            }
        )


def parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Historical replay runner")
    parser.add_argument("--start", required=True, help="Start date (YYYY-MM-DD)")
    parser.add_argument("--end", required=True, help="End date (YYYY-MM-DD)")
    parser.add_argument("--config", required=True, help="Path to configuration file")
    parser.add_argument("--out", required=True, help="Output directory")
    return parser.parse_args(argv)


def main(argv: Optional[List[str]] = None) -> int:
    args = parse_args(argv)
    config = load_config(args.config)
    runner = BacktestRunner(
        config,
        out_dir=Path(args.out),
        start=pd.Timestamp(args.start).date(),
        end=pd.Timestamp(args.end).date(),
    )
    runner.run()
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
