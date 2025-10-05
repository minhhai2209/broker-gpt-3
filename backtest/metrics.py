"""Metric computations for backtest runs."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date
from typing import Any, Dict, Iterable, List, Mapping, Optional

import numpy as np
import pandas as pd

from .fill_sim import FillResult
from .portfolio import PortfolioState
from .utils import ConfigError, cumulative_returns, max_drawdown, sharpe_ratio


@dataclass
class MetricsBundle:
    summary: Dict[str, float]
    daily: pd.DataFrame
    trades: pd.DataFrame
    fills: pd.DataFrame
    gates_passed: bool


@dataclass
class MetricsTracker:
    horizons: List[int]
    quality_gates: Mapping[str, float]
    returns: List[float] = field(default_factory=list)
    nav: List[float] = field(default_factory=list)
    dates: List[date] = field(default_factory=list)
    fills: List[Dict[str, object]] = field(default_factory=list)
    trades: List[Dict[str, object]] = field(default_factory=list)
    total_fill_qty: float = 0.0
    total_order_qty: float = 0.0

    def register_orders(self, orders: Iterable[Dict[str, Any]]) -> None:
        for order in orders:
            qty = float(order.get("qty", 0))
            if qty > 0:
                self.total_order_qty += qty

    def update(self, day: date, state: PortfolioState, fills: Iterable[FillResult]) -> None:
        if self.nav:
            prev_nav = self.nav[-1]
            daily_return = (state.nav - prev_nav) / prev_nav if prev_nav != 0 else 0.0
        else:
            daily_return = 0.0
        self.nav.append(state.nav)
        self.dates.append(day)
        self.returns.append(daily_return)
        for fill in fills:
            record = {
                "date": day,
                "symbol": fill.order.symbol,
                "side": fill.order.side,
                "qty": fill.executed_qty,
                "price": fill.executed_price,
                "status": fill.status,
            }
            self.fills.append(record)
            self.total_fill_qty += fill.executed_qty
        # Append trades from state? state doesn't hold; they will be provided separately

    def add_trades(self, trades: pd.DataFrame) -> None:
        if trades is not None and not trades.empty:
            self.trades.extend(trades.to_dict("records"))

    def finalize(self) -> MetricsBundle:
        nav_series = pd.Series(self.nav, index=pd.Index(self.dates, name="date"))
        returns_series = pd.Series(self.returns, index=nav_series.index)
        total_return = cumulative_returns(returns_series)
        sharpe = sharpe_ratio(returns_series)
        drawdown = max_drawdown(nav_series)
        vol = float(returns_series.std(ddof=0) * np.sqrt(252)) if not returns_series.empty else 0.0
        trades_df = pd.DataFrame.from_records(self.trades)
        fills_df = pd.DataFrame.from_records(self.fills)
        hit_rate = 0.0
        if not trades_df.empty:
            wins = trades_df["realized_pnl"] > 0
            if wins.any():
                hit_rate = float(wins.mean())
        fill_rate = (self.total_fill_qty / self.total_order_qty) if self.total_order_qty else 0.0
        summary = {
            "total_return": total_return,
            "sharpe": sharpe,
            "volatility": vol,
            "max_drawdown": drawdown,
            "hit_rate": hit_rate,
            "fill_rate": fill_rate,
            "trades": float(len(trades_df)),
        }
        gates_passed = self._check_quality(summary)
        daily_df = pd.DataFrame({"date": self.dates, "nav": self.nav, "return": self.returns})
        return MetricsBundle(
            summary=summary,
            daily=daily_df,
            trades=trades_df,
            fills=fills_df,
            gates_passed=gates_passed,
        )

    def _check_quality(self, summary: Mapping[str, float]) -> bool:
        min_fill = float(self.quality_gates.get("min_fill_rate", 0.0))
        min_trades = float(self.quality_gates.get("min_trades", 0.0))
        max_dd = float(self.quality_gates.get("max_drawdown", 1.0))
        if summary.get("fill_rate", 1.0) < min_fill:
            return False
        if summary.get("trades", 0.0) < min_trades:
            return False
        if abs(summary.get("max_drawdown", 0.0)) > max_dd:
            return False
        return True


def score_objective(summary: Mapping[str, float], objective: str) -> float:
    objective = objective.lower()
    if objective == "sharpe":
        return float(summary.get("sharpe", float("-inf")))
    if objective == "cagr":
        total_return = float(summary.get("total_return", 0.0))
        return total_return
    if objective == "cagr_fill":
        total_return = float(summary.get("total_return", 0.0))
        fill_rate = float(summary.get("fill_rate", 0.0))
        return total_return * fill_rate
    raise ConfigError(f"Unsupported objective: {objective}")
