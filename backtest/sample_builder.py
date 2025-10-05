"""Sample order builder used for CI smoke tests."""

from __future__ import annotations

from datetime import date
from typing import Any, Dict, Iterable, List

import pandas as pd

from .engine_wrapper import SimulatedOrder
from .utils import tick_size


def _select_symbols(context: Dict[str, Any], min_priority: float) -> List[str]:
    watchlist = context.get("watchlist")
    if isinstance(watchlist, pd.DataFrame) and not watchlist.empty:
        frame = watchlist.copy()
        if "priority" in frame.columns:
            frame = frame[frame["priority"] >= min_priority]
        if "symbol" in frame.columns and not frame.empty:
            return frame["symbol"].astype(str).tolist()
    prices = context.get("prices")
    if isinstance(prices, pd.DataFrame) and not prices.empty:
        return sorted(prices["symbol"].astype(str).unique().tolist())
    return []


def _determine_side(day: date, trading_days: Iterable[date]) -> str:
    try:
        days = list(trading_days)
        idx = days.index(day)
    except ValueError:
        return "BUY"
    return "SELL" if idx % 2 == 1 else "BUY"


def sample_builder(day: date, context: Dict[str, Any], flags: Dict[str, Any] | None = None):
    """Build deterministic limit orders for CI backtests."""

    prices = context.get("prices")
    if not isinstance(prices, pd.DataFrame) or prices.empty:
        return []
    engine_cfg = dict(context.get("engine_config") or {})
    min_priority = float((engine_cfg.get("watchlist") or {}).get("min_priority", 0.0))
    symbols = _select_symbols(context, min_priority)
    if not symbols:
        return []
    initial_cash = float(engine_cfg.get("initial_cash", 1_000_000.0))
    buy_budget_frac = float(engine_cfg.get("buy_budget_frac", 0.1))
    entry_offset = int(engine_cfg.get("entry_price_offset_ticks", 0))
    safety_spread = int(engine_cfg.get("limit_safety_spread_ticks", 1))
    ttl_days = int(engine_cfg.get("ttl_days", 3))
    lot_size = int(engine_cfg.get("lot_size", 100))
    side = _determine_side(day, context.get("trading_days") or [])
    orders: List[SimulatedOrder] = []
    for symbol in symbols:
        row = prices[prices["symbol"] == symbol]
        if row.empty:
            continue
        close_price = float(row.iloc[0]["close"])
        if close_price <= 0:
            continue
        raw_qty = (initial_cash * buy_budget_frac) / max(close_price, 1.0)
        lots = max(1, int(raw_qty // lot_size))
        qty = lots * lot_size
        if qty <= 0:
            continue
        tick = tick_size(symbol, close_price)
        if side == "BUY":
            limit_price = max(0.0, close_price - (entry_offset * tick))
        else:
            limit_price = close_price + (safety_spread * tick)
        orders.append(
            SimulatedOrder(
                symbol=symbol,
                side=side,
                type="LIMIT",
                qty=qty,
                limit_price=limit_price,
                ttl_days=ttl_days,
                meta={"builder": "sample", "side_pattern": side.lower()},
            )
        )
    return orders
