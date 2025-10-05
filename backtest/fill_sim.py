"""Order fill simulation based on daily OHLC data."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date
from typing import Dict, Iterable, List, Optional, Tuple

import pandas as pd

from .engine_wrapper import SimulatedOrder
from .utils import (
    LiquidityModel,
    SlippageModel,
    apply_slippage,
    normalize_liquidity_model,
    normalize_slippage_model,
    tick_size,
)


@dataclass
class FillResult:
    order: SimulatedOrder
    executed_qty: int
    executed_price: float
    status: str
    notes: Dict[str, str]


class FillSimulator:
    def __init__(self, liquidity_model: LiquidityModel, slippage_model: SlippageModel, *, lot_size: int = 100):
        self.liquidity_model = liquidity_model
        self.slippage_model = slippage_model
        self.lot_size = lot_size

    def execute(self, day: date, orders: Iterable[SimulatedOrder], day_ohlc: pd.DataFrame) -> Tuple[List[FillResult], List[SimulatedOrder]]:
        ohlc_map = self._build_price_map(day_ohlc)
        fills: List[FillResult] = []
        residual_orders: List[SimulatedOrder] = []
        for order in orders:
            daily = ohlc_map.get(order.symbol)
            if daily is None:
                residual_orders.append(order)
                continue
            fill = self._fill_order(day, order, daily)
            if fill.executed_qty > 0:
                fills.append(fill)
            if fill.status not in {"FILLED", "CANCELED"}:
                residual_orders.append(order)
        return fills, residual_orders

    def _build_price_map(self, frame: pd.DataFrame) -> Dict[str, Dict[str, float]]:
        result: Dict[str, Dict[str, float]] = {}
        for _, row in frame.iterrows():
            symbol = str(row.get("symbol") or row.get("ticker") or row.get("Symbol"))
            if not symbol:
                continue
            result[symbol] = {
                "open": float(row.get("open")),
                "high": float(row.get("high")),
                "low": float(row.get("low")),
                "close": float(row.get("close")),
                "volume": float(row.get("volume", 0.0)),
                "ceiling": float(row.get("ceiling", 0.0)) if row.get("ceiling") is not None else None,
                "floor": float(row.get("floor", 0.0)) if row.get("floor") is not None else None,
            }
        return result

    def _fill_order(self, day: date, order: SimulatedOrder, daily: Dict[str, float]) -> FillResult:
        tick = tick_size(order.symbol, daily.get("close", 0.0))
        notes: Dict[str, str] = {}
        remaining = order.qty
        exec_price = None
        status = "OPEN"
        executed_qty = 0
        if order.type == "MARKET":
            exec_price = daily.get("open")
            if exec_price is None:
                exec_price = daily.get("close")
            exec_price = apply_slippage(exec_price, order.side, self.slippage_model, tick=tick)
            executed_qty = remaining
            status = "FILLED"
        else:
            price = order.limit_price
            if order.side == "BUY":
                if price is None:
                    price = daily.get("close")
                if price is None:
                    return FillResult(order, 0, 0.0, "OPEN", notes)
                low = daily.get("low")
                ceiling = daily.get("ceiling")
                if ceiling and price < ceiling and daily.get("high") == ceiling:
                    notes["ceiling_lock"] = "true"
                    return FillResult(order, 0, 0.0, "OPEN", notes)
                if low is not None and low <= price + 1e-9:
                    exec_price = min(price, daily.get("close"))
                    exec_price = apply_slippage(exec_price, order.side, self.slippage_model, tick=tick)
                    executed_qty = self._apply_liquidity(order, daily)
                    status = "PARTIAL" if executed_qty < order.qty else "FILLED"
            elif order.side == "SELL":
                if price is None:
                    price = daily.get("close")
                if price is None:
                    return FillResult(order, 0, 0.0, "OPEN", notes)
                high = daily.get("high")
                floor = daily.get("floor")
                if floor and price > floor and daily.get("low") == floor:
                    notes["floor_lock"] = "true"
                    return FillResult(order, 0, 0.0, "OPEN", notes)
                if high is not None and high >= price - 1e-9:
                    exec_price = max(price, daily.get("close"))
                    exec_price = apply_slippage(exec_price, order.side, self.slippage_model, tick=tick)
                    executed_qty = self._apply_liquidity(order, daily)
                    status = "PARTIAL" if executed_qty < order.qty else "FILLED"
        if exec_price is None:
            return FillResult(order, 0, 0.0, status, notes)
        return FillResult(order, executed_qty, exec_price, status, notes)

    def _apply_liquidity(self, order: SimulatedOrder, daily: Dict[str, float]) -> int:
        if self.liquidity_model.type == "touch_full":
            return order.qty
        volume = daily.get("volume") or 0.0
        if volume <= 0.0:
            return 0
        cap = self.liquidity_model.participation_cap or 1.0
        fill_qty = int(max(self.lot_size, (volume * cap) // self.lot_size * self.lot_size))
        return min(order.qty, fill_qty)


def execute_orders(
    day: date,
    orders: Iterable[SimulatedOrder],
    day_ohlc: pd.DataFrame,
    liquidity_model: Dict[str, float] | LiquidityModel,
    slippage_model: Dict[str, float] | SlippageModel,
    *,
    lot_size: int = 100,
) -> Tuple[List[FillResult], List[SimulatedOrder]]:
    liquidity = (
        liquidity_model
        if isinstance(liquidity_model, LiquidityModel)
        else normalize_liquidity_model(liquidity_model)
    )
    slippage = (
        slippage_model
        if isinstance(slippage_model, SlippageModel)
        else normalize_slippage_model(slippage_model)
    )
    simulator = FillSimulator(liquidity, slippage, lot_size=lot_size)
    return simulator.execute(day, list(orders), day_ohlc)
