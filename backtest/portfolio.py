"""Portfolio accounting for the backtest runner."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date, timedelta
from typing import Dict, Iterable, List, Optional

import pandas as pd

from .engine_wrapper import SimulatedOrder
from .fill_sim import FillResult
from .utils import ensure_timezone


@dataclass
class FeeConfig:
    commission_bps: float = 0.0
    tax_bps: float = 0.0
    borrow_cost_bps: float = 0.0

    @classmethod
    def from_dict(cls, data: Dict[str, float]) -> "FeeConfig":
        return cls(
            commission_bps=float(data.get("commission_bps", 0.0) or 0.0),
            tax_bps=float(data.get("tax_bps", 0.0) or 0.0),
            borrow_cost_bps=float(data.get("borrow_cost_bps", 0.0) or 0.0),
        )

    @property
    def total_bps(self) -> float:
        return self.commission_bps + self.tax_bps + self.borrow_cost_bps


@dataclass
class Position:
    symbol: str
    quantity: int
    avg_cost: float


@dataclass
class PortfolioState:
    date: date
    cash: float
    nav: float
    positions: Dict[str, Position] = field(default_factory=dict)
    accrued_fees: float = 0.0
    turnover: float = 0.0


@dataclass
class PendingOrder:
    order: SimulatedOrder
    created: date
    expires: Optional[date]


class Portfolio:
    def __init__(
        self,
        initial_cash: float,
        positions: Iterable[Dict[str, object]] | None = None,
        *,
        fee_config: FeeConfig | None = None,
    ):
        self.cash = float(initial_cash)
        self.positions: Dict[str, Position] = {}
        if positions:
            for row in positions:
                symbol = str(row.get("symbol") or row.get("ticker"))
                qty = int(row.get("qty") or row.get("quantity") or 0)
                if qty == 0:
                    continue
                avg_cost = float(row.get("avg_cost") or row.get("price") or 0.0)
                self.positions[symbol] = Position(symbol=symbol, quantity=qty, avg_cost=avg_cost)
        self.fee_config = fee_config or FeeConfig()
        self.pending_orders: List[PendingOrder] = []
        self.trade_log: List[Dict[str, object]] = []
        self.daily_nav: List[Dict[str, object]] = []

    # ------------------------------------------------------------------
    # Order lifecycle
    # ------------------------------------------------------------------

    def submit_orders(self, day: date, orders: Iterable[SimulatedOrder], default_ttl: int | None = None) -> None:
        for order in orders:
            ttl_days = order.ttl_days or default_ttl
            expires = None
            if ttl_days and ttl_days > 0:
                expires = day + timedelta(days=ttl_days)
            self.pending_orders.append(PendingOrder(order=order, created=day, expires=expires))

    def expire_orders(self, day: date) -> List[PendingOrder]:
        still_open: List[PendingOrder] = []
        expired: List[PendingOrder] = []
        for pending in self.pending_orders:
            if pending.expires and day >= pending.expires:
                expired.append(pending)
            else:
                still_open.append(pending)
        self.pending_orders = still_open
        return expired

    def pending_for_day(self) -> List[SimulatedOrder]:
        return [p.order for p in self.pending_orders]

    def apply_fills(self, day: date, fills: Iterable[FillResult]) -> None:
        to_remove: List[PendingOrder] = []
        for fill in fills:
            order = fill.order
            qty = fill.executed_qty
            price = fill.executed_price
            if qty <= 0:
                continue
            notional = qty * price
            fee = notional * (self.fee_config.total_bps / 10_000.0)
            if order.side == "BUY":
                self.cash -= notional + fee
                self._increase_position(order.symbol, qty, price)
                realized = 0.0
            else:
                self.cash += notional - fee
                realized = self._decrease_position(order.symbol, qty, price)
            self.trade_log.append(
                {
                    "date": ensure_timezone(day),
                    "symbol": order.symbol,
                    "side": order.side,
                    "qty": qty,
                    "price": price,
                    "notional": notional,
                    "fee": fee,
                    "realized_pnl": realized,
                }
            )
            for pending in self.pending_orders:
                if pending.order is order:
                    if fill.executed_qty >= pending.order.qty:
                        to_remove.append(pending)
                    else:
                        pending.order = SimulatedOrder(
                            symbol=pending.order.symbol,
                            side=pending.order.side,
                            type=pending.order.type,
                            qty=pending.order.qty - fill.executed_qty,
                            limit_price=pending.order.limit_price,
                            ttl_days=pending.order.ttl_days,
                            meta=dict(pending.order.meta),
                        )
                    break
        self.pending_orders = [p for p in self.pending_orders if p not in to_remove]

    # ------------------------------------------------------------------
    # NAV computation
    # ------------------------------------------------------------------

    def mark_to_market(self, day: date, prices: pd.DataFrame) -> PortfolioState:
        price_map = {str(row["symbol"]): float(row["close"]) for _, row in prices.iterrows()}
        nav = self.cash
        turnover = 0.0
        for pos in list(self.positions.values()):
            px = price_map.get(pos.symbol, pos.avg_cost)
            nav += pos.quantity * px
        state = PortfolioState(date=day, cash=self.cash, nav=nav, positions=dict(self.positions), turnover=turnover)
        self.daily_nav.append({"date": day, "nav": nav, "cash": self.cash})
        return state

    # ------------------------------------------------------------------
    # Position management helpers
    # ------------------------------------------------------------------

    def _increase_position(self, symbol: str, qty: int, price: float) -> None:
        pos = self.positions.get(symbol)
        if pos is None:
            self.positions[symbol] = Position(symbol=symbol, quantity=qty, avg_cost=price)
            return
        total_qty = pos.quantity + qty
        if total_qty <= 0:
            self.positions.pop(symbol, None)
            return
        new_cost = ((pos.avg_cost * pos.quantity) + (price * qty)) / total_qty
        pos.quantity = total_qty
        pos.avg_cost = new_cost

    def _decrease_position(self, symbol: str, qty: int, price: float) -> float:
        pos = self.positions.get(symbol)
        if pos is None:
            return 0.0
        sell_qty = min(qty, pos.quantity)
        pnl = sell_qty * (price - pos.avg_cost)
        remaining = pos.quantity - sell_qty
        if remaining <= 0:
            self.positions.pop(symbol, None)
        else:
            pos.quantity = remaining
        return pnl

    # ------------------------------------------------------------------
    # Diagnostics
    # ------------------------------------------------------------------

    def to_frame(self) -> pd.DataFrame:
        records = []
        for state in self.daily_nav:
            records.append({"date": state["date"], "nav": state["nav"], "cash": state["cash"]})
        return pd.DataFrame.from_records(records)

    def trades_frame(self) -> pd.DataFrame:
        return pd.DataFrame.from_records(self.trade_log)
